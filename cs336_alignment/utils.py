from collections.abc import Callable
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    num_samples = len(prompt_strs)

    prompt_tokens = tokenizer(prompt_strs)['input_ids']
    output_tokens = tokenizer(output_strs)['input_ids']
    concat_tokens = [prompt + output for prompt, output in zip(prompt_tokens, output_tokens)]

    max_len = max(len(ids) for ids in concat_tokens)
    input_ids = torch.zeros(num_samples, max_len-1, dtype=torch.long)
    labels = torch.zeros(num_samples, max_len-1, dtype=torch.long)
    response_mask = torch.zeros(num_samples, max_len-1, dtype=torch.bool)
    for i in range(num_samples):
        concat = concat_tokens[i]
        len_concat = len(concat)
        len_prompt = len(prompt_tokens[i])
        if len(concat) == max_len:
            input_ids[i, : len_concat - 1] = torch.tensor(concat[:-1])
        else:
            input_ids[i, : len_concat] = torch.tensor(concat)
        labels[i, : len_concat - 1] = torch.tensor(concat[1:])
        response_mask[i, len_prompt - 1: len_concat - 1] = True

    return {
        'input_ids': input_ids,
        'labels': labels,
        'response_mask': response_mask
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Using logsumexp keeps everything in log-space, so nothing ever overflows or collapses to zero before you take the log.
    # If you do softmax -> log, tiny probs become 0 and log(0) blows up to -inf, destroying the entropy calculation.
    # See https://discuss.pytorch.org/t/justification-for-logsoftmax-being-better-than-log-softmax/140130
    with torch.no_grad():
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True) # (batch_size, sequence_length, vocab_size)
        return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1) # (batch_size, sequence_length)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    model = model.cuda()
    input_ids = input_ids.cuda()
    labels = labels.cuda()

    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=2).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    B, L = labels.shape
    if return_token_entropy: 
        loss = compute_entropy(logits)

    results = {'log_probs': log_probs}
    if return_token_entropy:
        results['token_entropy'] = loss
    return results


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)
    total_rewards = []
    sum_format_rewards = 0
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        rewards = reward_fn(response, gt)
        total_rewards.append(rewards['reward'])
        sum_format_rewards += rewards['format_reward']
    return torch.tensor(total_rewards), {'mean_format_rewards': sum_format_rewards / len(rollout_responses)}


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    assert raw_rewards.dim() < 2
    assert len(raw_rewards) % group_size == 0
    if baseline != 'mean' or advantage_normalizer != 'std':
        raise NotImplementedError('Support only mean-std normalization.')

    norm_rewards = raw_rewards[:]
    groups = raw_rewards.split(group_size)
    group_mean_stds = [(torch.mean(g), torch.std(g)) for g in groups]
    for i, (mu, std) in zip(range(0, len(raw_rewards), group_size), group_mean_stds):
        norm_rewards[i:i+group_size] = (raw_rewards[i:i+group_size] - mu) / (std + advantage_eps)
    return norm_rewards, {'mean_rewards': norm_rewards.mean(), 'min_rewards': norm_rewards.min(), 'max_rewards': norm_rewards.max()}


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError("importance_reweighting_method supports only `none`.")
    return -raw_rewards_or_advantages * policy_log_probs, {}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    per_token_policy_gradient_loss = per_token_policy_gradient_loss * mask
    losses = per_token_policy_gradient_loss.sum(1) / mask.sum(1)
    return losses.mean()


if __name__ == '__main__':
    model, tokenizer = get_model_and_tokenizer('./models/olmo', 'cuda')
    batch = tokenize_prompt_and_output(['hello how are you', "what's the capital of France"], ['i am good', 'the capital of France is Paris.'], tokenizer)
    get_response_log_probs(model, batch['input_ids'], batch['labels'], return_token_entropy=True)
