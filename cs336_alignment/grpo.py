from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from cs336_alignment.utils import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    compute_rollout_rewards,
    get_response_log_probs,
    tokenize_prompt_and_output,
)


def clip_grad_norm(params, max_norm: float = 1.0, eps: float = 1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    
    norms = [g.detach().norm(2) for g in grads]
    total_norm = torch.stack(norms).norm(2)

    clip_coef = max_norm / (total_norm + eps)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for g in grads:
        g.mul_(clip_coef_clamped)
    
    return total_norm


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = batch['input_ids']
    labels = batch['labels']
    response_mask = batch['response_mask'].cuda()

    raw_rewards, _ = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    normalized_rewards, _ = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    normalized_rewards = normalized_rewards.cuda()

    # Backward in mini-batch
    batch_loss = 0
    mbs = len(input_ids) // gradient_accumulation_steps
    for i in range(0, len(input_ids), mbs):
        inputs_mb = input_ids[i: i+mbs]
        labels_mb = labels[i: i+mbs]
        mask_mb = response_mask[i: i+mbs]
        rewards = normalized_rewards[i: i+mbs]

        log_probs_dict = get_response_log_probs(model, inputs_mb, labels_mb, False)
        log_probs = log_probs_dict['log_probs']

        per_token_loss, _ = compute_policy_gradient_loss(
            rewards.unsqueeze(1),
            log_probs,
            importance_reweighting_method,
            old_log_probs,
            cliprange,
            mask_mb
        )
        loss = aggregate_loss_across_microbatch(per_token_loss, mask_mb, loss_normalization, normalization_constant)
        loss *= len(inputs_mb) / len(input_ids)
        batch_loss += loss.item()
        loss.backward()
    clip_grad_norm(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return torch.tensor(batch_loss), {}
