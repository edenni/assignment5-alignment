import json
import random
from pathlib import Path

import torch

from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.utils import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer

root_dir = Path(__file__).parent.parent
model_dir = root_dir / "models/olmo"
# hypers
n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0


# load data
with open('data/gsm8k/train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
with open('data/gsm8k/test.jsonl', 'r') as f:
    val_data = [json.loads(line) for line in f]
train_data = train_data[:n_train_examples]
val_data = val_data[:n_val_examples]


# start vllm
vllm = VLLMServer(
    model_id=str(model_dir),
    gpu=0,
    gpu_memory_utilization=0.25,
)
vllm.start()
vllm.init_weight_sync("cuda:0")
base_sampling_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 42
}
completions = vllm.generate_completions(prompts=["start"], sampling_params=base_sampling_params)
response = completions[0].text.strip()
if response:
    print("vllm started successfully!")


# load model
policy, tokenizer = get_model_and_tokenizer(model_dir, "cuda")
optimizer = torch.optim.AdamW(
    policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)


for step in range(num_rollout_steps):
    vllm.sync_policy_weights(policy)
    break