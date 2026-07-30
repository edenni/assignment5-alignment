import json
from pathlib import Path

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer

prompt_templates = {}
for prompt_file in Path('cs336_alignment/prompts').glob('*.prompt'):
    with open(prompt_file, 'r') as f:
        prompt_templates[prompt_file.stem] = f.read()

with open('data/gsm8k/test.jsonl', 'r') as f:
    test_data = [json.loads(line) for line in f]

base_sampling_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 42
}

server = VLLMServer(
    model_id="./models/olmo",
    gpu=0,
    gpu_memory_utilization=0.25,
)
server.start()

total_grades = {prompt_name: {
    'format_reward': 0,
    'answer_reward': 0,
    'reward': 0,
} for prompt_name in prompt_templates}

max_samples = 20
for i, data in enumerate(test_data):
    question = data['question']
    answer = data['answer']
    gt = answer.split('####')[-1].strip()

    print("-" * 50)
    print(f"Question {i + 1}: {question}")
    print(f"Answer: {answer}")
    print(f"Ground Truth: {gt}")
    for prompt_name, prompt_template in prompt_templates.items():
        prompt = prompt_template.format(question=question)
        if prompt_name.startswith("r1_zero"):
            sampling_params = base_sampling_params.copy()
            sampling_params['stop'] = ["</answer>"]
            sampling_params['include_stop_str_in_output'] = True
            reward_fn = r1_zero_reward_fn
        else:
            sampling_params = base_sampling_params.copy()
            reward_fn = question_only_reward_fn
        completions = server.generate_completions(prompts=[prompt], sampling_params=sampling_params)
        response = completions[0].text.strip()
        grade = reward_fn(response, gt)
        print(f"Completion ({prompt_name}): {response}")
        print(f"Grade ({prompt_name}): {grade}")
        for key in total_grades[prompt_name]:
            total_grades[prompt_name][key] += grade[key]
    if i + 1 >= max_samples:
        break

print("Total Grades:")
for prompt_name, grades in total_grades.items():
    print(f"{prompt_name}: {grades}")