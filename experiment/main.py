import os
import json
import yaml
import torch as th
import torch.nn as nn
from core.prompts import get_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from hashlib import md5
from typing import Iterable, Mapping


def get_full_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Mapping[str, str | Iterable]
) -> th.Tensor:
    pass

def get_full_trajectories(
    model_name: str = "meta-llama/Llama-2-7b",
    output_dir: os.PathLike | str | None = None,
    prompt_types: Iterable[str] = [],
) -> Iterable[Mapping]:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = get_prompts(prompt_types)
    prompts_hash = int(md5(str.encode(json.dumps(prompts, sort_keys=True))).hexdigest(), 16)

    trajectories = [
        {
            "trajectory": get_full_trajectory(model, tokenizer, prompt),
            "prompt": prompt,
        } for prompt in prompts
    ]

    if output_dir:
        output_path = os.path.join(output_dir, model_name, prompts_hash)
        with open(output_path, "r", encoding="utf-8") as output_file:
            json.dump(trajectories, output_file)

    return trajectories


CONFIG_DIR = os.path.join("experiment", "configs")


if __name__ == "__main__":
    experiment_config_path = os.path.join(CONFIG_DIR, "main.yaml")
    if os.path.exists(experiment_config_path):
        with open(experiment_config_path, "r", encoding="utf-8") as experiment_config_file:
            experiment_config = yaml.safe_load(experiment_config_file)
    else:
        experiment_config = {}

    get_full_trajectories(output_dir="output", **experiment_config)
