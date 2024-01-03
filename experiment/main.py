import os
import yaml
import torch as th
from core.prompts import get_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Iterable


def get_full_trajectories(
    model_name: str = "meta-llama/Llama-2-7b",
    output_dir: os.PathLike | str | None = None,
    return_trajs: bool = True,
    prompt_types: Iterable[str] = [],
) -> th.Tensor | None:
    assert return_trajs or output_dir, "where to return the trajectories? please specify return_trajs or output_dir"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = get_prompts(prompt_types)

CONFIG_DIR = os.path.join("experiment", "configs")


if __name__ == "__main__":
    experiment_config_path = os.path.join(CONFIG_DIR, "main.yaml")
    if os.path.exists(experiment_config_path):
        with open(experiment_config_path, "r", encoding="utf-8") as experiment_config_file:
            experiment_config = yaml.safe_load(experiment_config_file)
    else:
        experiment_config = {}

    get_full_trajectories(output_dir="output", return_trajs=False, **experiment_config)
