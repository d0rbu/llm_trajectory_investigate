import os
import json
import yaml
import torch as th
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from typing import Iterable, Mapping
from more_itertools import batched
from core.prompts import get_prompts


def get_model_trajectories(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Iterable[Mapping],
    batch_size: int = 16,
    only_final_token: bool = True,
) -> Mapping[str, Iterable]:
    batched_prompts = batched(prompts, batch_size)
    expected_outputs = [tokenizer.encode(prompt["answer"])[0] for prompt in prompts]
    batched_expected_outputs = batched(expected_outputs, batch_size)

    model_trajectories = {
        "trajectories": None,
        "corrects": None,
        "prompts": []
    }
    with th.no_grad():
        for prompts, expected_outputs in zip(batched_prompts, batched_expected_outputs):
            prepared_prompts = tokenizer([prompt["prompt"] for prompt in prompts])

            model_output = model(**prepared_prompts, output_hidden_states = True)
            chosen_outputs = th.argmax(model_output.logits[:, -1], dim=-1)
            corrects = chosen_outputs == th.tensor(expected_outputs)

            if only_final_token:
                trajectories = [hidden_state[:, -1].unsqueeze(1) for hidden_state in model_output.hidden_states]  # (L, B, 1, D)
            else:
                trajectories = model_output.hidden_states  # (L, B, T, D)

            trajectories = th.stack(trajectories, dim = 2)  # (B, T, L, D)
            
            if model_trajectories["trajectories"] is None:
                model_trajectories["trajectories"] = trajectories
            else:
                model_trajectories["trajectories"] = th.cat(model_trajectories["trajectories"], trajectories, dim=0)
            
            if model_trajectories["corrects"] is None:
                model_trajectories["corrects"] = corrects
            else:
                model_trajectories["corrects"] = th.cat(model_trajectories["corrects"], corrects, dim=0)

            model_trajectories["prompts"].extend(prompts)

    return model_trajectories

def get_trajectories(
    model_names: Iterable[str] = ["meta-llama/Llama-2-7b"],
    output_dir: os.PathLike | str | None = None,
    prompt_types: Iterable[str] = [],
    batch_size: int = 16,
    only_final_token: bool = True,
) -> Iterable[Mapping]:
    prompts, prompts_hash = get_prompts(prompt_types, return_hash = True)
    output_paths = [os.path.join(output_dir, str(prompts_hash), f"{model_name}.pt") for model_name in model_names]
    all_trajectories = {}

    for model_name, output_path in zip(model_names, output_paths):
        if os.path.exists(output_path):
            try:
                print("Found cached trajectory, loading...")
                all_trajectories[model_name] = th.load(output_path)

                continue
            except Exception as e:
                print(f"Issue loading cached trajectory: {e}")

        print("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_trajectories = get_model_trajectories(
            model = model,
            tokenizer = tokenizer,
            prompts = prompts,
            batch_size = batch_size,
            only_final_token = only_final_token,
        )
        all_trajectories[model_name] = model_trajectories

        if output_dir:
            th.save(model_trajectories, output_path)

    return all_trajectories


CONFIG_DIR = os.path.join("experiment", "configs")

if __name__ == "__main__":
    experiment_config_path = os.path.join(CONFIG_DIR, "main.yaml")
    if os.path.exists(experiment_config_path):
        with open(experiment_config_path, "r", encoding="utf-8") as experiment_config_file:
            experiment_config = yaml.safe_load(experiment_config_file)
    else:
        experiment_config = {}

    get_trajectories(output_dir="outputs", **experiment_config)
