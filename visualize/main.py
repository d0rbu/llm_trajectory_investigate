import os
import pathlib
import torch as th
from core.prompts import get_prompts


def visualize(
    output_dir = pathlib.Path("outputs"),
    prompt_types = [["knowledge", "capitals"], ["reasoning", "math"]],
) -> None:
    for prompt_type in prompt_types:
        prompts, prompts_hash = get_prompts(prompt_type, return_hash=True)
        num_prompts = len(prompts)
        prompts_results_dir = os.path.join(output_dir, prompts_hash)

        if not os.path.exists(prompts_results_dir) or not os.path.isdir(prompts_results_dir)
            print(f"Results directory for prompt type {prompt_type} does not exist")
            continue

        all_trajectories = []
        all_corrects = []
        all_models = []
        for results_path in pathlib.Path(prompts_results_dir).iterdir():
            model_trajectories = th.load(results_path)
            all_trajectories.append(model_trajectories["trajectories"])
            all_corrects.append(model_trajectories["corrects"])
            all_models.append(results_path.stem)
        
        all_trajectories = th.cat(all_trajectories, dim=0)
        all_corrects = th.cat(all_corrects, dim=0)

        # TODO: T-SNE/PCA trajectories, 2d plot them and connect them by num_prompts


if __name__ == "__main__":
    visualize()
