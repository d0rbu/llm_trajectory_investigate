import os
import pathlib
import torch as th
import matplotlib.pyplot as plt
from core.prompts import get_prompts
from more_itertools import batched
from typing import Iterable


class ProjectionMethods:
    @staticmethod
    def pca(trajectories: th.Tensor) -> th.Tensor:
        pass

    @staticmethod
    def tsne(trajectories: th.Tensor) -> th.Tensor:
        pass

def visualize(
    output_dir: str | os.PathLike = pathlib.Path("outputs"),
    prompt_types: Iterable[Iterable[str]] = [["knowledge", "capitals"], ["reasoning", "math"]],
    projection_method: str = "pca"
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

        projection_fn = getattr(ProjectionMethods, projection_method)
        projected_trajectories = projection_fn(all_trajectories)

        for model_name, model_trajectories, model_corrects in zip(model_name, batched(projected_trajectories, num_prompts), batched(all_corrects, num_prompts)):
            for token_trajectories, correct in zip(model_trajectories, model_corrects):
                for trajectory in token_trajectories:
                    plt.plot(trajectory[:, 0], trajectory[:, 1])  # TODO: light/dark based on correct, alpha based on token distance from end


if __name__ == "__main__":
    visualize()
