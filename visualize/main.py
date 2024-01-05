import os
import pathlib
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from core.prompts import get_prompts
from more_itertools import batched
from tsne_torch import TorchTSNE as tsne
from typing import Iterable


class ProjectionMethods:
    def __init__(
        self,
        dim: int = 2
    ) -> None:
        self.projection_dim = dim

    def pca(self, trajectories: th.Tensor) -> th.Tensor:
        flattened_trajs = trajectories.view(-1, trajectories.shape[-1])

        U, S, V = th.pca_lowrank(flattened_trajs, center=False)
        projected_trajs = flattened_trajs @ V[:, :self.projection_dim]

        return projected_trajs

    def tsne(self, trajectories: th.Tensor, perplexity: int = 30, n_iter: int = 1000) -> th.Tensor:
        flattened_trajs = trajectories.view(-1, trajectories.shape[-1])

        return tsne(n_components=self.projection_dim, perplexity=perplexity, n_iter = n_iter).fit_transform(flattened_trajs)

def visualize(
    output_dir: str | os.PathLike = pathlib.Path("outputs"),
    prompt_types: Iterable[Iterable[str]] = [["knowledge", "capitals"], ["reasoning", "math"]],
    projection_method: str = "pca",
    color_map_name: str = "hsv",
    incorrect_answer_modifier: float = 0.5,  # make incorrect trajectories half as bright
    token_distance_modifier: float = 0.5,  # make the second from last token half as opaque, third from last a quarter as opaque, etc.
) -> None:
    projection_methods = ProjectionMethods(dim=2)
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

        projection_fn = getattr(projection_methods, projection_method)
        projected_trajectories = projection_fn(all_trajectories)

        cmap = cmaps[color_map_name]
        color_points = th.linspace(0, 1, len(all_models) + 1)[:-1]
        colors = [cmap(point) for point in color_points]

        for model_name, model_trajectories, model_corrects, color in zip(
            all_models,
            batched(projected_trajectories, num_prompts),
            batched(all_corrects, num_prompts),
            colors
        ):
            for token_trajectories, correct in zip(model_trajectories, model_corrects):
                adjusted_color = mpl.colors.rgb_to_hsv(color)
                if not correct:
                    adjusted_color[-1] *= incorrect_answer_modifier
                
                adjusted_color = mpl.colors.hsv_to_rgb(adjusted_color)
                for token_idx, trajectory in enumerate(token_trajectories):
                    distance_from_end = len(token_trajectories) - 1 - token_idx
                    alpha = token_distance_modifier ** distance_from_end
                    plt.plot(trajectory[:, 0], trajectory[:, 1], color=adjusted_color, alpha=alpha)
    
    plt.title(f"token trajectory visualization")
    plt.show()


if __name__ == "__main__":
    visualize()
