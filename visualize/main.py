import os
import pathlib
import glob
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

    def pca(self, trajectories: th.Tensor) -> tuple[th.Tensor, th.Tensor | None]:
        flattened_trajs = trajectories.view(-1, trajectories.shape[-1])

        U, S, V = th.pca_lowrank(flattened_trajs, center=False)
        total_variance = S.sum()
        explained_variances = S / total_variance
        projected_trajs = flattened_trajs @ V[:, :self.projection_dim]
        projected_trajs = projected_trajs.view(*trajectories.shape)

        print(f"explained variances: {explained_variances}")
        print(f"explained variances cumsum: {explained_variances.cumsum()}")

        return projected_trajs, explained_variances[:self.projection_dim]

    def tsne(self, trajectories: th.Tensor, perplexity: int = 30, n_iter: int = 1000) -> tuple[th.Tensor, th.Tensor | None]:
        flattened_trajs = trajectories.view(-1, trajectories.shape[-1])

        projected_trajs = tsne(n_components=self.projection_dim, perplexity=perplexity, n_iter = n_iter).fit_transform(flattened_trajs)
        projected_trajs = projected_trajs.view(*trajectories.shape)

        return projected_trajs, None


def plot_token_trajectories(
    token_trajectories: th.Tensor,
    color: tuple[float, float, float],
    model_name: str,
    token_distance_modifier: float = 0.5,
    token_average_modifier: float = 0.5,
) -> None:
    for token_idx, trajectory in enumerate(token_trajectories):
        distance_from_end = len(token_trajectories) - 1 - token_idx
        alpha = token_average_modifier * (token_distance_modifier ** distance_from_end)
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=alpha, label=model_name)

def visualize(
    output_dir: str | os.PathLike = pathlib.Path("outputs"),
    prompt_types: Iterable[Iterable[str]] = [["knowledge", "capitals"], ["reasoning", "math"]],
    projection_method: str = "pca",
    reorder: bool = False,  # whether to permute the activations to try and be more similar
    color_map_name: str = "hsv",
    incorrect_answer_modifier: float = 0.5,  # make incorrect trajectories half as bright
    token_distance_modifier: float = 0.5,  # make the second from last token half as opaque, third from last a quarter as opaque, etc.
    token_average_modifier: float = 0.5,  # make each trajectory half as opaque, show the average at full opacity
) -> None:
    projection_methods = ProjectionMethods(dim=2)
    for prompt_type in prompt_types:
        prompts, prompts_hash = get_prompts(prompt_type, return_hash=True)
        num_prompts = len(prompts)
        prompts_results_dir = os.path.join(output_dir, prompts_hash)

        if not os.path.exists(prompts_results_dir) or not os.path.isdir(prompts_results_dir):
            print(f"Results directory for prompt type {prompt_type} does not exist")
            continue

        all_trajectories = []
        all_corrects = []
        all_models = []
        for results_path in glob.iglob(os.path.join(prompts_results_dir, "*.pt"), recursive=True):
            model_trajectories = th.load(results_path)
            all_trajectories.append(model_trajectories["trajectories"])
            all_corrects.append(model_trajectories["corrects"])
            all_models.append(results_path.stem)
        
        all_trajectories = th.cat(all_trajectories, dim=0)
        if reorder:
            all_trajectories = all_trajectories.sort(dim=-1, descending=True)
        all_corrects = th.cat(all_corrects, dim=0)

        projection_fn = getattr(projection_methods, projection_method)
        projected_trajectories, explained_variances = projection_fn(all_trajectories)

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
                plot_token_trajectories(
                    token_trajectories,
                    adjusted_color,
                    token_distance_modifier,
                    token_average_modifier,
                    model_name,
                )

            average_trajectories = model_trajectories.mean(dim=0)  # (T, L, 2)
            plot_token_trajectories(
                average_trajectories,
                adjusted_color,
                token_distance_modifier,
                token_average_modifier,
                model_name,
            )

        plt.title(f"token trajectory visualization")
        if explained_variances is not None:
            plt.xlabel(f"explained variance: {explained_variances[0]}")
            plt.ylabel(f"explained variance: {explained_variances[1]}")
        plt.legend(handles=colors)
        plt.show()


if __name__ == "__main__":
    visualize()
