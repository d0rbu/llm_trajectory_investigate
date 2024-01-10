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
from typing import Iterable, Mapping, Any


class ProjectionMethods:
    def __init__(
        self,
        dim: int = 2
    ) -> None:
        self.projection_dim = dim
    
    def project(self, trajectories: Iterable[th.Tensor], method: str = "pca", **kwargs: dict[str, Any]) -> tuple[Iterable[th.Tensor], th.Tensor | None]:
        flattened_trajs = [model_trajectories.view(-1, model_trajectories.shape[-1]) for model_trajectories in trajectories]
        flattened_traj_lens = [flattened_traj.shape[0] for flattened_traj in flattened_trajs]
        flattened_trajs = th.cat(flattened_trajs, dim=0)

        projection_fn = getattr(self, method)
        flattened_projected_trajs, explained_variances = projection_fn(flattened_trajs, **kwargs)

        projected_trajs = []
        start_idx = 0
        for traj_len in flattened_traj_lens:
            end_idx = start_idx + traj_len
            projected_trajs.append(flattened_projected_trajs[start_idx:end_idx])
            start_idx = end_idx

        projected_trajs = [
            projected_model_trajs.view(*model_trajectories.shape[:-1], self.projection_dim)
            for projected_model_trajs, model_trajectories in zip(projected_trajs, trajectories)
        ]

        return projected_trajs, explained_variances

    def pca(self, trajectories: th.Tensor) -> tuple[th.Tensor, th.Tensor | None]:
        U, S, V = th.pca_lowrank(trajectories, q=min(*trajectories.shape), center=False)
        total_variance = S.sum()
        explained_variances = S / total_variance
        projected_trajs = trajectories @ V[:, :self.projection_dim]

        print(f"explained variances: {explained_variances}")
        print(f"explained variances cumsum: {explained_variances.cumsum(dim=0)}")

        return projected_trajs, explained_variances[:self.projection_dim]

    def tsne(self, trajectories: th.Tensor, perplexity: int = 30, n_iter: int = 1000) -> tuple[th.Tensor, th.Tensor | None]:
        projected_trajs = tsne(n_components=self.projection_dim, perplexity=perplexity, n_iter = n_iter).fit_transform(trajectories)

        return projected_trajs, None


def batch_by_trajectory_dim(
    trajectories: Iterable[th.Tensor],
    corrects: Iterable[th.Tensor],
    model_names: Iterable[str],
    separate_models: bool = True,
    dim: int = -1,
) -> Mapping[int, Mapping]:
    batched_data = {}
    for trajectory, correct, model_name in zip(trajectories, corrects, model_names):
        dim_size = trajectory.shape[dim]
        if separate_models:
            dim_size = (dim_size, model_name)

        if dim_size not in batched_data:
            batched_data[dim_size] = {
                "trajectories": [],
                "corrects": [],
                "model_names": [],
            }
        
        batched_data[dim_size]["trajectories"].append(trajectory)
        batched_data[dim_size]["corrects"].append(correct)
        batched_data[dim_size]["model_names"].append(model_name)

    return batched_data

def plot_token_trajectories(
    token_trajectories: th.Tensor,
    color: tuple[float, float, float],
    model_name: str,
    token_distance_modifier: float = 0.5,
    token_average_modifier: float = 0.5,
) -> None:
    handles = []

    for token_idx, trajectory in enumerate(token_trajectories):
        distance_from_end = len(token_trajectories) - 1 - token_idx
        alpha = token_average_modifier * (token_distance_modifier ** distance_from_end)
        handle = plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, marker='o', alpha=alpha, label=model_name)
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], color=color, marker='s', alpha=alpha)  # make the start of the trajectory a square
        handles.extend(handle)
    
    return handles

def visualize(
    output_dir: str | os.PathLike = pathlib.Path("outputs"),
    prompt_types: Iterable[Iterable[str]] = [[]],
    projection_method: str = "pca",
    reorder: bool = False,  # whether to permute the activations to try and be more similar
    separate_models: bool = True,  # whether to plot each model separately
    color_map_name: str = "hsv",
    incorrect_answer_modifier: float = 0.5,  # make incorrect trajectories half as bright
    token_distance_modifier: float = 0.5,  # make the second from last token half as opaque, third from last a quarter as opaque, etc.
    token_average_modifier: float = 0.5,  # make each trajectory half as opaque, show the average at full opacity
) -> None:
    projection_methods = ProjectionMethods(dim=2)
    for prompt_type in prompt_types:
        _, prompts_hash = get_prompts(prompt_type, return_hash=True)
        prompts_results_dir = os.path.join(output_dir, str(prompts_hash))

        if not os.path.exists(prompts_results_dir) or not os.path.isdir(prompts_results_dir):
            print(f"Results directory for prompt type {prompt_type} does not exist")
            continue

        all_trajectories = []
        all_corrects = []
        all_models = []
        for results_path in glob.iglob(os.path.join(prompts_results_dir, "*", "*.pt"), recursive=True):
            model_trajectories = th.load(results_path)
            all_trajectories.append(model_trajectories["trajectories"])
            all_corrects.append(model_trajectories["corrects"])
            results_path = pathlib.Path(results_path)
            full_model_name = f"{results_path.parent.name}/{results_path.stem}"
            all_models.append(full_model_name)
        
        batched_data = batch_by_trajectory_dim(all_trajectories, all_corrects, all_models, separate_models, dim=-1)
        if reorder:
            for data in batched_data.values():
                for trajectories in data["trajectories"]:
                    trajectories.sort(dim=-1, descending=True)

        for dim, data in batched_data.items():
            trajectories, corrects, model_names = data["trajectories"], data["corrects"], data["model_names"]

            projected_trajectories, explained_variances = projection_methods.project(trajectories, projection_method)

            cmap = cmaps[color_map_name]
            color_points = th.linspace(0, 1, len(model_names) + 1)[:-1]
            colors = [cmap(point)[:-1] for point in color_points]

            model_handles = []

            for model_name, model_trajectories, model_corrects, color in zip(
                model_names,
                projected_trajectories,
                corrects,
                colors
            ):
                for token_trajectories, correct in zip(model_trajectories, model_corrects):
                    adjusted_color = mpl.colors.rgb_to_hsv(color)
                    if not correct:
                        adjusted_color[-1] *= incorrect_answer_modifier

                    adjusted_color = mpl.colors.hsv_to_rgb(adjusted_color)
                    traj_handle = plot_token_trajectories(
                        token_trajectories,
                        adjusted_color,
                        model_name,
                        token_distance_modifier,
                        token_average_modifier,
                    )

                # average_trajectories = model_trajectories.mean(dim=0)  # (T, L, 2)
                # plot_token_trajectories(
                #     average_trajectories,
                #     adjusted_color,
                #     model_name,
                #     token_distance_modifier,
                #     token_average_modifier,
                # )

                model_handles.append(traj_handle[0])

            plt.title(f"token trajectories with {projection_method}, dim={dim}")
            if explained_variances is not None:
                plt.xlabel(f"explained variance: {explained_variances[0]}")
                plt.ylabel(f"explained variance: {explained_variances[1]}")
            plt.legend(handles=model_handles)
            plt.show()


if __name__ == "__main__":
    visualize()
