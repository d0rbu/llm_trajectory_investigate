import os
import pathlib
import glob
import torch as th
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from core.prompts import get_prompts
from itertools import product
from more_itertools import batched
from tsne_torch import TorchTSNE as tsne
from typing import Iterable, Mapping, Any


def flatten_tokens(trajectories: th.Tensor) -> th.Tensor:
    return trajectories.reshape(-1, trajectories.shape[-2], trajectories.shape[-1])

def get_metric_between_vectors(
    first_vectors: th.Tensor,  # (B, L, D)
    second_vectors: th.Tensor,  # (B, L', D)
    metric: str,
) -> th.Tensor:  # (B, L, L')
    if metric == "cosine":
        return th.nn.functional.cosine_similarity(first_vectors[:, :, None], second_vectors[:, None], dim=-1)
    elif metric == "euclidean":
        return th.cdist(first_vectors, second_vectors, p=2)
    elif metric == "manhattan":
        return th.cdist(first_vectors, second_vectors, p=1)
    elif metric == "dot":
        return first_vectors @ second_vectors.transpose(-1, -2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def get_metric(
    first_vectors: th.Tensor,  # (B, L, D)
    second_vectors: th.Tensor,  # (B, L', D)
    metric: str,
    max_layer_batch_size: int = 64,  # maximum number of layers to compute the metric for at once
) -> th.Tensor:  # (B, L, L')
    if max_layer_batch_size >= first_vectors.shape[1] and max_layer_batch_size >= second_vectors.shape[1]:
        return get_metric_between_vectors(first_vectors, second_vectors, metric)

    first_layer_idx_generator = range(0, first_vectors.shape[1], max_layer_batch_size)
    second_layer_idx_generator = range(0, second_vectors.shape[1], max_layer_batch_size)
    output = th.empty(first_vectors.shape[0], first_vectors.shape[1], second_vectors.shape[1])

    for first_layer_idx, second_layer_idx in product(first_layer_idx_generator, second_layer_idx_generator):
        first_layer_slice = slice(first_layer_idx, first_layer_idx + max_layer_batch_size)
        second_layer_slice = slice(second_layer_idx, second_layer_idx + max_layer_batch_size)

        first_layers_vectors = first_vectors[:, first_layer_slice]
        second_layers_vectors = second_vectors[:, second_layer_slice]

        output[:, first_layer_slice, second_layer_slice] = get_metric(first_layers_vectors, second_layers_vectors, metric, max_layer_batch_size)

    return output

def get_trajectory_deltas(
    flattened_trajectories: th.Tensor  # (B, L, D)
) -> th.Tensor:  # (B, (L * (L + 1)) / 2, D)
    # first get deltas between each layer, then between each 2nd layer, then between each 3rd layer, etc.
    traj_deltas = [
        flattened_trajectories[:, layer_idx:] - flattened_trajectories[:, :-layer_idx]
        for layer_idx in range(1, flattened_trajectories.shape[1])
    ]

    return th.cat(traj_deltas, dim=1)


class ProjectionMethods:
    def __init__(
        self,
        dim: int = 2
    ) -> None:
        self.projection_dim = dim
    
    def project(self, trajectories: Iterable[th.Tensor], method: str = "pca", **kwargs: dict[str, Any]) -> tuple[Iterable[th.Tensor], th.Tensor | None]:
        flattened_trajs = [model_trajectories.reshape(-1, model_trajectories.shape[-1]) for model_trajectories in trajectories]
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

def visualize_trajectories(
    output_dir: str | os.PathLike = pathlib.Path("outputs"),
    prompt_types: Iterable[Iterable[str]] = [[]],
    projection_method: str = "pca",
    reorder: bool = False,  # whether to permute the activations to try and be more similar
    separate_models: bool = True,  # whether to plot each model separately
    color_map_name: str = "hsv",
    incorrect_answer_modifier: float = 0.5,  # make incorrect trajectories half as bright
    token_distance_modifier: float = 0.5,  # make the second from last token half as opaque, third from last a quarter as opaque, etc.
    token_average_modifier: float = 0.5,  # make each trajectory half as opaque, show the average at full opacity
    only_final_token: bool = True,
    last_n_tokens: int = 4,  # if only_final_token is False, how many tokens to plot
) -> None:
    projection_methods = ProjectionMethods(dim=2)
    for prompt_type in prompt_types:
        _, prompts_hash = get_prompts(prompt_type, return_hash=True)
        prompts_results_dir = os.path.join(output_dir, f"{prompts_hash}{1 if only_final_token else 0}")

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
            trajectories = [trajectory[:, -last_n_tokens:] for trajectory in trajectories]

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


def visualize_hidden_deltas(
    output_dir: str | os.PathLike = pathlib.Path("outputs"),
    prompt_types: Iterable[Iterable[str]] = [[]],
    metric: str = "euclidean",  # metric to use for distance or similarity between hidden states
    reorder: bool = False,  # whether to permute the activations to try and be more similar
    separate_models: bool = True,  # whether to plot each model separately
    colormap_name: str = "hot",
    only_final_token: bool = True,
    last_n_tokens: int = 4,  # if only_final_token is False, how many tokens to plot
) -> None:
    for prompt_type in prompt_types:
        _, prompts_hash = get_prompts(prompt_type, return_hash=True)
        prompts_results_dir = os.path.join(output_dir, f"{prompts_hash}{1 if only_final_token else 0}")

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

        for data in batched_data.values():
            trajectories, model_names = data["trajectories"], data["model_names"]
            trajectories = [trajectory[:, -last_n_tokens:] for trajectory in trajectories]

            for model_name, model_trajectories in zip(
                model_names,
                trajectories,  # (B, T, L, D)
            ):
                flattened_trajectories = flatten_tokens(model_trajectories)  # (B', L, D), where B' is at most B * T
                trajectory_deltas = get_trajectory_deltas(flattened_trajectories)  # (B', L', D) where L' = (L * (L + 1)) / 2
                sim_or_dist = get_metric(trajectory_deltas, trajectory_deltas, metric)  # (B', L', L')
                sim_or_dist = sim_or_dist.mean(dim=0)  # (L', L')

                delta_distance_ticks = []
                current_delta_distance = 0
                for layer_idx in range(flattened_trajectories.shape[1]):
                    delta_distance_ticks.append(current_delta_distance)
                    current_delta_distance += flattened_trajectories.shape[1] - 1 - layer_idx

                plt.imshow(sim_or_dist)
                plt.title(f"{model_name} average {metric} between hidden state deltas, {model_trajectories.shape[0]} layers")
                plt.xticks(delta_distance_ticks)
                plt.yticks(delta_distance_ticks)
                plt.clim(0, 40)
                plt.colorbar()
                plt.set_cmap(colormap_name)
                plt.show()


if __name__ == "__main__":
    # visualize_trajectories(only_final_token=False)
    visualize_hidden_deltas(
        only_final_token=False
    )
