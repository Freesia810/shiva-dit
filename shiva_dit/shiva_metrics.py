import torch
import torch.nn.functional as F
from typing import Any

from shiva_dit.hook import HookManager
from shiva_dit.shiva_hook import ShivaDitBlockHook
from shiva_dit.utils import ShivaTopkFunc


def _get_shiva_hooks():
    if not HookManager._enabled:
        return

    for hook in HookManager.get_all_hooks().values():
        if isinstance(hook, ShivaDitBlockHook):
            yield hook


def calculate_consistency_loss(config: dict, device: torch.device) -> torch.Tensor:
    consistency_loss = []

    for hook in _get_shiva_hooks():
        prune_list = hook.prune_output_accumulator
        fuse_list = hook.fuse_output_accumulator

        if prune_list and fuse_list:
            assert len(prune_list) == len(
                fuse_list
            ), "Prune/Fuse accumulator length mismatch"

            micro_batch_losses = []
            for prune_out, fuse_out in zip(prune_list, fuse_list):
                loss = F.mse_loss(
                    prune_out.detach().to(torch.float32),
                    fuse_out.to(torch.float32),
                )
                micro_batch_losses.append(loss)

            if micro_batch_losses:
                avg_loss_for_hook = torch.stack(micro_batch_losses).mean()
                consistency_loss.append(avg_loss_for_hook)
                device = avg_loss_for_hook.device

        hook.prune_output_accumulator.clear()
        hook.fuse_output_accumulator.clear()

    metrics = {
        "shiva/gater/consistency_loss_raw": 0.0,
        "shiva/gater/consistency_loss": 0.0,
    }
    loss_weight = config["consistency_loss_weight"]
    if consistency_loss:
        raw_loss = torch.stack(consistency_loss).mean()
        final_loss = raw_loss * loss_weight
        metrics["shiva/gater/consistency_loss_raw"] = raw_loss.item()
        metrics["shiva/gater/consistency_loss"] = final_loss.item()
        return final_loss, metrics
    else:
        return torch.tensor(0.0, device=device), metrics


def calculate_gate_loss(config: dict, device: torch.device) -> torch.Tensor:
    gate_losses = []

    for hook in _get_shiva_hooks():
        accumulator = hook.gate_loss_accumulator

        if accumulator:
            avg_loss_for_hook = torch.stack(accumulator).mean()
            gate_losses.append(avg_loss_for_hook)
            device = avg_loss_for_hook.device

        hook.gate_loss_accumulator.clear()

    metrics = {
        "shiva/gater/gate_loss_raw": 0.0,
        "shiva/gater/gate_loss": 0.0,
    }
    loss_weight = config["gate_loss_weight"]
    if gate_losses:
        raw_loss = torch.stack(gate_losses).mean()
        final_loss = raw_loss * loss_weight
        metrics["shiva/gater/gate_loss_raw"] = raw_loss.item()
        metrics["shiva/gater/gate_loss"] = final_loss.item()
        return final_loss, metrics
    else:
        return torch.tensor(0.0, device=device), metrics


def calculate_ratio_loss(
    config: dict, device: torch.device
) -> tuple[torch.Tensor, dict]:
    ratios_per_layer = {}
    ratios = []
    for hook in _get_shiva_hooks():
        layer_name = f"layer_{hook.layer_idx:02d}"
        current_ratio_unclamped = hook.shiva_ratio
        current_ratio = torch.clamp(current_ratio_unclamped, min=1e-9, max=1.0)
        ratios_per_layer[layer_name] = current_ratio.item()
        ratios.append(current_ratio)

    metrics = {
        "shiva/ratio/avg_ratio": 0.0,
        "shiva/ratio/std_ratio": 0.0,
        "shiva/ratio/ratio_loss_raw": 0.0,
        "shiva/ratio/ratio_loss": 0.0,
    }

    loss_weight = config["ratio_loss_weight"]
    current_target_ratio = config["target_ratio"]
    if ratios:
        ratios_tensor = torch.stack(ratios)
        avg_ratio = torch.mean(ratios_tensor)
        std_ratio = torch.std(ratios_tensor)
        target_ratio_tensor = torch.tensor(
            current_target_ratio, device=device, dtype=avg_ratio.dtype
        )
        mse_loss = F.mse_loss(avg_ratio, target_ratio_tensor)
        final_loss = mse_loss * loss_weight
        metrics = {
            "shiva/ratio/avg_ratio": avg_ratio.item(),
            "shiva/ratio/std_ratio": std_ratio.item(),
            "shiva/ratio/ratio_loss_raw": mse_loss.item(),
            "shiva/ratio/ratio_loss": final_loss.item(),
        }

        return final_loss, metrics, ratios_per_layer
    else:
        return torch.tensor(0.0, device=device), metrics, ratios_per_layer


@torch.no_grad()
def calculate_shiva_gate_prob(device: torch.device) -> tuple[torch.Tensor, dict]:
    gate_probs_per_hook = []
    gate_probs_per_layer = {}

    for hook in _get_shiva_hooks():
        accumulator = hook.gater_fuse_prob_accumulator

        if accumulator:
            avg_prob_for_hook = torch.stack(accumulator).mean()
            gate_probs_per_hook.append(avg_prob_for_hook)

            layer_name = f"layer_{hook.layer_idx:02d}"
            gate_probs_per_layer[layer_name] = avg_prob_for_hook.item()

        hook.gater_fuse_prob_accumulator.clear()

    if gate_probs_per_hook:
        global_avg_prob = torch.stack(gate_probs_per_hook).mean()
        return global_avg_prob, gate_probs_per_layer
    else:
        return torch.tensor(0.0, device=device), {}


def collect_autograd_stats(device):
    stats = {}
    if (
        hasattr(ShivaTopkFunc, "_last_grad_probs_stats")
        and ShivaTopkFunc._last_grad_probs_stats
    ):
        stats["shiva/autograd/grad_probs_norm"] = ShivaTopkFunc._last_grad_probs_stats[
            "norm"
        ]
        stats["shiva/autograd/grad_probs_sum"] = ShivaTopkFunc._last_grad_probs_stats[
            "sum"
        ]
        stats["shiva/autograd/grad_probs_std"] = ShivaTopkFunc._last_grad_probs_stats[
            "std"
        ]

    if (
        hasattr(ShivaTopkFunc, "_last_soft_ranks_stats")
        and ShivaTopkFunc._last_soft_ranks_stats
    ):
        stats["shiva/autograd/soft_ranks_std"] = ShivaTopkFunc._last_soft_ranks_stats[
            "std"
        ]
        stats["shiva/autograd/soft_ranks_mean"] = ShivaTopkFunc._last_soft_ranks_stats[
            "mean"
        ]

    if hasattr(ShivaTopkFunc, "_last_z_stats") and ShivaTopkFunc._last_z_stats:
        stats["shiva/autograd/z_std"] = ShivaTopkFunc._last_z_stats["std"]
        stats["shiva/autograd/z_mean"] = ShivaTopkFunc._last_z_stats["mean"]

    if (
        hasattr(ShivaTopkFunc, "_last_selection_probs_stats")
        and ShivaTopkFunc._last_selection_probs_stats
    ):
        stats["shiva/autograd/sigmoid_prime_mean"] = (
            ShivaTopkFunc._last_selection_probs_stats["grad_mean"]
        )
        stats["shiva/autograd/sigmoid_prime_std"] = (
            ShivaTopkFunc._last_selection_probs_stats["grad_std"]
        )

    return stats

@torch.no_grad()
def collect_router_stats(device: torch.device) -> dict:
    total_mean = []
    total_std = []
    total_min = []
    total_max = []

    for hook in _get_shiva_hooks():
        acc = hook.router_stats_accumulator
        if acc:
            total_mean.append(torch.stack([x["mean"] for x in acc]).mean())
            total_std.append(torch.stack([x["std"] for x in acc]).mean())
            total_min.append(torch.stack([x["min"] for x in acc]).min())
            total_max.append(torch.stack([x["max"] for x in acc]).max())
        acc.clear()

    if total_mean:
        return {
            "shiva/router/mean_score": torch.stack(total_mean).mean(),
            "shiva/router/std_score": torch.stack(total_std).mean(),
            "shiva/router/min_score": torch.stack(total_min).min(),
            "shiva/router/max_score": torch.stack(total_max).max(),
        }
    else:
        return {}


@torch.no_grad()
def collect_pruner_T_stats(device: torch.device) -> dict:
    tk_values = []
    ts_values = []
    t_metrics_per_layer = {}

    for hook in _get_shiva_hooks():
        if hasattr(hook, "prune_expert") and hasattr(hook.prune_expert, "shiva_Tk"):
            if isinstance(hook.prune_expert.shiva_Tk, torch.nn.Parameter):
                t_val = hook.prune_expert.shiva_Tk.item()
                tk_values.append(t_val)

                layer_name = f"layer_{hook.layer_idx:02d}_Tk"
                t_metrics_per_layer[layer_name] = t_val

        if hasattr(hook, "prune_expert") and hasattr(hook.prune_expert, "shiva_Ts"):
            if isinstance(hook.prune_expert.shiva_Ts, torch.nn.Parameter):
                t_val = hook.prune_expert.shiva_Ts.item()
                ts_values.append(t_val)

                layer_name = f"layer_{hook.layer_idx:02d}_Ts"
                t_metrics_per_layer[layer_name] = t_val

    metrics = {}

    if tk_values:
        t_tensor_k = torch.tensor(tk_values, device=device)
        metrics["shiva/pruner/avg_T_k"] = t_tensor_k.mean().item()
        metrics["shiva/pruner/std_T_k"] = t_tensor_k.std().item()
        metrics["shiva/pruner/min_T_k"] = t_tensor_k.min().item()
        metrics["shiva/pruner/max_T_k"] = t_tensor_k.max().item()

    if ts_values:
        t_tensor_s = torch.tensor(ts_values, device=device)
        metrics["shiva/pruner/avg_T_s"] = t_tensor_s.mean().item()
        metrics["shiva/pruner/std_T_s"] = t_tensor_s.std().item()
        metrics["shiva/pruner/min_T_s"] = t_tensor_s.min().item()
        metrics["shiva/pruner/max_T_s"] = t_tensor_s.max().item()

    return metrics, {"shiva/pruner/per_layer_T": t_metrics_per_layer}


@torch.no_grad()
def _weighted_avg(values, weights, default_val=0.0):
    if not values or not weights:
        return default_val, 0.0

    total_weight = sum(weights)
    if total_weight == 0:
        return default_val, 0.0

    weighted_sum = sum(v.item() * w for v, w in zip(values, weights))
    return weighted_sum / total_weight, total_weight


@torch.no_grad()
def collect_fuser_stats(device: torch.device) -> tuple[dict, dict]:
    metrics_global_lists = {
        "shiva/fuser/avg_discarded_norm": [],
        "shiva/fuser/avg_diff_norm": [],
        "shiva/fuser/avg_cos_sim": [],
    }
    metrics_per_layer = {
        "shiva/fuser/per_layer_discarded_norm": {},
        "shiva/fuser/per_layer_diff_norm": {},
        "shiva/fuser/per_layer_cos_sim": {},
    }

    for hook in _get_shiva_hooks():
        layer_name = f"layer_{hook.layer_idx:02d}"

        k_weights = hook.k_accumulator
        n_minus_k_weights = hook.n_minus_k_accumulator

        # Discarded Norm (weighted by N-k, number of discarded tokens)
        avg_discarded_norm, total_n_minus_k = _weighted_avg(
            hook.discarded_info_norm_accumulator, n_minus_k_weights
        )
        metrics_global_lists["shiva/fuser/avg_discarded_norm"].append(
            (avg_discarded_norm, total_n_minus_k)
        )
        metrics_per_layer["shiva/fuser/per_layer_discarded_norm"][
            layer_name
        ] = avg_discarded_norm

        # Diff Norm (weighted by k, number of survivor tokens)
        avg_diff_norm, total_k = _weighted_avg(
            hook.fuser_diff_norm_accumulator, k_weights
        )
        metrics_global_lists["shiva/fuser/avg_diff_norm"].append(
            (avg_diff_norm, total_k)
        )
        metrics_per_layer["shiva/fuser/per_layer_diff_norm"][layer_name] = avg_diff_norm

        # Cos Sim (weighted by k, number of survivor tokens)
        avg_cos_sim, total_k_cos = _weighted_avg(
            hook.fuser_cos_sim_accumulator, k_weights
        )
        metrics_global_lists["shiva/fuser/avg_cos_sim"].append(
            (avg_cos_sim, total_k_cos)
        )
        metrics_per_layer["shiva/fuser/per_layer_cos_sim"][layer_name] = avg_cos_sim

        # Clear accumulators
        hook.discarded_info_norm_accumulator.clear()
        hook.fuser_diff_norm_accumulator.clear()
        hook.fuser_cos_sim_accumulator.clear()
        hook.k_accumulator.clear()
        hook.n_minus_k_accumulator.clear()

    # Calculate global weighted average across all hooks
    final_metrics_global = {}
    for key, avg_weight_list in metrics_global_lists.items():
        if not avg_weight_list:
            final_metrics_global[key] = 0.0
            continue

        total_weighted_sum = sum(avg * weight for avg, weight in avg_weight_list)
        global_total_weight = sum(weight for _, weight in avg_weight_list)

        if global_total_weight == 0:
            final_metrics_global[key] = 0.0
        else:
            final_metrics_global[key] = total_weighted_sum / global_total_weight

    return final_metrics_global, metrics_per_layer
