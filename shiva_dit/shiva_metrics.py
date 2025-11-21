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
        "loss/gater/gate_loss_raw": 0.0,
        "loss/gater/gate_loss": 0.0,
    }
    loss_weight = config["gate_loss_weight"]
    if gate_losses:
        raw_loss = torch.stack(gate_losses).mean()
        final_loss = raw_loss * loss_weight
        metrics["loss/gater/gate_loss_raw"] = raw_loss
        metrics["loss/gater/gate_loss"] = final_loss
        return final_loss, metrics
    else:
        return torch.tensor(0.0, device=device), metrics


def calculate_ratio_loss(
    config: dict, device: torch.device, timestep: torch.Tensor = None
) -> tuple[torch.Tensor, dict, dict]:

    ratios_per_layer = {}
    ratios_for_loss = []
    ratios_for_logging = []

    for hook in _get_shiva_hooks():
        layer_name = f"layer_{hook.layer_idx:02d}"
        accumulator = hook.ratio_accumulator

        if accumulator:
            avg_ratio_for_hook = torch.stack(accumulator).mean()
            ratios_for_loss.append(avg_ratio_for_hook)

            avg_ratio_item = avg_ratio_for_hook.detach().item()
            ratios_per_layer[layer_name] = avg_ratio_item
            ratios_for_logging.append(avg_ratio_item)

        hook.ratio_accumulator.clear()

    metrics = {
        "adaptive_ratio/avg_ratio": 0.0,
        "adaptive_ratio/target_ratio": 0.0,
        "loss/ratio/budget_loss": 0.0,
        "loss/ratio/floor_loss": 0.0,
        "loss/ratio/ratio_loss": 0.0,
    }

    if not ratios_for_loss:
        return torch.tensor(0.0, device=device), metrics, ratios_per_layer

    w_config = config.get("ratio_loss_weight", 1.0)
    if isinstance(w_config, list):
        w_budget = w_config[0]
        w_floor = w_config[1] if len(w_config) > 1 else 0.0
    else:
        w_budget = w_config
        w_floor = 0.0

    current_target_ratio = config.get("target_ratio", 0.8)

    all_ratios = torch.stack(ratios_for_loss) # (Num_Layers,)
    
    global_avg_ratio = torch.mean(all_ratios)
    target = torch.tensor(current_target_ratio, device=device, dtype=all_ratios.dtype)

    # Budget Loss: Hinge Loss (ReLU)
    diff = global_avg_ratio - target
    budget_loss = w_budget * torch.relu(diff)

    if w_floor > 0.0:
        min_allowed = 0.05 
        floor_penalties = torch.relu(min_allowed - all_ratios)
        floor_loss = torch.sum(floor_penalties) * w_floor
    else:
        floor_loss = torch.tensor(0.0, device=device)

    final_loss = budget_loss + floor_loss

    with torch.no_grad():
        metrics.update({
            "adaptive_ratio/avg_ratio": global_avg_ratio.detach(),
            "adaptive_ratio/target_ratio": target,
            "loss/ratio/budget_loss": budget_loss.detach(),
            "loss/ratio/floor_loss": floor_loss.detach(),
            "loss/ratio/ratio_loss": final_loss.detach(),
        })

    return final_loss, metrics, ratios_per_layer

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
            gate_probs_per_layer[layer_name] = avg_prob_for_hook

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
        stats["topkgrads/grad_probs_norm"] = ShivaTopkFunc._last_grad_probs_stats[
            "norm"
        ]
        stats["topkgrads/grad_probs_sum"] = ShivaTopkFunc._last_grad_probs_stats[
            "sum"
        ]
        stats["topkgrads/grad_probs_std"] = ShivaTopkFunc._last_grad_probs_stats[
            "std"
        ]

    if (
        hasattr(ShivaTopkFunc, "_last_soft_ranks_stats")
        and ShivaTopkFunc._last_soft_ranks_stats
    ):
        stats["topkgrads/soft_ranks_std"] = ShivaTopkFunc._last_soft_ranks_stats[
            "std"
        ]
        stats["topkgrads/soft_ranks_mean"] = ShivaTopkFunc._last_soft_ranks_stats[
            "mean"
        ]
        stats["topkgrads/soft_ranks_min"] = ShivaTopkFunc._last_soft_ranks_stats[
            "min"
        ]
        stats["topkgrads/soft_ranks_max"] = ShivaTopkFunc._last_soft_ranks_stats[
            "max"
        ]

    if hasattr(ShivaTopkFunc, "_last_z_stats") and ShivaTopkFunc._last_z_stats:
        stats["topkgrads/z_std"] = ShivaTopkFunc._last_z_stats["std"]
        stats["topkgrads/z_mean"] = ShivaTopkFunc._last_z_stats["mean"]

    if (
        hasattr(ShivaTopkFunc, "_last_selection_probs_stats")
        and ShivaTopkFunc._last_selection_probs_stats
    ):
        stats["topkgrads/sigmoid_prime_mean"] = (
            ShivaTopkFunc._last_selection_probs_stats["grad_mean"]
        )
        stats["topkgrads/sigmoid_prime_std"] = (
            ShivaTopkFunc._last_selection_probs_stats["grad_std"]
        )

    return stats

@torch.no_grad()
def collect_router_stats(device: torch.device) -> dict:
    total_mean = []
    total_std = []
    total_min = []
    total_max = []
    total_p25 = []
    total_p50 = []
    total_p75 = []
    q_tensor = torch.tensor([0.25, 0.50, 0.75], device=device)

    total_s_scale_mean = []
    total_s_scale_std = []

    for hook in _get_shiva_hooks():
        acc = hook.router_stats_accumulator
        if acc:
            total_mean.append(torch.stack([x["mean"] for x in acc]).mean())
            total_std.append(torch.stack([x["std"] for x in acc]).mean())
            total_min.append(torch.stack([x["min"] for x in acc]).min())
            total_max.append(torch.stack([x["max"] for x in acc]).max())
            total_s_scale_mean.append(torch.stack([x["s_scale_mean"] for x in acc]).mean())
            total_s_scale_std.append(torch.stack([x["s_scale_std"] for x in acc]).mean())

            all_scores = torch.stack([x["scores_tensor"] for x in acc]) 
            quantiles = torch.quantile(all_scores.to(torch.float32), q_tensor.to(torch.float32), dim=None) 
            total_p25.append(quantiles[0])
            total_p50.append(quantiles[1])
            total_p75.append(quantiles[2])
        acc.clear()

    if total_mean:
        return {
            "router/mean_score": torch.stack(total_mean).mean(),
            "router/std_score": torch.stack(total_std).mean(),
            "router/min_score": torch.stack(total_min).min(),
            "router/max_score": torch.stack(total_max).max(),
            "router/s_scale_mean": torch.stack(total_s_scale_mean).mean(),
            "router/s_scale_std": torch.stack(total_s_scale_std).mean(),
            "router/p25_score": torch.stack(total_p25).mean(),
            "router/p50_score": torch.stack(total_p50).mean(),
            "router/p75_score": torch.stack(total_p75).mean(),
        }
    else:
        return {}

def generate_router_heatmap(sel_idx, total_tokens, height, width):
    B = sel_idx.shape[0]
    heatmap = torch.zeros((B, total_tokens), dtype=torch.float32)

    heatmap.scatter_(1, sel_idx.long(), 1.0)
    
    try:
        heatmap_img = heatmap.reshape(B, height, width)
    except Exception as e:
        print(f"Heatmap reshape failed: {e}. Tokens: {total_tokens}, H: {height}, W: {width}")
        return None

    return heatmap_img # Tensor (B, H, W)
