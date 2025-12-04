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


# def calculate_gate_loss(config: dict, device: torch.device) -> torch.Tensor:
#     gate_losses = []
#     metrics = {}
#     gate_probs_per_hook = []
#     gate_probs_per_layer = {}

#     for hook in _get_shiva_hooks():
#         accumulator = hook.gate_loss_accumulator
#         if accumulator:
#             avg_loss_for_hook = torch.stack(accumulator).mean()
#             gate_losses.append(avg_loss_for_hook)
#             device = avg_loss_for_hook.device
#         hook.gate_loss_accumulator.clear()

#         accumulator = hook.gater_fuse_prob_accumulator
#         if accumulator:
#             avg_prob_for_hook = torch.stack(accumulator).mean()
#             gate_probs_per_hook.append(avg_prob_for_hook)
#             gate_probs_per_layer[f"layer_{hook.layer_idx:02d}"] = avg_prob_for_hook
#         hook.gater_fuse_prob_accumulator.clear()
    
#     final_loss = torch.tensor(0.0, device=device)

#     if gate_losses:
#         loss_weight = config.get("gate_loss_weight", 0.0)
#         raw_loss = torch.stack(gate_losses).mean()
#         final_loss = raw_loss * loss_weight
        
#         metrics["loss/gater/gate_loss_raw"] = raw_loss.detach()
#         metrics["loss/gater/gate_loss"] = final_loss.detach()
    
#     if gate_probs_per_hook:
#         global_avg_prob = torch.stack(gate_probs_per_hook).mean()
#         metrics['gater/avg_fuse_prob'] = global_avg_prob
        
#     return final_loss, metrics, gate_probs_per_layer

def calculate_gate_loss(config: dict, device: torch.device) -> torch.Tensor:
    metrics = {}
    gate_probs_per_hook = []
    gate_probs_per_layer = {}

    for hook in _get_shiva_hooks():
        if hook.gate_loss_accumulator:
            hook.gate_loss_accumulator.clear()

        accumulator = hook.gater_fuse_prob_accumulator
        if accumulator:
            avg_prob_for_hook = torch.stack(accumulator).mean()
            gate_probs_per_hook.append(avg_prob_for_hook)
            gate_probs_per_layer[f"layer_{hook.layer_idx:02d}"] = avg_prob_for_hook
        hook.gater_fuse_prob_accumulator.clear()
    
    final_loss = torch.tensor(0.0, device=device)

    if gate_probs_per_hook:
        all_probs = torch.stack(gate_probs_per_hook)
        global_avg_prob = all_probs.mean()
        
        loss_weight = config.get("gate_loss_weight", 0.0)
        final_loss = global_avg_prob * loss_weight
        
        metrics["loss/gater/fuse_penalty"] = final_loss.detach()
        metrics["gater/avg_fuse_prob"] = global_avg_prob.detach()
        metrics["gater/max_fuse_prob"] = all_probs.max().detach()
        metrics["gater/min_fuse_prob"] = all_probs.min().detach()
        
    return final_loss, metrics, gate_probs_per_layer


def _calc_ratio_loss_flat(all_ratios, config, device, prefix="adaptive_ratio"):
    if all_ratios.numel() == 0:
        return torch.tensor(0.0, device=device), {}

    # Config parsing
    # Format: [w_budget_L1, w_floor, w_budget_MSE]
    w_config = config.get("ratio_loss_weight", 1.0)
    
    w_budget_l1 = 0.0
    w_floor = 0.0
    w_budget_mse = 0.0

    if isinstance(w_config, list):
        if len(w_config) >= 1: w_budget_l1 = w_config[0]
        if len(w_config) >= 2: w_floor = w_config[1]
        if len(w_config) >= 3: w_budget_mse = w_config[2]
    else:
        w_budget_l1 = w_config

    current_target_ratio = config.get("target_ratio", 0.8)
    target = torch.tensor(current_target_ratio, device=device, dtype=all_ratios.dtype)
    
    global_avg_ratio = torch.mean(all_ratios)

    # 1. Budget Loss (L1)
    if w_budget_l1 > 0.0:
        budget_loss_l1 = w_budget_l1 * torch.abs(global_avg_ratio - target)
    else:
        budget_loss_l1 = torch.tensor(0.0, device=device)

    # 2. Floor Loss
    if w_floor > 0.0:
        min_allowed = 0.1
        floor_penalties = torch.relu(min_allowed - all_ratios)
        floor_loss = torch.sum(floor_penalties) * w_floor / all_ratios.numel()
    else:
        floor_loss = torch.tensor(0.0, device=device)

    # 3. Budget Loss (MSE)
    if w_budget_mse > 0.0:
        # MSE = (avg - target)^2
        budget_loss_mse = w_budget_mse * ((global_avg_ratio - target) ** 2)
    else:
        budget_loss_mse = torch.tensor(0.0, device=device)

    ratio_std = torch.std(all_ratios)

    final_loss = budget_loss_l1 + floor_loss + budget_loss_mse

    metrics = {
        f"{prefix}/avg_ratio": global_avg_ratio.detach(),
        f"{prefix}/std_ratio": ratio_std.detach(),
        f"{prefix}/target_ratio": target,
        f"loss/{prefix}/budget_loss": budget_loss_l1.detach(),
        f"loss/{prefix}/mse_loss": budget_loss_mse.detach(),
        f"loss/{prefix}/floor_loss": floor_loss.detach(),
        f"loss/{prefix}/ratio_loss": final_loss.detach(),
    }
    return final_loss, metrics


def calculate_image_ratio_loss(
    config: dict, device: torch.device
) -> tuple[torch.Tensor, dict, dict]:
    all_ratios_flat = []
    per_layer_ratios = {}

    for hook in _get_shiva_hooks():            
        accumulator = getattr(hook, "image_ratio_accumulator", [])
        if accumulator:
            stacked_ratios = torch.stack(accumulator)
            all_ratios_flat.append(stacked_ratios)
            per_layer_ratios[f"layer_{hook.layer_idx:02d}"] = stacked_ratios.mean().item()
        
        if hasattr(hook, "image_ratio_accumulator"):
            hook.image_ratio_accumulator.clear()

    if not all_ratios_flat:
        return torch.tensor(0.0, device=device), {}, {}

    # (Total_Layers * Total_MicroBatches, )
    ratios_concat = torch.cat(all_ratios_flat).flatten()

    loss, metrics = _calc_ratio_loss_flat(ratios_concat, config, device, prefix="image_ratio")
    
    return loss, metrics, per_layer_ratios


def calculate_context_ratio_loss(
    config: dict, device: torch.device
) -> tuple[torch.Tensor, dict]:
    all_ratios_flat = []
    per_layer_ratios = {}
    
    for hook in _get_shiva_hooks():
        accumulator = getattr(hook, "context_ratio_accumulator", [])
        if accumulator:
            stacked_ratios = torch.stack(accumulator)
            all_ratios_flat.append(stacked_ratios)
            per_layer_ratios[f"layer_{hook.layer_idx:02d}"] = stacked_ratios.mean().item()
        
        if hasattr(hook, "context_ratio_accumulator"):
            hook.context_ratio_accumulator.clear()
            
    if not all_ratios_flat:
        return torch.tensor(0.0, device=device), {}, {}

    ratios_concat = torch.cat(all_ratios_flat).flatten()
    loss, metrics = _calc_ratio_loss_flat(ratios_concat, config, device, prefix="context_ratio")
    
    return loss, metrics, per_layer_ratios


# def _calc_ratio_loss_common(ratios_list, config, device, prefix="adaptive_ratio"):
#     """
#     Common logic for calculating budget/floor/mse loss for a list of ratios.
#     """
#     if not ratios_list:
#         return torch.tensor(0.0, device=device), {}

#     # Config parsing: [budget_weight, floor_weight, mse_weight]
#     w_config = config.get("ratio_loss_weight", 1.0)
#     if isinstance(w_config, list):
#         w_budget = w_config[0]
#         w_floor = w_config[1] if len(w_config) > 1 else 0.0
#         w_mse = w_config[2] if len(w_config) > 2 else 0.0
#     else:
#         w_budget = w_config
#         w_floor = 0.0
#         w_mse = 0.0

#     current_target_ratio = config.get("target_ratio", 0.8)

#     all_ratios = torch.stack(ratios_list) # (Num_Active_Layers,)
    
#     global_avg_ratio = torch.mean(all_ratios)
#     target = torch.tensor(current_target_ratio, device=device, dtype=all_ratios.dtype)

#     # 1. Budget Loss (Hinge Loss)
#     diff = global_avg_ratio - target
#     budget_loss = w_budget * torch.relu(diff)

#     # 2. Floor Loss (Penalty for going too low)
#     if w_floor > 0.0:
#         min_allowed = 0.1
#         floor_penalties = torch.relu(min_allowed - all_ratios)
#         floor_loss = torch.sum(floor_penalties) * w_floor
#     else:
#         floor_loss = torch.tensor(0.0, device=device)

#     # 3. MSE Loss (Target adherence)
#     if w_mse > 0.0:
#         target_full = torch.full_like(all_ratios, current_target_ratio)
#         mse_loss_val = F.mse_loss(all_ratios, target_full)
#         mse_loss = w_mse * mse_loss_val
#     else:
#         mse_loss = torch.tensor(0.0, device=device)

#     final_loss = budget_loss + floor_loss + mse_loss

#     metrics = {
#         f"{prefix}/avg_ratio": global_avg_ratio.detach(),
#         f"{prefix}/target_ratio": target,
#         f"loss/{prefix}/budget_loss": budget_loss.detach(),
#         f"loss/{prefix}/floor_loss": floor_loss.detach(),
#         f"loss/{prefix}/mse_loss": mse_loss.detach(),
#         f"loss/{prefix}/ratio_loss": final_loss.detach(),
#     }
#     return final_loss, metrics


# def calculate_image_ratio_loss(
#     config: dict, device: torch.device
# ) -> tuple[torch.Tensor, dict, dict]:
#     """
#     Calculates loss for Image Token Pruning ratios.
#     """
#     ratios_for_loss = []
#     per_layer_ratios = {}

#     for hook in _get_shiva_hooks():            
#         accumulator = getattr(hook, "image_ratio_accumulator", [])
#         if accumulator:
#             avg_ratio = torch.stack(accumulator).mean()
#             ratios_for_loss.append(avg_ratio)
#             per_layer_ratios[f"layer_{hook.layer_idx:02d}"] = avg_ratio.item()
        
#         # Clear after reading
#         if hasattr(hook, "image_ratio_accumulator"):
#             hook.image_ratio_accumulator.clear()

#     # Use 'adaptive_ratio' prefix for backward compatibility/standard image naming
#     loss, metrics = _calc_ratio_loss_common(ratios_for_loss, config, device, prefix="image_ratio")
    
#     return loss, metrics, per_layer_ratios


# def calculate_context_ratio_loss(
#     config: dict, device: torch.device
# ) -> tuple[torch.Tensor, dict]:
#     """
#     Calculates loss for Context Pruning ratios.
#     """
#     ratios_for_loss = []
#     per_layer_ratios = {}
    
#     for hook in _get_shiva_hooks():
#         accumulator = getattr(hook, "context_ratio_accumulator", [])
#         if accumulator:
#             avg_ratio = torch.stack(accumulator).mean()
#             ratios_for_loss.append(avg_ratio)
#             per_layer_ratios[f"layer_{hook.layer_idx:02d}"] = avg_ratio.item()
        
#         if hasattr(hook, "context_ratio_accumulator"):
#             hook.context_ratio_accumulator.clear()
            
#     loss, metrics = _calc_ratio_loss_common(ratios_for_loss, config, device, prefix="context_ratio")
    
#     return loss, metrics, per_layer_ratios


def collect_context_stats(device: torch.device) -> dict:
    total_mean = []
    total_std = []
    total_max = []
    total_min = []
    
    for hook in _get_shiva_hooks():
        acc = getattr(hook, "context_score_accumulator", [])
        if acc:
            stacked = torch.stack(acc)
            total_mean.append(stacked.mean())
            total_std.append(stacked.std())
            total_max.append(stacked.max())
            total_min.append(stacked.min())
            
        if hasattr(hook, "context_score_accumulator"):
            hook.context_score_accumulator.clear()

    if total_mean:
        return {
            "context/align_score_mean": torch.stack(total_mean).mean(),
            "context/align_score_std": torch.stack(total_std).mean(),
            "context/align_score_max": torch.stack(total_max).max(),
            "context/align_score_min": torch.stack(total_min).min(),
        }
    else:
        return {}


def collect_autograd_stats(device):
    stats = {}
    # Check static stats from the Function
    if (
        hasattr(ShivaTopkFunc, "_last_grad_probs_stats")
        and ShivaTopkFunc._last_grad_probs_stats
    ):
        stats["topkgrads/grad_probs_norm"] = ShivaTopkFunc._last_grad_probs_stats.get("norm", 0.0)
        stats["topkgrads/grad_probs_sum"] = ShivaTopkFunc._last_grad_probs_stats.get("sum", 0.0)
        stats["topkgrads/grad_probs_std"] = ShivaTopkFunc._last_grad_probs_stats.get("std", 0.0)

    if (
        hasattr(ShivaTopkFunc, "_last_soft_ranks_stats")
        and ShivaTopkFunc._last_soft_ranks_stats
    ):
        stats["topkgrads/soft_ranks_std"] = ShivaTopkFunc._last_soft_ranks_stats.get("std", 0.0)
        stats["topkgrads/soft_ranks_mean"] = ShivaTopkFunc._last_soft_ranks_stats.get("mean", 0.0)
        stats["topkgrads/soft_ranks_min"] = ShivaTopkFunc._last_soft_ranks_stats.get("min", 0.0)
        stats["topkgrads/soft_ranks_max"] = ShivaTopkFunc._last_soft_ranks_stats.get("max", 0.0)

    if hasattr(ShivaTopkFunc, "_last_z_stats") and ShivaTopkFunc._last_z_stats:
        stats["topkgrads/z_std"] = ShivaTopkFunc._last_z_stats.get("std", 0.0)
        stats["topkgrads/z_mean"] = ShivaTopkFunc._last_z_stats.get("mean", 0.0)

    if (
        hasattr(ShivaTopkFunc, "_last_selection_probs_stats")
        and ShivaTopkFunc._last_selection_probs_stats
    ):
        stats["topkgrads/sigmoid_prime_mean"] = (
            ShivaTopkFunc._last_selection_probs_stats.get("grad_mean", 0.0)
        )
        stats["topkgrads/sigmoid_prime_std"] = (
            ShivaTopkFunc._last_selection_probs_stats.get("grad_std", 0.0)
        )

    return stats


@torch.no_grad()
def collect_router_stats(device: torch.device) -> dict:
    total_mean = []
    total_std = []
    total_min = []
    total_max = []
    
    # Quantiles might be expensive, check if accumulator has data first
    q_tensor = torch.tensor([0.25, 0.50, 0.75], device=device)
    total_p25 = []
    total_p50 = []
    total_p75 = []

    for hook in _get_shiva_hooks():
        acc = hook.router_stats_accumulator
        if acc:
            total_mean.append(torch.stack([x["mean"] for x in acc]).mean())
            total_std.append(torch.stack([x["std"] for x in acc]).mean())
            total_min.append(torch.stack([x["min"] for x in acc]).min())
            total_max.append(torch.stack([x["max"] for x in acc]).max())

            all_scores = torch.stack([x["scores_tensor"] for x in acc]) 
            # Reshape to flat for quantile
            all_scores_flat = all_scores.float().view(-1)
            if all_scores_flat.numel() > 0:
                quantiles = torch.quantile(all_scores_flat, q_tensor.float()) 
                total_p25.append(quantiles[0])
                total_p50.append(quantiles[1])
                total_p75.append(quantiles[2])
        
        hook.router_stats_accumulator.clear()

    if total_mean:
        stats = {
            "router/mean_score": torch.stack(total_mean).mean(),
            "router/std_score": torch.stack(total_std).mean(),
            "router/min_score": torch.stack(total_min).min(),
            "router/max_score": torch.stack(total_max).max(),
        }
        if total_p25:
            stats.update({
                "router/p25_score": torch.stack(total_p25).mean(),
                "router/p50_score": torch.stack(total_p50).mean(),
                "router/p75_score": torch.stack(total_p75).mean(),
            })
        return stats
    else:
        return {}


def generate_router_heatmap(token_scores, sel_idx, height, width):
    try:
        import matplotlib.cm
    except ImportError:
        print("matplotlib not found.")
        return None

    if token_scores.dim() == 3:
        token_scores = token_scores.squeeze(-1)
    
    B, total_tokens = token_scores.shape
    
    try:
        score_map = token_scores.reshape(B, height, width)
    except Exception as e:
        print(f"Reshape failed: {e}")
        return None

    score_np = score_map.float().cpu().numpy() 
    
    cm = matplotlib.cm.get_cmap('jet') 
    colored_np = cm(score_np) # (B, H, W, 4)
    rgb_np = colored_np[..., :3] # (B, H, W, 3)

    mask = torch.zeros((B, total_tokens), dtype=torch.float32)
    if sel_idx.max() < total_tokens:
        mask.scatter_(1, sel_idx.long(), 1.0)
    
    mask_map = mask.reshape(B, height, width).numpy() # (B, H, W)
    mask_expanded = mask_map[..., None]

    masked_rgb_np = rgb_np * mask_expanded

    # (B, H, W, 3) -> (B, 3, H, W)
    final_tensor = torch.from_numpy(masked_rgb_np).permute(0, 3, 1, 2).float()
    
    return final_tensor