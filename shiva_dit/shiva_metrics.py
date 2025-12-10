import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
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


# def _calc_ratio_loss_flat(all_ratios, config, device, prefix="adaptive_ratio"):
#     if all_ratios.numel() == 0:
#         return torch.tensor(0.0, device=device), {}

#     # Config parsing
#     # Format: [w_budget_L1, w_floor, w_budget_MSE]
#     w_config = config.get("ratio_loss_weight", 1.0)
    
#     w_budget_l1 = 0.0
#     w_floor = 0.0
#     w_budget_mse = 0.0

#     if isinstance(w_config, list):
#         if len(w_config) >= 1: w_budget_l1 = w_config[0]
#         if len(w_config) >= 2: w_floor = w_config[1]
#         if len(w_config) >= 3: w_budget_mse = w_config[2]
#     else:
#         w_budget_l1 = w_config

#     current_target_ratio = config.get("target_ratio", 0.8)
#     target = torch.tensor(current_target_ratio, device=device, dtype=all_ratios.dtype)
    
#     global_avg_ratio = torch.mean(all_ratios)

#     # 1. Budget Loss (L1)
#     if w_budget_l1 > 0.0:
#         budget_loss_l1 = w_budget_l1 * torch.abs(global_avg_ratio - target)
#     else:
#         budget_loss_l1 = torch.tensor(0.0, device=device)

#     # 2. Floor Loss
#     if w_floor > 0.0:
#         min_allowed = 0.1
#         floor_penalties = torch.relu(min_allowed - all_ratios)
#         floor_loss = torch.sum(floor_penalties) * w_floor / all_ratios.numel()
#     else:
#         floor_loss = torch.tensor(0.0, device=device)

#     # 3. Budget Loss (MSE)
#     if w_budget_mse > 0.0:
#         # MSE = (avg - target)^2
#         budget_loss_mse = w_budget_mse * ((global_avg_ratio - target) ** 2)
#     else:
#         budget_loss_mse = torch.tensor(0.0, device=device)

#     ratio_std = torch.std(all_ratios)

#     final_loss = budget_loss_l1 + floor_loss + budget_loss_mse

#     metrics = {
#         f"{prefix}/avg_ratio": global_avg_ratio.detach(),
#         f"{prefix}/std_ratio": ratio_std.detach(),
#         f"{prefix}/target_ratio": target,
#         f"loss/{prefix}/budget_loss": budget_loss_l1.detach(),
#         f"loss/{prefix}/mse_loss": budget_loss_mse.detach(),
#         f"loss/{prefix}/floor_loss": floor_loss.detach(),
#         f"loss/{prefix}/ratio_loss": final_loss.detach(),
#     }
#     return final_loss, metrics


# def calculate_image_ratio_loss(
#     config: dict, device: torch.device
# ) -> tuple[torch.Tensor, dict, dict]:
#     all_ratios_flat = []
#     per_layer_ratios = {}

#     for hook in _get_shiva_hooks():            
#         accumulator = getattr(hook, "image_ratio_accumulator", [])
#         if accumulator:
#             stacked_ratios = torch.stack(accumulator)
#             all_ratios_flat.append(stacked_ratios)
#             per_layer_ratios[f"layer_{hook.layer_idx:02d}"] = stacked_ratios.mean().item()
        
#         if hasattr(hook, "image_ratio_accumulator"):
#             hook.image_ratio_accumulator.clear()

#     if not all_ratios_flat:
#         return torch.tensor(0.0, device=device), {}, {}

#     # (Total_Layers * Total_MicroBatches, )
#     ratios_concat = torch.cat(all_ratios_flat).flatten()

#     loss, metrics = _calc_ratio_loss_flat(ratios_concat, config, device, prefix="image_ratio")
    
#     return loss, metrics, per_layer_ratios


def _calc_ratio_loss_stratified(
    all_ratios, 
    config, 
    device, 
    state_buffer, 
    prefix="adaptive_ratio"
):
    """
    state_buffer (dict): 需要是一个持久化的字典，必须包含以下 key:
        - "history_ratios": list, 用来存 accumulation 期间的 ratio
        - "last_cycle_mean": float, 上一个周期的均值 (初始建议设为 target_ratio)
    """
    if all_ratios.numel() == 0:
        return torch.tensor(0.0, device=device), {}

    # 1. 基础配置解析
    w_config = config.get("ratio_loss_weight", 1.0)
    w_budget_mse = 0.0
    if isinstance(w_config, list):
        if len(w_config) >= 3: w_budget_mse = w_config[2]
    else:
        w_budget_mse = w_config

    target_val = config.get("target_ratio", 0.8)
    accum_steps = config.get("gradient_accumulation_steps", 16)

    # -------------------------------------------------------------
    # 计算统计量
    # -------------------------------------------------------------
    # 1. 当前这个 micro-batch 的平均 ratio (整张图的平均)
    current_ratio = torch.mean(all_ratios)
    
    # 2. 【补回】层间标准差
    # all_ratios 包含所有层的 ratio，std 反映了层与层之间的差异
    # 如果 std 很大，说明模型学会了“有的层多留，有的层少留”
    ratio_std = torch.std(all_ratios)

    # =================================================================
    # 核心逻辑：Buffer 管理
    # =================================================================
    
    # 1. 初始化 State
    if "history_ratios" not in state_buffer:
        state_buffer["history_ratios"] = []
    if "last_cycle_mean" not in state_buffer:
        state_buffer["last_cycle_mean"] = target_val 
        print(f"[{prefix}] Stratified Buffer Initialized. Target={target_val}")

    # 2. 将当前的数值 (Detach!) 存入 Buffer
    state_buffer["history_ratios"].append(current_ratio.item())

    # 3. 检查是否凑齐了一轮
    if len(state_buffer["history_ratios"]) >= accum_steps:
        local_sum = sum(state_buffer["history_ratios"])
        local_count = len(state_buffer["history_ratios"])
        
        sum_tensor = torch.tensor([local_sum], device=device, dtype=torch.float32)
        
        # DDP 同步
        if dist.is_initialized():
            dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            total_count = local_count * world_size
            cycle_mean = sum_tensor.item() / total_count
        else:
            cycle_mean = local_sum / local_count
            
        state_buffer["last_cycle_mean"] = cycle_mean
        state_buffer["history_ratios"] = []
        
        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_main_process:
             print(f"[{prefix}] Cycle Update: Global Mean Est = {cycle_mean:.4f}")

    # =================================================================
    # 核心逻辑：构造全局梯度
    # =================================================================
    
    global_mean_est = state_buffer["last_cycle_mean"]
    global_diff = global_mean_est - target_val
    grad_scale = 2.0 * global_diff 
    
    # Linear Loss trick
    loss = w_budget_mse * (current_ratio * grad_scale)
    
    # 展示用的 MSE (依然是单样本的，仅供参考)
    display_mse = (current_ratio.detach() - target_val) ** 2

    metrics = {
        f"{prefix}/avg_ratio": current_ratio.detach(),
        f"{prefix}/std_ratio": ratio_std.detach(), # <--- 这里加上了
        f"{prefix}/target_ratio": torch.tensor(target_val, device=device),
        f"{prefix}/cycle_mean_est": torch.tensor(global_mean_est, device=device),
        f"loss/{prefix}/mse_loss": display_mse,
        f"loss/{prefix}/ratio_loss": loss.detach(),
    }
    
    return loss, metrics

# ============================================================
# 外层调用封装 (替换你原来的 calculate_image_ratio_loss)
# ============================================================

def calculate_image_ratio_loss(
    config: dict, 
    device: torch.device,
    trainer_state: dict # <--- 需要传入 trainer 身上挂的一个 dict
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

    ratios_concat = torch.cat(all_ratios_flat).flatten()

    # 调用新的 Stratified Buffer Loss
    # 确保 trainer_state 里有一个唯一的 key 给 image ratio 用
    if "image_ratio_buffer" not in trainer_state:
        trainer_state["image_ratio_buffer"] = {}
        
    loss, metrics = _calc_ratio_loss_stratified(
        ratios_concat, 
        config, 
        device, 
        state_buffer=trainer_state["image_ratio_buffer"], # 传入引用
        prefix="image_ratio"
    )
    
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
    
    if (
        hasattr(ShivaTopkFunc, "_last_gap_stats")
        and ShivaTopkFunc._last_gap_stats
    ):
        stats["topkgrads/gap_mean"] = ShivaTopkFunc._last_gap_stats.get("mean", 0.0)
    
    if (
        hasattr(ShivaTopkFunc, "_last_k_grad_stats")
        and ShivaTopkFunc._last_k_grad_stats
    ):
        stats["topkgrads/grad_k_norm"] = ShivaTopkFunc._last_k_grad_stats.get("norm", 0.0)
        stats["topkgrads/grad_k_mean"] = ShivaTopkFunc._last_k_grad_stats.get("mean", 0.0)
        
        grad_std = ShivaTopkFunc._last_k_grad_stats.get("std", torch.tensor(0.0))
        if torch.isnan(grad_std):
            grad_std = torch.tensor(0.0, device=device)
        stats["topkgrads/grad_k_std"] = grad_std
        
        stats["topkgrads/grad_k_pos_rate"] = ShivaTopkFunc._last_k_grad_stats.get("pos_rate", 0.0)

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

def collect_policy_gradient_stats(model, hook_manager, device) -> dict:
    stats = {}
    target_hook = None
    for hook in hook_manager.get_all_hooks().values():
        if hasattr(hook, 'image_ratio_policy') and hook.image_ratio_policy is not None:
            target_hook = hook
            break
            
    if target_hook is None:
        return stats

    policies = {
        "img_policy": target_hook.image_ratio_policy,
        # "ctx_policy": getattr(target_hook, 'context_ratio_policy', None) 
    }

    for name, policy_net in policies.items():
        if policy_net is None: continue

        # ---------------------------------------------------------------------
        # 1. Layer Embedding Gradients (已修正 Skip Layer 干扰)
        # ---------------------------------------------------------------------
        layer_grad_norm_total = 0.0
        if hasattr(policy_net, 'layer_emb'):
            param = policy_net.layer_emb.weight # Shape: [Total_Layers, Dim]
            if param.grad is not None:
                grad = param.grad.detach()
                
                # 计算每一层 Embedding 的梯度范数, Shape: [Total_Layers]
                layer_norms = grad.norm(dim=1)
                
                # === 关键修正: 过滤掉梯度极小(被Skip)的层 ===
                # 我们假设梯度小于 1e-8 的就是被 Skip 的层
                active_mask = layer_norms > 1e-8
                active_layer_norms = layer_norms[active_mask]
                
                # 记录被 Skip 了多少层
                num_skipped = (~active_mask).sum().item()
                stats[f"grad/{name}/num_skipped_layers_detected"] = num_skipped

                # 只有在有 Active Layer 的情况下才计算统计量
                if active_layer_norms.numel() > 0:
                    # Metric 1: 层间敏感度变异系数 (修正后)
                    # 现在只计算真正参与训练的层之间的差异
                    stats[f"grad/{name}/layer_sensitivity_cv"] = active_layer_norms.std() / (active_layer_norms.mean() + 1e-6)
                    
                    # Metric 2: Layer Max/Min Ratio (修正后)
                    # 现在 Min 不会是 0 了，而是最小的"有效更新"
                    max_l_grad = active_layer_norms.max()
                    min_l_grad = active_layer_norms.min()
                    stats[f"grad/{name}/layer_grad_max_min_ratio"] = max_l_grad / (min_l_grad + 1e-6)
                    
                    # 基础统计也只看 Active 的
                    stats[f"grad/{name}/layer_proj_active_mean"] = active_layer_norms.mean()
                else:
                    # 极端情况：所有层都没梯度
                    stats[f"grad/{name}/layer_sensitivity_cv"] = 0.0
                    stats[f"grad/{name}/layer_grad_max_min_ratio"] = 1.0

                # 总梯度量级 (Total Norm) 不需要过滤，因为加 0 不影响总和
                layer_grad_norm_total = grad.norm()

        # ---------------------------------------------------------------------
        # 2. Time Projection Gradients (这部分不受 Layer Skip 影响)
        # ---------------------------------------------------------------------
        time_grad_norm_total = 0.0
        if hasattr(policy_net, 'time_proj'):
            param = policy_net.time_proj[-1].weight 
            if param.grad is not None:
                grad = param.grad.detach() 
                
                stats[f"grad/{name}/temb_proj_std"] = grad.std()
                grad_norm_per_channel = grad.norm(dim=1) 
                stats[f"grad/{name}/temb_spatial_cv"] = grad_norm_per_channel.std() / (grad_norm_per_channel.mean() + 1e-6)

                out_dim = grad.shape[0]
                scale_grad = grad[:out_dim//2, :]
                shift_grad = grad[out_dim//2:, :]
                
                norm_scale = scale_grad.norm()
                norm_shift = shift_grad.norm()
                
                stats[f"grad/{name}/temb_scale_grad_norm"] = norm_scale
                stats[f"grad/{name}/temb_shift_grad_norm"] = norm_shift
                stats[f"grad/{name}/temb_scale_shift_ratio"] = norm_scale / (norm_shift + 1e-6)
                
                time_grad_norm_total = grad.norm()

        # ---------------------------------------------------------------------
        # 3. Global Balance
        # ---------------------------------------------------------------------
        if layer_grad_norm_total > 0 and time_grad_norm_total > 0:
            stats[f"grad/{name}/balance_layer_vs_time"] = layer_grad_norm_total / (time_grad_norm_total + 1e-6)

        # ---------------------------------------------------------------------
        # 4. Debug Print (修正后)
        # ---------------------------------------------------------------------
        if hasattr(policy_net, 'layer_emb') and policy_net.layer_emb.weight.grad is not None:
            # 使用 float() 转换
            grads = policy_net.layer_emb.weight.grad.detach().norm(dim=1).float().cpu().numpy()
            
            # 区分 Active 和 Skipped 用于打印
            active_indices = np.where(grads > 1e-8)[0]
            skipped_indices = np.where(grads <= 1e-8)[0]
            
            if len(active_indices) > 0:
                active_grads = grads[active_indices]
                
                # 在 Active 内部排序
                sorted_active_args = np.argsort(active_grads)
                top5_local_idx = sorted_active_args[::-1][:5]
                bot5_local_idx = sorted_active_args[:5]
                
                # 映射回全局 Layer ID
                top5_global_idx = active_indices[top5_local_idx]
                bot5_global_idx = active_indices[bot5_local_idx]

                print("\n=== DiT Policy Gradient Diagnosis ===")
                print(f"Skipped Layers (Grad=0): {skipped_indices}")
                print(f"Active Layers Gradients (Top 5): Layer {top5_global_idx} -> Val {grads[top5_global_idx]}")
                print(f"Active Layers Gradients (Bottom 5): Layer {bot5_global_idx} -> Val {grads[bot5_global_idx]}")
                
                ratio = active_grads.max() / (active_grads.min() + 1e-9)
                print(f"Active Layer Max/Min Ratio: {ratio:.2f}")
            else:
                 print("\n=== DiT Policy Gradient Diagnosis ===")
                 print("⚠️ No active layer gradients found.")

    return stats