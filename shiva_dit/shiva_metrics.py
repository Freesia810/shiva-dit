import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import random
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
    w_config = config.get("ratio_loss_weight", 1.0)
    w_budget_mse = 0.0
    if isinstance(w_config, list):
        if len(w_config) >= 3: w_budget_mse = w_config[2]
    else:
        w_budget_mse = w_config

    target_val = config.get("target_ratio", 0.8)
    accum_steps = config.get("gradient_accumulation_steps", 16)
    
    DEBUG_PRINT_PROB = 0.01
    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    if all_ratios.numel() > 0:
        current_ratio = torch.mean(all_ratios)
        ratio_std = torch.std(all_ratios)
        has_data = True
    else:
        current_ratio = torch.tensor(target_val, device=device)
        ratio_std = torch.tensor(0.0, device=device)
        has_data = False

    if "history_ratios" not in state_buffer:
        state_buffer["history_ratios"] = []
    if "history_counts" not in state_buffer:
        state_buffer["history_counts"] = []
        
    if "last_cycle_mean" not in state_buffer:
        state_buffer["last_cycle_mean"] = target_val

    if has_data:
        state_buffer["history_ratios"].append(current_ratio.item())
        state_buffer["history_counts"].append(1.0)
    else:
        state_buffer["history_ratios"].append(0.0)
        state_buffer["history_counts"].append(0.0)

    if len(state_buffer["history_ratios"]) >= accum_steps:
        if dist.is_initialized():
            dist.barrier() 

        local_sum = sum(state_buffer["history_ratios"])
        local_valid_count = sum(state_buffer["history_counts"])
        
        reduce_tensor = torch.tensor([local_sum, local_valid_count], device=device, dtype=torch.float32)
        
        # DDP AllReduce
        if dist.is_initialized():
            dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        
        global_sum = reduce_tensor[0].item()
        global_count = reduce_tensor[1].item()
        
        if global_count == 0:
            raise RuntimeError(f"[{prefix}] Error: All ranks reported NO data during accumulation cycle. Cannot compute average.")
            
        cycle_mean = global_sum / global_count
            
        state_buffer["last_cycle_mean"] = cycle_mean
        state_buffer["history_ratios"] = []
        state_buffer["history_counts"] = []
        
    global_mean_est = state_buffer["last_cycle_mean"]
    global_diff = global_mean_est - target_val
    grad_scale = 2.0 * global_diff 
    
    if has_data:
        # Policy Loss: current_ratio * grad_scale
        loss = w_budget_mse * (current_ratio * grad_scale)
        display_mse = (current_ratio.detach() - target_val) ** 2
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        display_mse = torch.tensor(0.0, device=device)

    if is_main_process and random.random() < DEBUG_PRINT_PROB:
        log_cycle_mean = global_mean_est if 'cycle_mean' not in locals() else cycle_mean
        log_est_mean = global_mean_est
        log_status = "CYCLE_UPDATE" if 'cycle_mean' in locals() else "ACCUMULATING"
        
        print("-" * 60)
        print(f"[{prefix.upper()} DEBUG LOG] Status: {log_status}")
        print(f"  > Target: {target_val:.4f} | W_MSE: {w_budget_mse:.4f}")
        print(f"  > Current Ratio (Micro): {current_ratio.item():.4f}")
        print(f"  > Estimated Cycle Mean: {log_est_mean:.4f}")
        print(f"  > Global Diff: {global_diff:.4f} | Grad Scale: {grad_scale:.4f}")
        print(f"  > Loss Component: {loss.item():.6f}")
        print("-" * 60)

    metrics = {
        f"{prefix}/avg_ratio": current_ratio.detach(),
        f"{prefix}/std_ratio": ratio_std.detach(),
        f"{prefix}/target_ratio": torch.tensor(target_val, device=device),
        f"{prefix}/cycle_mean_est": torch.tensor(global_mean_est, device=device),
        f"loss/{prefix}/mse_loss": display_mse,
        f"loss/{prefix}/ratio_loss": loss.detach(),
    }
    
    return loss, metrics

def calculate_image_ratio_loss(
    config: dict, 
    device: torch.device,
    trainer_state: dict
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

    if all_ratios_flat:
        ratios_concat = torch.cat(all_ratios_flat).flatten()
    else:
        ratios_concat = torch.tensor([], device=device)

    if "image_ratio_buffer" not in trainer_state:
        trainer_state["image_ratio_buffer"] = {}
        
    loss, metrics = _calc_ratio_loss_stratified(
        ratios_concat, 
        config, 
        device, 
        state_buffer=trainer_state["image_ratio_buffer"],
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
    
    buffer = getattr(ShivaTopkFunc, "_stats_buffer", {})
    if not buffer:
        return stats

    stats["topk/grad/logits_norm"] = buffer.get("grad_logits_norm", 0.0)
    stats["topk/grad/k_norm"] = buffer.get("grad_k_norm", 0.0)
    stats["topk/grad/k_mean"] = buffer.get("grad_k_mean", 0.0)

    stats["topk/status/active_tokens"] = buffer.get("active_grad_tokens", 0.0)
    stats["topk/status/boundary_width"] = buffer.get("soft_boundary_width", 0.0)
    stats["topk/status/rank_spread"] = buffer.get("rank_spread", 0.0)
    
    stats["topk/status/k_target"] = buffer.get("k_target", 0.0)

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
    }

    verbose_log = False 
    if torch.rand(1).item() < 0.5: 
        verbose_log = True

    for name, policy_net in policies.items():
        if policy_net is None: continue

        # Layer Path
        grad_l_emb = torch.tensor(0.0, device=device)
        grad_l_mlp = torch.tensor(0.0, device=device)
        
        # Time Path
        grad_t_mlp = torch.tensor(0.0, device=device)
        
        # Bias
        grad_bias = torch.tensor(0.0, device=device)

        # ============================================================
        # 1. Layer Branch Analysis (拟合空间差异的主力)
        # ============================================================
        
        # 1.1 Layer Embedding
        if hasattr(policy_net, 'layer_emb'):
            param = policy_net.layer_emb.weight
            if param.grad is not None:
                grad = param.grad.detach()
                grad_l_emb = grad.norm()
                
                # 统计活跃层的梯度分布 (检查是否只有部分层在更新)
                layer_norms = grad.norm(dim=1)
                active_mask = layer_norms > 1e-8
                active_layer_norms = layer_norms[active_mask]
                
                if active_layer_norms.numel() > 0:
                    layer_mean = active_layer_norms.mean()
                    layer_std = active_layer_norms.std()
                    stats[f"grad/{name}/layer_emb_mean"] = layer_mean.item()
                    stats[f"grad/{name}/layer_emb_cv"] = (layer_std / (layer_mean + 1e-6)).item()

        # 1.2 Layer MLP
        if hasattr(policy_net, 'layer_mlp'):
            grad_list = [p.grad.detach().norm().square() for p in policy_net.layer_mlp.parameters() if p.grad is not None]
            if grad_list:
                grad_l_mlp = torch.stack(grad_list).sum().sqrt()
                stats[f"grad/{name}/layer_mlp_norm"] = grad_l_mlp.item()

        # ============================================================
        # 2. Time Branch Analysis (应该保持安静的辅助分支)
        # ============================================================
        if hasattr(policy_net, 'time_mlp'):
            grad_list = [p.grad.detach().norm().square() for p in policy_net.time_mlp.parameters() if p.grad is not None]
            if grad_list:
                grad_t_mlp = torch.stack(grad_list).sum().sqrt()
                stats[f"grad/{name}/time_mlp_norm"] = grad_t_mlp.item()

        # ============================================================
        # 3. Global Bias Analysis (全局锚点)
        # ============================================================
        if hasattr(policy_net, 'global_bias') and policy_net.global_bias.grad is not None:
            grad_bias = policy_net.global_bias.grad.detach().norm()
            stats[f"grad/{name}/global_bias_norm"] = grad_bias.item()

        # ============================================================
        # 4. Visualization & Logging
        # ============================================================
        
        total_l_energy = grad_l_emb + grad_l_mlp
        total_t_energy = grad_t_mlp
        total_energy = total_l_energy + total_t_energy + grad_bias + 1e-9
        
        l_pct = total_l_energy / total_energy
        t_pct = total_t_energy / total_energy
        b_pct = grad_bias / total_energy

        if verbose_log and total_energy > 1e-8:
            def draw_bar(val, length=25):
                fill = int(val * length)
                return f"|{'#'*fill}{'-'*(length-fill)}|"

            print(f"\n{'-'*30} [Policy Gradient Health: {name.upper()}] {'-'*30}")
            print(f"  > Gradient Energy Distribution (Target: L >> T):")
            print(f"    [L] Layer Branch : {draw_bar(l_pct)} {l_pct*100:5.1f}% (Norm: {total_l_energy:.4f})")
            print(f"    [T] Time Branch  : {draw_bar(t_pct)} {t_pct*100:5.1f}% (Norm: {total_t_energy:.4f})")
            print(f"    [B] Global Bias  : {draw_bar(b_pct)} {b_pct*100:5.1f}% (Norm: {grad_bias:.4f})")
            
            print(f"\n  > Detailed Components:")
            print(f"    Layer Emb Norm : {grad_l_emb:.4f} (CV: {stats.get(f'grad/{name}/layer_emb_cv', 0):.2f})")
            print(f"    Layer MLP Norm : {grad_l_mlp:.4f}")
            print(f"    Time MLP Norm  : {grad_t_mlp:.4f} {'(Healthy: Low)' if grad_t_mlp < grad_l_mlp else '(Warning: High)'}")
            
            if t_pct > l_pct:
                print(f"    ⚠️  ALARM: Time branch is dominating gradients! Check initialization (should be zero).")
            elif l_pct > 0.9:
                print(f"    ✅  HEALTHY: Spatial adaptivity is driving the learning.")
                
            print(f"{'-'*92}")

    return stats