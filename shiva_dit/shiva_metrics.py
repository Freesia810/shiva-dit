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
    if all_ratios.numel() == 0:
        return torch.tensor(0.0, device=device), {}

    w_config = config.get("ratio_loss_weight", 1.0)
    w_budget_mse = 0.0
    if isinstance(w_config, list):
        if len(w_config) >= 3: w_budget_mse = w_config[2]
    else:
        w_budget_mse = w_config

    target_val = config.get("target_ratio", 0.8)
    accum_steps = config.get("gradient_accumulation_steps", 16)

    current_ratio = torch.mean(all_ratios)
    ratio_std = torch.std(all_ratios)

    if "history_ratios" not in state_buffer:
        state_buffer["history_ratios"] = []
    if "last_cycle_mean" not in state_buffer:
        state_buffer["last_cycle_mean"] = target_val
        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_main_process:
            print(f"[{prefix}] Stratified Buffer Initialized. Target={target_val}")

    state_buffer["history_ratios"].append(current_ratio.item())

    if len(state_buffer["history_ratios"]) >= accum_steps:
        local_sum = sum(state_buffer["history_ratios"])
        local_count = len(state_buffer["history_ratios"])
        
        sum_tensor = torch.tensor([local_sum], device=device, dtype=torch.float32)
        
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

    global_mean_est = state_buffer["last_cycle_mean"]
    global_diff = global_mean_est - target_val
    grad_scale = 2.0 * global_diff 
    
    # Linear Loss trick
    loss = w_budget_mse * (current_ratio * grad_scale)
    display_mse = (current_ratio.detach() - target_val) ** 2

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

    if not all_ratios_flat:
        return torch.tensor(0.0, device=device), {}, {}

    ratios_concat = torch.cat(all_ratios_flat).flatten()

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
        # "ctx_policy": getattr(target_hook, 'context_ratio_policy', None) 
    }

    for name, policy_net in policies.items():
        if policy_net is None: continue

        layer_grad_norm_total = 0.0
        active_layer_stats = None
        skipped_indices = []
        
        if hasattr(policy_net, 'layer_emb'):
            param = policy_net.layer_emb.weight # [Total_Layers, Dim]
            if param.grad is not None:
                grad = param.grad.detach()
                layer_norms = grad.norm(dim=1) # [Total_Layers]
                
                active_mask = layer_norms > 1e-8
                active_layer_norms = layer_norms[active_mask]
                skipped_indices = torch.where(~active_mask)[0].cpu().tolist()
                
                num_skipped = len(skipped_indices)
                stats[f"grad/{name}/num_skipped_layers"] = num_skipped

                if active_layer_norms.numel() > 0:
                    layer_mean = active_layer_norms.mean()
                    layer_std = active_layer_norms.std()
                    
                    stats[f"grad/{name}/layer_proj_active_mean"] = layer_mean
                    stats[f"grad/{name}/layer_sensitivity_cv"] = layer_std / (layer_mean + 1e-6)
                    stats[f"grad/{name}/layer_max_min_ratio"] = active_layer_norms.max() / (active_layer_norms.min() + 1e-6)
                    
                    active_layer_stats = {
                        "norms": layer_norms.cpu().numpy(),
                        "active_indices": torch.where(active_mask)[0].cpu().numpy(),
                        "active_vals": active_layer_norms.cpu().numpy(),
                        "mean": layer_mean.item()
                    }
                
                layer_grad_norm_total = grad.norm().item()

        time_grad_norm_total = 0.0
        time_stats = None
        
        if hasattr(policy_net, 'time_proj'):
            if isinstance(policy_net.time_proj, torch.nn.Sequential):
                param = policy_net.time_proj[-1].weight
            elif isinstance(policy_net.time_proj, torch.nn.Linear):
                param = policy_net.time_proj.weight
            else:
                param = None

            if param is not None and param.grad is not None:
                grad = param.grad.detach()
                time_grad_norm_total = grad.norm().item()
                
                out_dim = grad.shape[0]
                scale_grad = grad[:out_dim//2, :]
                shift_grad = grad[out_dim//2:, :]
                
                norm_scale = scale_grad.norm().item()
                norm_shift = shift_grad.norm().item()
                
                stats[f"grad/{name}/temb_norm_total"] = time_grad_norm_total
                stats[f"grad/{name}/temb_scale_norm"] = norm_scale
                stats[f"grad/{name}/temb_shift_norm"] = norm_shift
                stats[f"grad/{name}/temb_scale_shift_ratio"] = norm_scale / (norm_shift + 1e-6)
                
                time_stats = {"scale": norm_scale, "shift": norm_shift}

        mlp_grad_norm_total = 0.0
        if hasattr(policy_net, 'mix_mlp'):
            for p in policy_net.mix_mlp.parameters():
                if p.grad is not None:
                    mlp_grad_norm_total += p.grad.detach().norm().item()
            stats[f"grad/{name}/mlp_head_norm"] = mlp_grad_norm_total

        if layer_grad_norm_total > 0 and time_grad_norm_total > 0:
            balance_ratio = layer_grad_norm_total / (time_grad_norm_total + 1e-9)
            stats[f"grad/{name}/balance_layer_vs_time"] = balance_ratio

        if (layer_grad_norm_total > 0 or time_grad_norm_total > 0) and torch.rand(1) < 0.1:
            print(f"\n{'-'*30} Policy Gradient Diagnosis: {name.upper()} {'-'*30}")
            
            # 1. Component Balance Bar
            total_energy = layer_grad_norm_total + time_grad_norm_total + mlp_grad_norm_total + 1e-9
            l_pct = layer_grad_norm_total / total_energy
            t_pct = time_grad_norm_total / total_energy
            m_pct = mlp_grad_norm_total / total_energy
            
            def draw_bar(val, length=20):
                fill = int(val * length)
                return f"|{'#'*fill}{'-'*(length-fill)}|"

            print(f"  > Component Impact Distribution:")
            print(f"    Layer Emb : {draw_bar(l_pct)} {l_pct*100:5.1f}% (Norm: {layer_grad_norm_total:.4f})")
            print(f"    Time Proj : {draw_bar(t_pct)} {t_pct*100:5.1f}% (Norm: {time_grad_norm_total:.4f})")
            print(f"    Mix MLP   : {draw_bar(m_pct)} {m_pct*100:5.1f}% (Norm: {mlp_grad_norm_total:.4f})")
            
            if time_grad_norm_total > 1e-9:
                ratio_lt = layer_grad_norm_total / time_grad_norm_total
                dom = "LAYER" if ratio_lt > 1.0 else "TIME"
                print(f"    Force Ratio (Layer/Time): {ratio_lt:.3f}x --> Dominant: [{dom}]")

            # 2. Time Branch Details
            if time_stats:
                s_ratio = time_stats['scale'] / (time_stats['scale'] + time_stats['shift'] + 1e-9)
                print(f"  > Time Branch Detail:")
                print(f"    Scale vs Shift: {time_stats['scale']:.4f} vs {time_stats['shift']:.4f} (Scale share: {s_ratio*100:.1f}%)")

            # 3. Active Layer Details
            if active_layer_stats:
                vals = active_layer_stats['active_vals']
                idxs = active_layer_stats['active_indices']
                
                # Sort to find outliers
                sorted_args = np.argsort(vals)
                top_k = sorted_args[::-1][:5] # Top 5
                bot_k = sorted_args[:5]       # Bottom 5
                
                print(f"  > Layer Branch Detail (Active: {len(idxs)}, Skipped: {len(skipped_indices)}):")
                
                # Format Lists
                top_str = ", ".join([f"L{idxs[i]}({vals[i]:.4f})" for i in top_k])
                bot_str = ", ".join([f"L{idxs[i]}({vals[i]:.4f})" for i in bot_k])
                
                print(f"    Top-5 Gradients : {top_str}")
                print(f"    Bot-5 Gradients : {bot_str}")
                
                # Check for uneven training
                cv = vals.std() / (vals.mean() + 1e-9)
                if cv > 1.0:
                    print(f"    ⚠️  High Variance (CV={cv:.2f}): Some layers are learning much faster than others!")
            else:
                print(f"  > Layer Branch Detail: No active gradients found (All skipped?).")

            print(f"{'-'*90}")

    return stats