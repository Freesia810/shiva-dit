import torch
import torch.nn as nn
import torch.nn.functional as F

class CeilSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ceil(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit):
        return torch.sigmoid(logit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ShivaTopkFunc(torch.autograd.Function):
    _stats_buffer = {}

    @staticmethod
    def _differentiable_rank_descending(
        scores: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        B, N = scores.shape
        scores_rows = scores.unsqueeze(2)
        scores_cols = scores.unsqueeze(1)
        pairwise_diffs = scores_cols - scores_rows
        soft_comparisons = torch.sigmoid(pairwise_diffs * temperature)

        mask = torch.eye(N, device=scores.device, dtype=torch.bool).unsqueeze(0)
        soft_comparisons_masked = torch.where(mask, 0.0, soft_comparisons)

        soft_ranks = 1 + soft_comparisons_masked.sum(dim=2)
        return soft_ranks

    @staticmethod
    def forward(ctx, x, logits, k_float, Ts, Tk):
        B, N, D = x.shape
        idx_sorted = torch.argsort(logits, dim=-1, descending=True)
        k_int = int(torch.ceil(k_float).item())
        k_int = max(1, min(k_int, N))

        topk_idx = idx_sorted[:, :k_int]
        rest_idx = idx_sorted[:, k_int:]

        x_selected = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(B, k_int, D))
        x_rejected = torch.gather(x, 1, rest_idx.unsqueeze(-1).expand(B, N - k_int, D))
        ctx.save_for_backward(x, logits, k_float, Ts, Tk, topk_idx, rest_idx)
        return x_selected, x_rejected, topk_idx.detach(), rest_idx.detach()

    @staticmethod
    def backward(ctx, grad_x_selected, grad_x_rejected, grad_topk_idx, grad_rest_idx):
        x, logits, k_float, Ts, Tk, topk_idx, rest_idx = ctx.saved_tensors
        B, N, D = x.shape
        k_int = topk_idx.shape[1]

        grad_x = torch.zeros_like(x)
        if grad_x_selected is not None:
            grad_x.scatter_add_(1, topk_idx.unsqueeze(-1).expand_as(grad_x_selected), grad_x_selected)
        if grad_x_rejected is not None:
            grad_x.scatter_add_(1, rest_idx.unsqueeze(-1).expand_as(grad_x_rejected), grad_x_rejected)

        # dL/dp = (dL/dx_sel - dL/dx_rej) * x
        grad_x_sel_scattered = torch.zeros_like(x)
        if grad_x_selected is not None:
            grad_x_sel_scattered.scatter_add_(1, topk_idx.unsqueeze(-1).expand_as(grad_x_selected), grad_x_selected)

        grad_x_rej_scattered = torch.zeros_like(x)
        if k_int < N and grad_x_rejected is not None:
            grad_x_rej_scattered.scatter_add_(1, rest_idx.unsqueeze(-1).expand_as(grad_x_rejected), grad_x_rejected)

        grad_probs = torch.sum((grad_x_sel_scattered - grad_x_rej_scattered) * x.detach(), dim=-1)

        logits_detached = logits.detach().requires_grad_(True)
        k_detached = k_float.detach().requires_grad_(True)
        
        with torch.enable_grad():
            scores_for_soft_rank = torch.sigmoid(logits_detached)
            noise = torch.randn_like(logits_detached) * 1e-5
            scores_for_soft_rank = scores_for_soft_rank + noise

            soft_ranks = ShivaTopkFunc._differentiable_rank_descending(scores_for_soft_rank, Ts)
            N_float = float(N)

            dist_to_k = k_detached.unsqueeze(-1) - soft_ranks
            z = Tk * dist_to_k / N_float
            selection_probs = torch.sigmoid(z)

            grads = torch.autograd.grad(
                outputs=selection_probs,
                inputs=[logits_detached, k_detached],
                grad_outputs=grad_probs,
                allow_unused=True
            )
            grad_logits, grad_k = grads[0], grads[1]


        if torch.rand(1) < 1e-4:
            with torch.no_grad():
                g_logits_norm = grad_logits.norm().item()
                g_k_norm = grad_k.norm().item() if grad_k is not None else 0.0
                g_k_val = grad_k.mean().item() if grad_k is not None else 0.0
                
                in_transition = (selection_probs > 0.1) & (selection_probs < 0.9)
                width_mean = in_transition.float().sum(dim=1).mean().item()
                
                active_grads = (grad_logits.abs() > 1e-8).float().sum(dim=1).mean().item()
                
                rank_spread = soft_ranks.std().item()

                ShivaTopkFunc._stats_buffer = {
                    "grad_logits_norm": g_logits_norm,
                    "grad_k_norm": g_k_norm,
                    "grad_k_mean": g_k_val,
                    "soft_boundary_width": width_mean,
                    "active_grad_tokens": active_grads,
                    "rank_spread": rank_spread,
                    "Ts": Ts.item() if isinstance(Ts, torch.Tensor) else Ts,
                    "Tk": Tk.item() if isinstance(Tk, torch.Tensor) else Tk,
                    "k_target": k_float.item(),
                    "N": N
                }

                try:
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        _print_console_stats(ShivaTopkFunc._stats_buffer)
                except:
                    _print_console_stats(ShivaTopkFunc._stats_buffer)

        return grad_x, grad_logits, grad_k, None, None

def _print_console_stats(stats):
    """Auxiliary function for beautiful printing"""
    print(f"\n{'-'*30} [TopK Autograd Monitor] {'-'*30}")
    
    # 1. Gradient Health
    print(f"  > Gradient Health:")
    print(f"    Grad Logits Norm : {stats['grad_logits_norm']:.4f}")
    print(f"    Grad K (Mean)    : {stats['grad_k_mean']:.4f} (Norm: {stats['grad_k_norm']:.4f})")
    
    # 2. Temperature & Selection Analysis
    width = stats['soft_boundary_width']
    active = stats['active_grad_tokens']
    N = stats['N']
    
    print(f"  > Softness & Participation (N={N}, k~={stats['k_target']:.1f}):")
    print(f"    Ts (Rank Temp)   : {stats['Ts']:.2f}")
    print(f"    Tk (Sel Temp)    : {stats['Tk']:.2f}")
    print(f"    Boundary Width   : {width:.1f} tokens ({width/N*100:.1f}%) [Tokens with 0.1<p<0.9]")
    print(f"    Active Gradients : {active:.1f} tokens ({active/N*100:.1f}%) [Tokens with |g|>1e-8]")
    
    # 3. Diagnosis
    if active < 2.0:
        print(f"    ⚠️  Gradient Vanishing! (Too Hard? Decrease Ts/Tk or check upstream)")
    elif active > N * 0.9:
        print(f"    ⚠️  All Tokens Updating! (Too Soft? Increase Ts/Tk to enforce sparsity)")
    else:
        print(f"    ✅  Healthy Sparsity.")
        
    print(f"{'-'*83}")

ceil_ste = CeilSTE.apply
sigmoid_ste = SigmoidSTE.apply
