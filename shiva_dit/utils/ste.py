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
    _last_grad_probs_stats = {}
    _last_soft_ranks_stats = {}
    _last_z_stats = {}
    _last_selection_probs_stats = {}

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
    def forward(ctx, x, scores, k_float, Ts, Tk):
        B, N, D = x.shape

        k_int = int(torch.ceil(k_float).long().item())
        k_int = max(1, min(k_int, N))

        _, idx_sorted = torch.sort(scores, dim=-1, descending=True)

        topk_idx = idx_sorted[:, :k_int]
        rest_idx = idx_sorted[:, k_int:]

        x_selected = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(B, k_int, D))
        x_rejected = torch.gather(x, 1, rest_idx.unsqueeze(-1).expand(B, N - k_int, D))

        ctx.save_for_backward(x, scores, k_float, Ts, Tk, topk_idx, rest_idx)

        return x_selected, x_rejected, topk_idx.detach(), rest_idx.detach()

    @staticmethod
    def backward(ctx, grad_x_selected, grad_x_rejected, grad_topk_idx, grad_rest_idx):
        x, scores, k_float, Ts, Tk, topk_idx, rest_idx = ctx.saved_tensors
        B, N, D = x.shape
        k_int = topk_idx.shape[1]

        grad_x = None
        grad_scores = None
        grad_k = None
        grad_Ts = None
        grad_Tk = None

        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(
            1, topk_idx.unsqueeze(-1).expand_as(grad_x_selected), grad_x_selected
        )
        grad_x.scatter_add_(
            1, rest_idx.unsqueeze(-1).expand_as(grad_x_rejected), grad_x_rejected
        )

        # dL/dx_sel (B, N, D)
        grad_x_sel_scattered = torch.zeros_like(x)
        grad_x_sel_scattered.scatter_add_(
            1, topk_idx.unsqueeze(-1).expand_as(grad_x_selected), grad_x_selected
        )

        # dL/dx_rej (B, N, D)
        grad_x_rej_scattered = torch.zeros_like(x)
        if k_int < N and grad_x_rejected is not None:
            grad_x_rej_scattered.scatter_add_(
                1, rest_idx.unsqueeze(-1).expand_as(grad_x_rejected), grad_x_rejected
            )

        # dL/dp = (dL/dx_sel - dL/dx_rej) * x
        grad_probs = torch.sum(
            (grad_x_sel_scattered - grad_x_rej_scattered) * x.detach(), dim=-1
        )  # Shape: [B, N]

        # --- dL/d(scores), dL/d(k), dL/d(T) ---
        scores_detached = scores.detach().requires_grad_(True)
        k_detached = k_float.detach().requires_grad_(True)
        Ts_detached = Ts
        Tk_detached = Tk

        with torch.enable_grad():
            T_rank = F.softplus(Ts_detached)
            T_select = F.softplus(Tk_detached)

            soft_ranks = ShivaTopkFunc._differentiable_rank_descending(
                scores_detached, T_rank
            )

            z = T_select * (k_detached.unsqueeze(-1) - soft_ranks)
            selection_probs = torch.sigmoid(z)  # Shape: [B, N]

            if torch.rand(1) < 0.05:
                with torch.no_grad():
                    ShivaTopkFunc._last_grad_probs_stats = {
                        "norm": torch.norm(grad_probs).item(),
                        "mean": grad_probs.mean().item(),
                        "std": grad_probs.std().item(),
                        "sum": grad_probs.sum().item(),
                    }

                    ShivaTopkFunc._last_soft_ranks_stats = {
                        "mean": soft_ranks.mean().item(),
                        "std": soft_ranks.std().item(),
                        "min": soft_ranks.min().item(),
                        "max": soft_ranks.max().item(),
                    }

                    ShivaTopkFunc._last_z_stats = {
                        "mean": z.mean().item(),
                        "std": z.std().item(),
                    }

                    sigmoid_prime = selection_probs * (1 - selection_probs)
                    ShivaTopkFunc._last_selection_probs_stats = {
                        "grad_mean": sigmoid_prime.mean().item(),
                        "grad_std": sigmoid_prime.std().item(),
                    }

        grads = torch.autograd.grad(
            outputs=selection_probs,
            inputs=[scores_detached, k_detached],
            grad_outputs=grad_probs,
            allow_unused=True
        )

        grad_scores = grads[0]
        grad_k = grads[1]

        return grad_x, grad_scores, grad_k, grad_Ts, grad_Tk


ceil_ste = CeilSTE.apply
sigmoid_ste = SigmoidSTE.apply
