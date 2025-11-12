import torch


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
    @staticmethod
    def forward(ctx, x, scores, k, T):
        B, N, D = x.shape
        k_int = int(k.item())

        _, idx_sorted = torch.sort(scores, dim=-1, descending=True)
        topk_idx = idx_sorted[:, :k_int]
        rest_idx = idx_sorted[:, k_int:]

        x_selected = torch.gather(
            x, 1,
            topk_idx.unsqueeze(-1).expand(B, k_int, D)
        )
        x_rejected = torch.gather(
            x, 1,
            rest_idx.unsqueeze(-1).expand(B, N - k_int, D)
        )

        ranks = torch.argsort(idx_sorted, dim=-1).float()  # (B, N)

        ctx.save_for_backward(x, k, topk_idx, rest_idx, ranks, T)
        return x_selected, x_rejected, topk_idx, rest_idx

    @staticmethod
    def backward(ctx, grad_sel, grad_rej, grad_topk_idx, grad_rest_idx):
        x, k, topk_idx, rest_idx, ranks, T = ctx.saved_tensors
        B, N, D = x.shape
        k_int = topk_idx.size(1)

        grad_x = torch.zeros_like(x)
        grad_x.scatter_(
            1,
            topk_idx.unsqueeze(-1).expand(B, k_int, D),
            grad_sel,
        )
        grad_x.scatter_(
            1,
            rest_idx.unsqueeze(-1).expand(B, N - k_int, D),
            grad_rej,
        )

        ranks_expanded = ranks.unsqueeze(-1)  # (B, N, 1)
        soft_mask = torch.sigmoid((k - ranks_expanded) * T)  # (B, N, 1)

        grad_L_wrt_soft_mask_scalar = torch.sum(grad_x * x, dim=-1)  # (B, N)

        grad_scores = grad_L_wrt_soft_mask_scalar * soft_mask.squeeze(-1)

        dsoft_dk = soft_mask * (1 - soft_mask) * T  # (B, N, 1)
        grad_k = torch.sum(grad_L_wrt_soft_mask_scalar.unsqueeze(-1) * dsoft_dk)

        dsoft_dT = (k - ranks_expanded) * soft_mask * (1 - soft_mask)  # (B, N, 1)
        grad_T = torch.sum(grad_L_wrt_soft_mask_scalar.unsqueeze(-1) * dsoft_dT)

        return grad_x, grad_scores, grad_k, grad_T


ceil_ste = CeilSTE.apply
sigmoid_ste = SigmoidSTE.apply
