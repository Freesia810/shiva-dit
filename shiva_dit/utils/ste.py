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

        scores_sorted, idx_sorted = torch.sort(scores, dim=-1, descending=True)
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

        ctx.save_for_backward(x, scores, k, topk_idx, rest_idx, T)
        return x_selected, x_rejected, topk_idx, rest_idx

    @staticmethod
    def backward(ctx, grad_sel, grad_rej, grad_topk_idx, grad_rest_idx):
        x, scores, k, topk_idx, rest_idx, T = ctx.saved_tensors
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

        idx = torch.arange(N, device=x.device).float()
        soft_mask = torch.sigmoid((k - idx) * T).view(1, N, 1)

        # grad wrt scores
        grad_scores = torch.sum(grad_x * x * soft_mask, dim=-1)

        # grad wrt k
        dsoft_dk = soft_mask * (1 - soft_mask) * T
        grad_k = torch.sum((x * dsoft_dk) * grad_x)

        # d(soft)/dT = (k - idx) * soft*(1-soft)
        dsoft_dT = (k - idx) * soft_mask * (1 - soft_mask)
        grad_T = torch.sum((x * dsoft_dT) * grad_x)

        return grad_x, grad_scores, grad_k, grad_T


ceil_ste = CeilSTE.apply
sigmoid_ste = SigmoidSTE.apply
