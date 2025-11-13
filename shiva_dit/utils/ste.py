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
    def forward(ctx, x, scores, k, Tk, Ts):
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

        ctx.save_for_backward(x, k, topk_idx, rest_idx, ranks, Tk, Ts)
        return x_selected, x_rejected, topk_idx, rest_idx

    @staticmethod
    def backward(ctx, grad_sel, grad_rej, grad_topk_idx, grad_rest_idx):
        x, k, topk_idx, rest_idx, ranks, T_k, T_s = ctx.saved_tensors
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

        grad_L_wrt_soft_mask_scalar = torch.sum(grad_x * x, dim=-1)  # (B, N)
        grad_norm = torch.norm(grad_L_wrt_soft_mask_scalar, p=2, dim=-1, keepdim=True) + 1e-9
        normalized_grad_scalar = grad_L_wrt_soft_mask_scalar / grad_norm
        normalized_grad_scalar_expanded = normalized_grad_scalar.unsqueeze(-1) # (B, N, 1)

        soft_mask_s = torch.sigmoid((k - ranks_expanded) * T_s)  # (B, N, 1)
        d_soft_s_d_logits = soft_mask_s * (1 - soft_mask_s) # (B, N, 1)
        d_soft_d_scores = T_s * d_soft_s_d_logits
        grad_scores = (normalized_grad_scalar_expanded * d_soft_d_scores).squeeze(-1)  # (B, N)

        soft_mask_k = torch.sigmoid((k - ranks_expanded) * T_k) # (B, N, 1)
        d_soft_k_d_logits = soft_mask_k * (1 - soft_mask_k) # (B, N, 1)
        dsoft_dk = d_soft_k_d_logits * T_k
        grad_k = torch.sum(normalized_grad_scalar_expanded * dsoft_dk)

        dsoft_dk_dT = (k - ranks_expanded) * d_soft_k_d_logits
        grad_T_k = torch.sum(normalized_grad_scalar_expanded * dsoft_dk_dT)

        dsoft_ds_dT = (k - ranks_expanded) * d_soft_s_d_logits
        grad_T_s = torch.sum(normalized_grad_scalar_expanded * dsoft_ds_dT)

        return grad_x, grad_scores, grad_k, grad_T_k, grad_T_s

    # @staticmethod
    # def backward(ctx, grad_sel, grad_rej, grad_topk_idx, grad_rest_idx):
    #     x, k, topk_idx, rest_idx, ranks, T = ctx.saved_tensors
    #     B, N, D = x.shape
    #     k_int = topk_idx.size(1)

    #     grad_x = torch.zeros_like(x)
    #     grad_x.scatter_(
    #         1,
    #         topk_idx.unsqueeze(-1).expand(B, k_int, D),
    #         grad_sel,
    #     )
    #     grad_x.scatter_(
    #         1,
    #         rest_idx.unsqueeze(-1).expand(B, N - k_int, D),
    #         grad_rej,
    #     )

    #     ranks_expanded = ranks.unsqueeze(-1)  # (B, N, 1)
    #     soft_mask = torch.sigmoid((k - ranks_expanded) * T)  # (B, N, 1)

    #     grad_L_wrt_soft_mask_scalar = torch.sum(grad_x * x, dim=-1)  # (B, N)
    #     grad_norm = torch.norm(grad_L_wrt_soft_mask_scalar, p=2, dim=-1, keepdim=True) + 1e-9
    #     normalized_grad_scalar = grad_L_wrt_soft_mask_scalar / grad_norm

    #     # Derivative of sigmoid((k - ranks)*T)
    #     d_soft_d_logits = soft_mask * (1 - soft_mask)
    #     d_soft_d_scores = T * d_soft_d_logits

    #     grad_scores = (normalized_grad_scalar.unsqueeze(-1) * d_soft_d_scores).squeeze(-1)  # (B, N)

    #     # ---- Gradients wrt k and T ----
    #     dsoft_dk = d_soft_d_logits * T
    #     grad_k = torch.sum(normalized_grad_scalar.unsqueeze(-1) * dsoft_dk)

    #     dsoft_dT = (k - ranks_expanded) * d_soft_d_logits
    #     grad_T = torch.sum(normalized_grad_scalar.unsqueeze(-1) * dsoft_dT)

    #     return grad_x, grad_scores, grad_k, grad_T

ceil_ste = CeilSTE.apply
sigmoid_ste = SigmoidSTE.apply
