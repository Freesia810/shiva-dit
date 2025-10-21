import torch


class CeilSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ceil(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ceil_ste = CeilSTE.apply