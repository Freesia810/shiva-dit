import torch
from shiva_dit.utils.ste import ShivaTopkFunc

if __name__ == "__main__":
    import torch.nn as nn

    B = 2
    N = 5
    D = 3

    print(f"Batch Size (B): {B}, Num Items (N): {N}, Dim (D): {D}")

    class ShivaPrunerExpert(nn.Module):
        def __init__(
            self,
            initial_Ts: float = 1.0,
            initial_Tk: float = 10.0,
            learnable_T: bool = True,
        ):
            super().__init__()
            if learnable_T:
                self.shiva_Ts = nn.Parameter(torch.tensor(initial_Ts))  # Score-Temp
                self.shiva_Tk = nn.Parameter(torch.tensor(initial_Tk))  # K-Temp
            else:
                self.register_buffer("shiva_Ts", torch.tensor(initial_Ts))
                self.register_buffer("shiva_Tk", torch.tensor(initial_Tk))

        def forward(self, x, scores, k_float):
            return ShivaTopkFunc.apply(x, scores, k_float, self.shiva_Ts, self.shiva_Tk)

        def __repr__(self):
            return (
                f"{self.__class__.__name__}("
                f"shiva_Ts(T-Rank)={self.shiva_Ts.item():.2f}, "
                f"shiva_Tk(T-Select)={self.shiva_Tk.item():.2f})"
            )

    pruner = ShivaPrunerExpert(initial_Ts=1.0, initial_Tk=10.0, learnable_T=True)
    print(f"\nPruner: {pruner}")

    x = torch.randn(B, N, D, requires_grad=True)

    scores = torch.tensor(
        [[1.0, 5.0, 3.0, 2.0, 4.0], [8.0, 7.0, 6.0, 5.0, 9.0]], requires_grad=True
    )

    k_float = torch.tensor(2.0, requires_grad=True)

    print(f"\n--- 1. Forward (k=2.0) ---")

    x_sel, x_rej, topk_idx, rest_idx = pruner(x, scores, k_float)

    print(f"Top-k Indices (k_int=2):\n{topk_idx}")
    print(f"\nx_selected shape: {x_sel.shape}")
    print(f"x_rejected shape: {x_rej.shape}")

    assert x_sel.shape == (B, 2, D)
    print("\n✅ Shape of x_selected is correct.")

    loss = x_sel.sum() - x_rej.sum()
    loss.backward()

    print(f"\n--- 2. Checking grads ---")

    def check_grad(name, tensor):
        if tensor.grad is not None:
            grad_norm = torch.norm(tensor.grad)
            if grad_norm > 0:
                print(f"{name} grad ok (Norm: {grad_norm:.4f})")
            else:
                print(f"{name} grad = 0")
        else:
            print(f"{name} grad is None!")

    check_grad("x (original)", x)
    check_grad("scores (STE)", scores)
    check_grad("k_float (STE)", k_float)
    check_grad("pruner.shiva_Ts (T-score/Rank)", pruner.shiva_Ts)
    check_grad("pruner.shiva_Tk (T-k/Select)", pruner.shiva_Tk)
