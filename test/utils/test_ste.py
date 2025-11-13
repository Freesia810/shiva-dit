import torch
from shiva_dit.utils.ste import ShivaTopkFunc

if __name__ == "__main__":
    B, N, D = 1, 5, 2
    x = torch.tensor([[
        [1.0, 1.0],
        [-1.0, -1.0],
        [0.5, -0.5],
        [1.0, -1.0],
        [-1.0, 1.0],
    ]], requires_grad=False)

    scores = torch.tensor([[0.2, 0.1, -0.3, 0.0, -0.1]], requires_grad=True)
    k = torch.tensor(2.0, requires_grad=False)
    T = torch.tensor(1.0, requires_grad=False)

    # === Forward ===
    x_sel, x_rej, topk_idx, rest_idx = ShivaTopkFunc.apply(x, scores, k, T)

    target = torch.tensor([[1.0, 1.0]])
    loss = torch.nn.functional.mse_loss(x_sel.mean(dim=1), target)

    # === Backward ===
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print("Topk idx:", topk_idx)
    print("Scores (before):", scores.data)
    print("Grad_scores:", scores.grad)

    lr = 0.5
    new_scores = scores - lr * scores.grad
    print("Scores (after):", new_scores.data)