import torch
import torch.nn as nn
import torch.nn.functional as F
from shiva_dit.utils.ste import ShivaTopkFunc

# ==========================================
# 1. DiffCR-like Implementation (Mock)
# ==========================================
class DiffCRLikeTopK(nn.Module):
    """
    Re-implementation of the linear slot interpolation logic from baselines/diffcr.py
    for testing purposes.
    """
    def __init__(self, N, ratio_min=0.0, ratio_max=1.0, ratio_step=0.1):
        super().__init__()
        self.N = N
        steps = int((ratio_max - ratio_min) / ratio_step) + 1
        self.register_buffer("kept_ratio_candidate", torch.linspace(ratio_min, ratio_max, steps))
        
    def find_neighbour_bins(self, kept_ratio):
        differences = torch.abs(self.kept_ratio_candidate - kept_ratio)
        _, indices = torch.topk(differences, 2, largest=False)
        neighbour_bins = self.kept_ratio_candidate[indices]
        return neighbour_bins, indices

    def _process_bin(self, x, scores, ratio):
        B, N, D = x.shape
        k = int(torch.ceil(ratio * self.N).item())
        k = max(1, min(k, self.N))
        
        # 1. Calculate Probabilities
        probs = torch.sigmoid(scores) # (B, N)
        
        # 2. Top-K Selection
        # We need values (probs) to multiply later for gradients, and indices for gathering
        topk_probs_unsorted, topk_indices = torch.topk(probs, k=k, dim=1, sorted=False)
        
        # 3. Sort indices to preserve relative order (Spatial consistency)
        sorted_indices, sort_idx = torch.sort(topk_indices, dim=1)
        
        # 4. Reorder the topk probabilities to match the sorted indices
        # This corresponds to 'self.token_scores.gather(dim=1, index=self.index)' in diffcr.py
        topk_probs_sorted = torch.gather(topk_probs_unsorted, 1, sort_idx)
        
        # 5. Gather Features (x)
        gather_index = sorted_indices.unsqueeze(-1).expand(-1, -1, D)
        topk_x = torch.gather(x, dim=1, index=gather_index)
        
        # 7. Scatter back to full size (to compare with original shape)
        output_full = torch.zeros_like(x)
        output_full.scatter_(1, gather_index, topk_x)
        
        return output_full

    def forward(self, x, scores, k_float):
        ratio = k_float / self.N
        ratio = torch.clamp(ratio, 0.0, 1.0)
        
        neighbour_bins, _ = self.find_neighbour_bins(ratio)
        lower_bin = neighbour_bins.min()
        upper_bin = neighbour_bins.max()
        
        if torch.abs(upper_bin - lower_bin) < 1e-6:
            return self._process_bin(x, scores, lower_bin)
            
        # Linear interpolation of OUTPUTS (features)
        lower_weight = (upper_bin - ratio) / (upper_bin - lower_bin)
        upper_weight = 1.0 - lower_weight
        
        out_lower = self._process_bin(x, scores, lower_bin)
        out_upper = self._process_bin(x, scores, upper_bin)
        
        return lower_weight * out_lower + upper_weight * out_upper


# ==========================================
# 2. Shiva Implementation (Wrapper)
# ==========================================
class ShivaTopKWrapper(nn.Module):
    def __init__(self, initial_Ts=1.0, initial_Tk=10.0):
        super().__init__()
        self.shiva_Ts = nn.Parameter(torch.tensor(initial_Ts))
        self.shiva_Tk = nn.Parameter(torch.tensor(initial_Tk))

    def forward(self, x, scores, k_float):
        # ShivaTopkFunc returns: x_selected, x_rejected, topk_idx, rest_idx
        x_sel, x_rej, topk_idx, rest_idx = ShivaTopkFunc.apply(
            x, scores, k_float, self.shiva_Ts, self.shiva_Tk
        )
        
        # Scatter selected tokens back to full size for comparison
        B, N, D = x.shape
        output_full = torch.zeros_like(x)
        output_full.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, D), x_sel)
        
        return output_full


def compare_gradient_k():
    torch.manual_seed(42)
    
    B, N, D = 1, 5, 1
    k_init_val = 3.0
    
    print(f"--- Test Configuration ---")
    print(f"Batch(B)={B}, Seq(N)={N}, Dim(D)={D}")
    print(f"Target k = {k_init_val}")
    
    # ---------------- Create Inputs ----------------
    x_data = torch.randn(B, N, D) * 2.0
    scores_data = torch.randn(B, N)
    print(f'\nx_data (Values):\n{x_data.squeeze(-1).numpy()}')
    print(f'scores_data (Logits):\n{scores_data.numpy()}')
    
    # DiffCR inputs
    x_diff = x_data.clone().detach().requires_grad_(True)
    scores_diff = scores_data.clone().detach().requires_grad_(True)
    k_diff = torch.tensor(k_init_val, requires_grad=True)
    
    # Shiva inputs
    x_shiva = x_data.clone().detach().requires_grad_(True)
    scores_shiva = scores_data.clone().detach().requires_grad_(True)
    k_shiva = torch.tensor(k_init_val, requires_grad=True)
    
    # ---------------- Initialize Models ----------------
    diffcr_model = DiffCRLikeTopK(N=N, ratio_step=0.1)
    shiva_model = ShivaTopKWrapper(initial_Ts=1.0, initial_Tk=10.0)
    
    # ---------------- Forward Pass ----------------
    out_diff = diffcr_model(x_diff, scores_diff, k_diff)
    out_shiva = shiva_model(x_shiva, scores_shiva, k_shiva)
    
    print(f'\nout_diff (DiffCR Output):\n{out_diff.squeeze(-1).detach().numpy()}')
    print(f'out_shiva (Shiva Output):\n{out_shiva.squeeze(-1).detach().numpy()}')
    
    # ---------------- Backward Pass ----------------
    # Loss: Maximizing sum. 
    # NOTE: Gradient > 0 means increasing k will increase the Sum (Loss).
    loss_diff = out_diff.sum()
    loss_shiva = out_shiva.sum()

    print(f'\nLoss DiffCR: {loss_diff.item():.4f}')
    print(f'Loss Shiva: {loss_shiva.item():.4f}')
    
    loss_diff.backward()
    loss_shiva.backward()
    
    # ---------------- k Gradient Analysis ----------------
    print("\n--- k Gradient Analysis ---")
    g_k_diff = k_diff.grad
    g_k_shiva = k_shiva.grad
    
    print(f"DiffCR k Grad: {g_k_diff.item() if g_k_diff is not None else 'None'}")
    print(f"Shiva  k Grad: {g_k_shiva.item() if g_k_shiva is not None else 'None'}")

    if g_k_diff is not None and g_k_shiva is not None:
        # Check Direction
        match = (g_k_diff.item() * g_k_shiva.item()) > 0
        print(f"Direction Match: {'YES' if match else 'NO'}")
        
        # Interpretation
        print("\n[Interpretation]")
        print(f"Gradient Sign: {'Positive (+)' if g_k_shiva.item() > 0 else 'Negative (-)'}")
        if g_k_shiva.item() > 0:
            print("   -> Increasing k will include more POSITIVE values (Increase Sum).")
        else:
            print("   -> Increasing k will include more NEGATIVE values (Decrease Sum).")


def check_gradient_score():
    B, N, D = 1, 4, 1
    x = torch.tensor([[[10.0], [5.0], [-5.0], [-10.0]]], device='cuda').float()
    initial_logits = torch.tensor([[-5.0, -3.0, 3.0, 5.0]], device='cuda').requires_grad_(True)

    k_float = torch.tensor(2.0, device='cuda')
    
    Ts = torch.tensor(1.0, device='cuda')
    Tk = torch.tensor(10.0, device='cuda')
    
    print(f"\n=== Metadata ===")
    print(f"Token Values (x): {x.squeeze().cpu().numpy().tolist()}")
    print(f"Initial Logits  : {initial_logits.squeeze().detach().cpu().numpy().tolist()}")

    class ShivaLayer(nn.Module):
        def forward(self, x, logits, k, Ts, Tk):
            return ShivaTopkFunc.apply(x, logits, k, Ts, Tk)
            
    layer = ShivaLayer()
    x_selected, x_rejected, idx_topk, idx_rest = layer(x, initial_logits, k_float, Ts, Tk)
    
    print(f"\n=== Forward ===")
    print(f"Selected Indices: {idx_topk.squeeze().cpu().numpy().tolist()}")
    print(f"Selected Values : {x_selected.squeeze().cpu().detach().numpy().tolist()}")
    
    loss = -x_selected.sum()
    
    loss.backward()
    grad = initial_logits.grad.squeeze()
    
    print(f"\n=== (Loss = -Sum(Selected)) ===")
    print(f"Logits Gradients: {grad.cpu().numpy()}")
    
    g0 = grad[0].item()
    status0 = "✅ PASS" if g0 < 0 else "❌ FAIL"
    print(f"Token 0 (Value=10, Rejected): Grad={g0:.4f} (Expect < 0) -> {status0}")

    g1 = grad[1].item()
    status1 = "✅ PASS" if g1 < 0 else "❌ FAIL"
    print(f"Token 1 (Value= 5, Rejected): Grad={g1:.4f} (Expect < 0) -> {status1}")

    g2 = grad[2].item()
    status2 = "✅ PASS" if g2 > 0 else "❌ FAIL"
    print(f"Token 2 (Value=-5, Selected): Grad={g2:.4f} (Expect > 0) -> {status2}")

    g3 = grad[3].item()
    status3 = "✅ PASS" if g3 > 0 else "❌ FAIL"
    print(f"Token 3 (Value=-10,Selected): Grad={g3:.4f} (Expect > 0) -> {status3}")

if __name__ == "__main__":
    compare_gradient_k()