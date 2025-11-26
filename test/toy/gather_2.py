import torch
import time

def benchmark_gather_vs_indexing(B, N, D, K, runs=1000):
    device = 'cuda'
    if not torch.cuda.is_available():
        return

    x = torch.randn(B, N, D, device=device, dtype=torch.float32)

    topk_idx = torch.randint(0, N, (B, K), device=device, dtype=torch.long)

    gather_idx = topk_idx.unsqueeze(-1).expand(B, K, D)
    
    batch_idx = torch.arange(B, device=device).unsqueeze(1)

    print(f"\n--- Parameters ---")
    print(f"Data: B={B}, N={N}, D={D}, K={K}. Runs: {runs}")
    print(f"Gather: {gather_idx.numel():,} (elements)")
    print(f"Indexing: {batch_idx.numel() + topk_idx.numel():,} (elements)")
    
    torch.cuda.synchronize()
    start_time_a = time.time()
    for _ in range(runs):
        out_a = torch.gather(x, 1, gather_idx)
    torch.cuda.synchronize()
    end_time_a = time.time()

    time_a = (end_time_a - start_time_a) * 1000 / runs
    
    torch.cuda.synchronize()
    start_time_b = time.time()
    for _ in range(runs):
        out_b = x[batch_idx, topk_idx]
    torch.cuda.synchronize()
    end_time_b = time.time()
    
    time_b = (end_time_b - start_time_b) * 1000 / runs

    assert torch.allclose(out_a, out_b), "Different results!"
    
    print(f"\n1. Gather (ms): {time_a:.4f}")
    print(f"2. Indexing (ms): {time_b:.4f}")
    
    if time_b < time_a:
        speedup = time_a / time_b
        print(f"\n>>> Indexing Speedup: {speedup:.2f} (Save: {time_a - time_b:.4f} ms)")
    else:
         print(f"\n!!! Indexing Not Benefit ({time_a/time_b:.2f}x)")


benchmark_gather_vs_indexing(B=8, N=4096, D=1024, K=2048)