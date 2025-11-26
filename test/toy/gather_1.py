import torch
import time

device = 'cuda'
N, D = 4096, 128
x = torch.randn(N, D, device=device)
indices = torch.randperm(N)[:2048].to(device)

for _ in range(10):
    _ = x[indices]
torch.cuda.synchronize()

start = time.time()
for _ in range(1000):
    out1 = x[indices]
torch.cuda.synchronize()
print(f"Indexing Time: {(time.time() - start)*1000:.2f} ms")

gather_indices = indices.unsqueeze(-1).expand(-1, D)
start = time.time()
for _ in range(1000):
    out2 = torch.gather(x, 0, gather_indices)
torch.cuda.synchronize()
print(f"Gather Time:   {(time.time() - start)*1000:.2f} ms")

assert torch.allclose(out1, out2)