import torch

def reschedule_lut(path, target_avg):
    data = torch.load(path)
    ratio = data['ratio']
    current_avg = ratio.mean().item()
    
    new_ratio = torch.clamp(ratio * (target_avg / current_avg), max=1.0)
    data['ratio'] = new_ratio
    
    new_path = path.replace(".pt", f"_{int(target_avg*100)}.pt")
    torch.save(data, new_path)
    print(f"Saved {target_avg*100}% LUT to {new_path}. New Avg: {new_ratio.mean():.4f}")

reschedule_lut("lut-80.pt", 0.5)
reschedule_lut("lut-60.pt", 0.5)