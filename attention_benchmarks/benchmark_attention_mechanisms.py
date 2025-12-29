"""Compare standard vs hierarchical attention performance."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class StandardAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class HierarchicalAttention(nn.Module):
    def __init__(self, dim, n_heads=8, window=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Windowed attention
        if N > self.window:
            n_windows = (N + self.window - 1) // self.window
            outputs = []
            for i in range(n_windows):
                start = i * self.window
                end = min((i + 1) * self.window, N)
                q_w = q[:, :, start:end, :]
                k_w = k[:, :, start:end, :]
                v_w = v[:, :, start:end, :]
                attn = (q_w @ k_w.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn = F.softmax(attn, dim=-1)
                outputs.append((attn @ v_w))
            x = torch.cat(outputs, dim=2)
        else:
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

def benchmark(model, data, n_runs=50):
    """Measure execution time."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(data)
        
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(data)
            times.append((time.perf_counter() - start) * 1000)
    
    return np.mean(times), np.std(times)

def main():
    import numpy as np
    
    dim = 512
    n_heads = 8
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}\n")
    
    for seq_len in [128, 256, 512, 1024]:
        data = torch.randn(batch_size, seq_len, dim).to(device)
        
        std_attn = StandardAttention(dim, n_heads).to(device)
        hier_attn = HierarchicalAttention(dim, n_heads, window=64).to(device)
        
        std_time, std_std = benchmark(std_attn, data)
        hier_time, hier_std = benchmark(hier_attn, data)
        
        speedup = std_time / hier_time
        
        print(f"Sequence length: {seq_len}")
        print(f"  Standard:     {std_time:.2f} ± {std_std:.2f} ms")
        print(f"  Hierarchical: {hier_time:.2f} ± {hier_std:.2f} ms")
        print(f"  Speedup:      {speedup:.2f}x\n")

if __name__ == '__main__':
    main()
