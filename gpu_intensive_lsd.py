"""
GPU-Intensive Live Stream Diffusion Implementation
Demonstrating the real computational challenges of video diffusion

This implementation shows:
1. Realistic GPU memory usage and computation
2. Proper diffusion sampling with many timesteps
3. Heavy attention mechanisms
4. Progressive complexity scaling
5. GPU utilization monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
from collections import deque
import psutil
import os

# Try to import GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False
    print("pynvml not available - install with: pip install pynvml")


class GPUMonitor:
    """Monitor GPU utilization and memory usage"""
    
    def __init__(self):
        self.nvml_available = NVML_AVAILABLE
        if self.nvml_available:
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
    
    def get_gpu_stats(self):
        """Get GPU utilization and memory stats"""
        if not self.nvml_available:
            return {"gpu_util": 0, "memory_used": 0, "memory_total": 0}
        
        try:
            handle = self.handles[0]  # Use first GPU
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                "gpu_util": util.gpu,
                "memory_used": mem_info.used // 1024**2,  # MB
                "memory_total": mem_info.total // 1024**2,  # MB
                "memory_percent": (mem_info.used / mem_info.total) * 100
            }
        except:
            return {"gpu_util": 0, "memory_used": 0, "memory_total": 0, "memory_percent": 0}


class MultiHeadAttention(nn.Module):
    """Heavy multi-head attention that actually uses GPU resources"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Adjust num_heads to ensure divisibility
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Large projection matrices to stress GPU
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation - this is GPU intensive
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class SpatialAttentionBlock(nn.Module):
    """Spatial attention for image patches - very GPU intensive"""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        # Adjust num_heads to ensure divisibility
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        # Adjust groups for GroupNorm
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        groups = max(1, groups)
        self.norm = nn.GroupNorm(groups, channels)
        self.attention = MultiHeadAttention(channels, num_heads)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert to sequence for attention
        x_norm = self.norm(x)
        x_seq = x_norm.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Apply attention (very GPU intensive for large H*W)
        attended = self.attention(x_seq)
        
        # Convert back to spatial
        attended = attended.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + attended


class ResidualBlock(nn.Module):
    """Heavy residual block with attention"""
    
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        # Adjust groups for GroupNorm to ensure divisibility
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        groups = max(1, groups)
        
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.SiLU()
        
        # Add attention to make it more GPU intensive
        self.attention = SpatialAttentionBlock(channels, num_heads=8)
        
    def forward(self, x):
        residual = x
        
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        # Apply heavy attention
        x = self.attention(x)
        
        return x + residual


class HeavyUNet(nn.Module):
    """Production-scale U-Net that actually stresses the GPU"""
    
    def __init__(self, in_channels=6, out_channels=3, base_channels=128, num_res_blocks=3):
        super().__init__()
        
        # Much larger architecture similar to production models
        self.base_channels = base_channels
        
        # Time embedding (production-scale)
        time_embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Encoder (downsampling path)
        self.enc_conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Level 1: 64x64 -> 32x32
        self.enc_level1 = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_res_blocks)
        ])
        self.enc_down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        
        # Level 2: 32x32 -> 16x16  
        self.enc_level2 = nn.ModuleList([
            ResidualBlock(base_channels * 2) for _ in range(num_res_blocks)
        ])
        self.enc_down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        
        # Level 3: 16x16 -> 8x8
        self.enc_level3 = nn.ModuleList([
            ResidualBlock(base_channels * 4) for _ in range(num_res_blocks)
        ])
        self.enc_down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        
        # Bottleneck: 8x8 (very GPU intensive)
        self.bottleneck = nn.ModuleList([
            ResidualBlock(base_channels * 8) for _ in range(num_res_blocks * 2)  # Extra blocks
        ])
        
        # Decoder (upsampling path)
        self.dec_up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1)
        self.dec_level3 = nn.ModuleList([
            ResidualBlock(base_channels * 8) for _ in range(num_res_blocks)  # Skip connections
        ])
        
        self.dec_up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec_level2 = nn.ModuleList([
            ResidualBlock(base_channels * 4) for _ in range(num_res_blocks)
        ])
        
        self.dec_up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec_level1 = nn.ModuleList([
            ResidualBlock(base_channels * 2) for _ in range(num_res_blocks)
        ])
        
        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
    
    def get_time_embedding(self, timesteps):
        """Sinusoidal time embeddings"""
        device = timesteps.device
        half_dim = self.base_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)
    
    def forward(self, x, timestep):
        # Time embedding
        t_emb = self.get_time_embedding(timestep)
        t_emb = t_emb.view(t_emb.size(0), t_emb.size(1), 1, 1)
        
        # Encoder
        x = self.enc_conv_in(x)
        
        # Level 1
        skip1 = x
        for block in self.enc_level1:
            x = block(x)
        x = x + t_emb.expand(-1, -1, x.size(2), x.size(3))  # Add time embedding
        x = self.enc_down1(x)
        
        # Level 2
        skip2 = x
        for block in self.enc_level2:
            x = block(x)
        x = x + t_emb.expand(-1, -1, x.size(2), x.size(3))
        x = self.enc_down2(x)
        
        # Level 3
        skip3 = x
        for block in self.enc_level3:
            x = block(x)
        x = x + t_emb.expand(-1, -1, x.size(2), x.size(3))
        x = self.enc_down3(x)
        
        # Bottleneck (most GPU intensive part)
        for block in self.bottleneck:
            x = block(x)
        
        # Decoder
        # Level 3
        x = self.dec_up3(x)
        x = torch.cat([x, skip3], dim=1)  # Skip connection
        for block in self.dec_level3:
            x = block(x)
        
        # Level 2
        x = self.dec_up2(x)
        x = torch.cat([x, skip2], dim=1)
        for block in self.dec_level2:
            x = block(x)
        
        # Level 1
        x = self.dec_up1(x)
        x = torch.cat([x, skip1], dim=1)
        for block in self.dec_level1:
            x = block(x)
        
        # Output
        x = self.out_conv(x)
        
        return x


class ProductionDiffusionScheduler:
    """Production-grade diffusion scheduler with proper noise scheduling"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Create noise schedule (much more realistic than simplified version)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x0, noise, timesteps):
        """Add noise to clean image"""
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
    
    def denoise_step(self, x_t, noise_pred, timestep):
        """Single denoising step (DDIM sampling)"""
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        
        # Compute x0 prediction
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
        
        # Compute x_{t-1}
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
        sqrt_one_minus_alpha_cumprod_prev = torch.sqrt(1 - alpha_cumprod_prev)
        
        x_prev = sqrt_alpha_cumprod_prev * x0_pred + sqrt_one_minus_alpha_cumprod_prev * noise_pred
        
        return x_prev


class GPUIntensiveLSD(nn.Module):
    """GPU-intensive Live Stream Diffusion model that actually uses resources"""
    
    def __init__(self, channels=3, base_channels=128, height=64, width=64, 
                 context_length=8, num_timesteps=50):  # Reduced for real-time
        super().__init__()
        
        self.channels = channels
        self.height = height
        self.width = width
        self.context_length = context_length
        self.num_timesteps = num_timesteps
        
        # Heavy U-Net (this will actually stress the GPU)
        self.unet = HeavyUNet(
            in_channels=channels * 2,  # Input + previous frame
            out_channels=channels,
            base_channels=base_channels,
            num_res_blocks=3
        )
        
        # Production diffusion scheduler
        self.scheduler = ProductionDiffusionScheduler(
            num_timesteps=num_timesteps,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Context memory with heavy attention
        self.context_memory = nn.ModuleList([
            ResidualBlock(channels) for _ in range(3)  # Multiple heavy blocks
        ])
        
        # Temporal blending network
        self.temporal_blend = nn.Sequential(
            nn.Conv2d(channels * 2, base_channels, 3, padding=1),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, input_frame, previous_frame, num_inference_steps=20):
        """Full diffusion sampling process - GPU intensive"""
        
        B, C, H, W = input_frame.shape
        device = input_frame.device
        
        # Start with random noise
        x = torch.randn(B, C, H, W, device=device)
        
        # Condition input (concatenate input and previous frame)
        condition = torch.cat([input_frame, previous_frame], dim=1)
        
        # Sampling loop (this is where the real computation happens)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, 
                                 dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            # Expand timestep for batch
            t_batch = t.expand(B)
            
            # Predict noise with heavy U-Net
            noise_pred = self.unet(torch.cat([x, condition], dim=1), t_batch)
            
            # Denoising step
            x = self.scheduler.denoise_step(x, noise_pred, t.item())
            
            # Apply context memory processing (additional GPU work)
            for memory_block in self.context_memory:
                x = memory_block(x)
        
        # Temporal blending with previous frame
        blend_input = torch.cat([x, previous_frame], dim=1)
        blend_weight = self.temporal_blend(blend_input)
        
        # Final blending
        output = blend_weight * x + (1 - blend_weight) * previous_frame
        
        return output


def progressive_complexity_demo():
    """Demonstrate progressive complexity scaling and GPU usage"""
    
    print("=" * 80)
    print("GPU-INTENSIVE LIVE STREAM DIFFUSION DEMONSTRATION")
    print("Showing real computational challenges and GPU utilization")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Running on CPU - GPU recommended for realistic performance testing")
    
    monitor = GPUMonitor()
    
    # Progressive complexity levels
    complexity_levels = [
        {"name": "Lightweight", "base_channels": 64, "inference_steps": 10, "batch_size": 1},
        {"name": "Medium", "base_channels": 96, "inference_steps": 15, "batch_size": 2}, 
        {"name": "Heavy", "base_channels": 128, "inference_steps": 20, "batch_size": 2},
        {"name": "Production", "base_channels": 160, "inference_steps": 25, "batch_size": 4},
    ]
    
    height, width = 64, 64
    channels = 3
    
    for level in complexity_levels:
        print(f"\n{'='*60}")
        print(f"Testing {level['name']} Configuration")
        print(f"  Base channels: {level['base_channels']}")
        print(f"  Inference steps: {level['inference_steps']}")
        print(f"  Batch size: {level['batch_size']}")
        print(f"{'='*60}")
        
        # Create model
        model = GPUIntensiveLSD(
            channels=channels,
            base_channels=level['base_channels'],
            height=height,
            width=width,
            num_timesteps=50
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Create test data
        batch_size = level['batch_size']
        input_frame = torch.randn(batch_size, channels, height, width, device=device)
        previous_frame = torch.randn(batch_size, channels, height, width, device=device)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            _ = model(input_frame[:1], previous_frame[:1], num_inference_steps=5)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Benchmark
        print("Benchmarking...")
        num_runs = 3
        times = []
        
        for run in range(num_runs):
            # Get initial GPU stats
            initial_stats = monitor.get_gpu_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                output = model(input_frame, previous_frame, 
                             num_inference_steps=level['inference_steps'])
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            frame_time = end_time - start_time
            times.append(frame_time)
            
            # Get final GPU stats
            final_stats = monitor.get_gpu_stats()
            
            print(f"  Run {run+1}: {frame_time:.3f}s | "
                  f"GPU Util: {final_stats['gpu_util']}% | "
                  f"Memory: {final_stats['memory_used']}/{final_stats['memory_total']}MB "
                  f"({final_stats['memory_percent']:.1f}%)")
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"\nResults:")
        print(f"  Average time per frame: {avg_time:.3f}s")
        print(f"  FPS: {fps:.2f}")
        print(f"  Mirage targets:")
        print(f"    40ms target: {'✅' if avg_time <= 0.040 else '❌'} ({avg_time/0.040:.1f}x)")
        print(f"    16ms target: {'✅' if avg_time <= 0.016 else '❌'} ({avg_time/0.016:.1f}x)")
        
        # Memory cleanup
        del model, input_frame, previous_frame, output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        time.sleep(1)  # Let GPU recover
    
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print("This demonstrates the real computational challenge of production video diffusion.")
    print("Notice how performance scales with model complexity and batch size.")
    print("The 'Production' level shows why Mirage needed PTX-level optimizations!")
    print(f"{'='*80}")


def long_sequence_test():
    """Test error accumulation over long sequences with GPU monitoring"""
    
    print(f"\n{'='*60}")
    print("LONG SEQUENCE ERROR ACCUMULATION TEST")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor = GPUMonitor()
    
    # Create a medium-complexity model for sustained testing
    model = GPUIntensiveLSD(
        channels=3,
        base_channels=96,  # Medium complexity
        height=64,
        width=64,
        num_timesteps=50
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate a longer sequence
    num_frames = 50
    input_frames = []
    
    # Create animated test pattern
    for t in range(20):  # Base pattern
        frame = torch.zeros(1, 3, 64, 64, device=device)
        
        # Complex animated pattern
        x = torch.linspace(0, 4 * math.pi, 64, device=device)
        y = torch.linspace(0, 4 * math.pi, 64, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # Multi-frequency pattern
        pattern = (torch.sin(X + t * 0.3) * torch.cos(Y + t * 0.2) + 
                  torch.sin(2 * X - t * 0.4) * torch.cos(3 * Y + t * 0.1))
        
        frame[0, 0] = pattern * 0.3
        frame[0, 1] = torch.roll(pattern, shifts=t*2, dims=0) * 0.3
        frame[0, 2] = torch.roll(pattern, shifts=-t*2, dims=1) * 0.3
        
        input_frames.append(frame)
    
    print(f"Generating {num_frames} frames with error accumulation prevention...")
    print("This will stress the GPU and show realistic performance...")
    
    generated_frames = []
    frame_times = []
    gpu_utils = []
    memory_usage = []
    
    # Initialize with first input
    previous_frame = input_frames[0]
    
    model.eval()
    with torch.no_grad():
        for i in range(num_frames):
            # Cycle through input patterns
            input_idx = i % len(input_frames)
            current_input = input_frames[input_idx]
            
            # Monitor GPU before generation
            gpu_stats = monitor.get_gpu_stats()
            
            start_time = time.time()
            
            # Generate frame (this is GPU intensive)
            generated_frame = model(current_input, previous_frame, 
                                  num_inference_steps=15)  # Moderate steps for balance
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            frame_time = time.time() - start_time
            
            generated_frames.append(generated_frame)
            frame_times.append(frame_time)
            gpu_utils.append(gpu_stats['gpu_util'])
            memory_usage.append(gpu_stats['memory_used'])
            
            # Update previous frame for autoregressive generation
            previous_frame = generated_frame
            
            if i % 5 == 0:
                avg_time = np.mean(frame_times[-5:]) if len(frame_times) >= 5 else np.mean(frame_times)
                print(f"Frame {i:2d}: {frame_time:.2f}s | Avg: {avg_time:.2f}s | "
                      f"GPU: {gpu_stats['gpu_util']}% | Mem: {gpu_stats['memory_used']}MB")
    
    # Analysis
    print(f"\n{'='*40}")
    print("PERFORMANCE ANALYSIS:")
    print(f"{'='*40}")
    
    total_time = sum(frame_times)
    avg_time = np.mean(frame_times)
    min_time = np.min(frame_times)
    max_time = np.max(frame_times)
    
    print(f"Total generation time: {total_time:.1f}s")
    print(f"Average time per frame: {avg_time:.2f}s")
    print(f"Min/Max time: {min_time:.2f}s / {max_time:.2f}s")
    print(f"Effective FPS: {1/avg_time:.2f}")
    
    print(f"\nMirage Performance Targets:")
    print(f"  40ms target: {'✅ ACHIEVED' if avg_time <= 0.040 else '❌ TOO SLOW'} ({avg_time:.3f}s)")
    print(f"  16ms target: {'✅ ACHIEVED' if avg_time <= 0.016 else '❌ TOO SLOW'} ({avg_time:.3f}s)")
    
    if monitor.nvml_available:
        print(f"\nGPU Utilization:")
        print(f"  Average GPU usage: {np.mean(gpu_utils):.1f}%")
        print(f"  Peak memory usage: {np.max(memory_usage)}MB")
    
    # Check for error accumulation
    print(f"\nError Accumulation Analysis:")
    
    # Compare early vs late frames
    early_frames = torch.stack(generated_frames[:10], dim=1)
    late_frames = torch.stack(generated_frames[-10:], dim=1)
    
    early_var = torch.var(early_frames).item()
    late_var = torch.var(late_frames).item()
    variance_ratio = late_var / early_var if early_var > 0 else 1.0
    
    print(f"  Early frames variance: {early_var:.6f}")
    print(f"  Late frames variance:  {late_var:.6f}")
    print(f"  Variance preservation: {variance_ratio:.3f}")
    
    if variance_ratio > 0.7:
        print("  ✅ Good error accumulation prevention")
    else:
        print("  ⚠️ Possible error accumulation detected")
    
    return generated_frames, frame_times


if __name__ == '__main__':
    print("Starting GPU-intensive LSD demonstration...")
    print("This shows the REAL computational challenges of video diffusion!")
    
    # Run progressive complexity demo
    progressive_complexity_demo()
    
    # Run long sequence test
    long_sequence_test()
    
    print("\n🎯 Key Takeaways:")
    print("1. Real video diffusion models are computationally intensive")
    print("2. GPU utilization is crucial for practical performance") 
    print("3. Model complexity directly impacts generation time")
    print("4. Mirage's 40ms/16ms targets are genuinely challenging")
    print("5. PTX-level optimizations become necessary at production scale")