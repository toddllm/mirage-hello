"""
Working GPU-Intensive Video Diffusion Demo
This implementation actually stresses the GPU and demonstrates real performance challenges

Key features:
1. Large models that use significant GPU memory (2-8GB)
2. Heavy computational workloads with measurable GPU utilization
3. Progressive scaling to show performance degradation
4. Real-time monitoring of GPU usage during generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
import os
import subprocess
import threading
from collections import deque

# Try to import GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False


def get_gpu_stats():
    """Get current GPU utilization and memory"""
    if not NVML_AVAILABLE:
        return {"gpu_util": 0, "memory_used": 0, "memory_total": 0}
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
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


class HeavyConvBlock(nn.Module):
    """Computationally intensive convolution block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        
        # Multiple large convolutions to stress GPU
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        ])
        
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in range(4)
        ])
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class GPUIntensiveAttention(nn.Module):
    """Attention mechanism designed to stress GPU memory and compute"""
    
    def __init__(self, channels, num_heads=16):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Large projection matrices
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels) 
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        # Additional processing layers
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for attention
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # B, HW, C
        
        # Self-attention (very GPU intensive for large HW)
        residual = x_flat
        x_flat = self.norm1(x_flat)
        
        # QKV projections
        q = self.q_proj(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation (quadratic in sequence length)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, H * W, C)
        out = self.out_proj(out)
        
        # Residual connection
        x_flat = residual + out
        
        # Feed-forward network
        residual = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = residual + self.ffn(x_flat)
        
        # Reshape back to spatial
        return x_flat.transpose(1, 2).view(B, C, H, W)


class ProductionScaleUNet(nn.Module):
    """Production-scale U-Net that actually uses significant GPU resources"""
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=256):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Encoder - progressively more channels (more GPU memory)
        self.enc1 = HeavyConvBlock(in_channels, base_channels)
        self.enc2 = HeavyConvBlock(base_channels, base_channels * 2)  
        self.enc3 = HeavyConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = HeavyConvBlock(base_channels * 4, base_channels * 8)
        
        # Attention blocks at different scales (very GPU intensive)
        self.attn1 = GPUIntensiveAttention(base_channels * 2)
        self.attn2 = GPUIntensiveAttention(base_channels * 4) 
        self.attn3 = GPUIntensiveAttention(base_channels * 8)
        
        # Bottleneck with maximum channels
        self.bottleneck = nn.Sequential(
            HeavyConvBlock(base_channels * 8, base_channels * 16),
            GPUIntensiveAttention(base_channels * 16),
            HeavyConvBlock(base_channels * 16, base_channels * 8),
        )
        
        # Decoder with skip connections
        self.dec4 = HeavyConvBlock(base_channels * 16, base_channels * 4)  # 8 + 8 from skip
        self.dec3 = HeavyConvBlock(base_channels * 8, base_channels * 2)   # 4 + 4 from skip  
        self.dec2 = HeavyConvBlock(base_channels * 4, base_channels)       # 2 + 2 from skip
        self.dec1 = HeavyConvBlock(base_channels * 2, base_channels)       # 1 + 1 from skip
        
        # Output projection
        self.output = nn.Conv2d(base_channels, out_channels, 1)
        
        # Downsampling and upsampling
        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Encoder with skip connections
        x1 = self.enc1(x)          # 64x64
        x2 = self.enc2(self.down(x1))   # 32x32
        x2 = self.attn1(x2)             # Heavy attention
        
        x3 = self.enc3(self.down(x2))   # 16x16  
        x3 = self.attn2(x3)             # Heavy attention
        
        x4 = self.enc4(self.down(x3))   # 8x8
        x4 = self.attn3(x4)             # Heavy attention
        
        # Bottleneck (most GPU intensive)
        bottleneck = self.bottleneck(self.down(x4))  # 4x4
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up(bottleneck), x4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), x3], dim=1)) 
        d2 = self.dec2(torch.cat([self.up(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), x1], dim=1))
        
        return self.output(d1)


class GPUStressTestModel(nn.Module):
    """Model specifically designed to stress GPU resources"""
    
    def __init__(self, channels=3, base_channels=256, num_timesteps=100):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Multiple U-Nets for extreme GPU usage
        self.unets = nn.ModuleList([
            ProductionScaleUNet(channels + 1, channels, base_channels),  # +1 for timestep
            ProductionScaleUNet(channels, channels, base_channels // 2),
        ])
        
        # Heavy temporal processing
        self.temporal_processor = nn.Sequential(
            HeavyConvBlock(channels * 2, base_channels),
            GPUIntensiveAttention(base_channels),
            HeavyConvBlock(base_channels, channels),
        )
        
    def forward(self, x, timestep, previous_frame=None):
        B, C, H, W = x.shape
        
        if previous_frame is None:
            previous_frame = torch.zeros_like(x)
        
        # Add timestep information
        t_embed = torch.full((B, 1, H, W), timestep / self.num_timesteps, 
                           device=x.device, dtype=x.dtype)
        x_with_t = torch.cat([x, t_embed], dim=1)
        
        # Multiple passes through heavy networks
        for i, unet in enumerate(self.unets):
            if i == 0:
                out = unet(x_with_t)
            else:
                out = unet(out)
                
        # Heavy temporal processing
        temporal_input = torch.cat([out, previous_frame], dim=1)
        out = self.temporal_processor(temporal_input)
        
        return out


def gpu_stress_test():
    """Run progressively more intensive GPU workloads"""
    
    print("=" * 80)
    print("GPU STRESS TEST - REAL VIDEO DIFFUSION COMPUTATIONAL LOAD")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        print("❌ ERROR: This demo requires GPU to show real computational challenges")
        return
    
    # Get initial GPU state
    initial_stats = get_gpu_stats()
    print(f"Initial GPU state: {initial_stats['memory_used']}/{initial_stats['memory_total']}MB "
          f"({initial_stats['memory_percent']:.1f}% used)")
    
    print(f"\nTesting progressive model sizes to stress GPU...")
    
    # Progressive model sizes
    model_configs = [
        {"name": "Medium Load", "base_channels": 128, "batch_size": 1, "resolution": 64},
        {"name": "Heavy Load", "base_channels": 192, "batch_size": 2, "resolution": 64},
        {"name": "Extreme Load", "base_channels": 256, "batch_size": 2, "resolution": 128},
        {"name": "Maximum Load", "base_channels": 320, "batch_size": 4, "resolution": 128},
    ]
    
    for i, config in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {config['name']}")
        print(f"Base channels: {config['base_channels']}, Batch: {config['batch_size']}, "
              f"Resolution: {config['resolution']}x{config['resolution']}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = GPUStressTestModel(
                channels=3,
                base_channels=config['base_channels'],
                num_timesteps=50
            ).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")
            
            # Create large input tensors
            batch_size = config['batch_size']
            resolution = config['resolution']
            
            input_tensor = torch.randn(batch_size, 3, resolution, resolution, device=device)
            prev_tensor = torch.randn(batch_size, 3, resolution, resolution, device=device)
            
            # Check GPU memory after model loading
            after_load_stats = get_gpu_stats()
            print(f"GPU after model load: {after_load_stats['memory_used']}/{after_load_stats['memory_total']}MB "
                  f"({after_load_stats['memory_percent']:.1f}% used)")
            
            # Warmup
            print("Warming up...")
            with torch.no_grad():
                _ = model(input_tensor[:1], timestep=25, previous_frame=prev_tensor[:1])
            torch.cuda.synchronize()
            
            # Benchmark with GPU monitoring
            print("Running GPU-intensive generation...")
            num_iterations = 10
            times = []
            gpu_utils = []
            memory_usage = []
            
            for iteration in range(num_iterations):
                # Get GPU stats before
                pre_stats = get_gpu_stats()
                
                start_time = time.time()
                
                with torch.no_grad():
                    # Multiple forward passes to stress GPU
                    for timestep in range(0, 50, 10):  # 5 timesteps
                        output = model(input_tensor, timestep=timestep, previous_frame=prev_tensor)
                        prev_tensor = output  # Autoregressive
                
                torch.cuda.synchronize()
                frame_time = time.time() - start_time
                
                # Get GPU stats after
                post_stats = get_gpu_stats()
                
                times.append(frame_time)
                gpu_utils.append(post_stats['gpu_util'])
                memory_usage.append(post_stats['memory_used'])
                
                if iteration % 2 == 0:
                    print(f"  Iteration {iteration+1}: {frame_time:.2f}s | "
                          f"GPU: {post_stats['gpu_util']}% | "
                          f"Memory: {post_stats['memory_used']}MB")
            
            # Results
            avg_time = np.mean(times)
            avg_gpu_util = np.mean(gpu_utils)
            max_memory = np.max(memory_usage)
            
            print(f"\n📊 Results:")
            print(f"  Average time per sequence: {avg_time:.2f}s")
            print(f"  Average GPU utilization: {avg_gpu_util:.1f}%")
            print(f"  Peak memory usage: {max_memory}MB")
            print(f"  FPS (single frame): {5/avg_time:.1f}")  # 5 timesteps per sequence
            
            # Performance assessment
            if avg_time <= 0.040:  # Mirage 40ms target
                print(f"  🎯 Mirage 40ms target: ✅ ACHIEVED ({avg_time:.3f}s)")
            else:
                print(f"  🎯 Mirage 40ms target: ❌ MISSED ({avg_time:.3f}s, {avg_time/0.040:.1f}x slower)")
            
            if avg_gpu_util > 50:
                print(f"  💪 GPU Utilization: ✅ GOOD ({avg_gpu_util:.1f}%)")
            else:
                print(f"  💪 GPU Utilization: ⚠️  LOW ({avg_gpu_util:.1f}%)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ GPU OUT OF MEMORY - Model too large for available GPU memory")
                print(f"  This demonstrates the real memory constraints of production video diffusion!")
                # Free memory and continue
                torch.cuda.empty_cache()
                break
            else:
                print(f"  ❌ Error: {e}")
                break
        
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            break
        
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            if 'input_tensor' in locals():
                del input_tensor
            if 'prev_tensor' in locals():
                del prev_tensor
            if 'output' in locals():
                del output
            torch.cuda.empty_cache()
            time.sleep(2)  # Let GPU cool down
    
    print(f"\n{'='*80}")
    print("🎯 ANALYSIS:")
    print("This demonstration shows the REAL computational challenge of video diffusion:")
    print("1. Models require 2-8GB+ GPU memory for production quality")
    print("2. GPU utilization should be 50-90% for optimal performance") 
    print("3. Memory bandwidth becomes the limiting factor")
    print("4. Mirage's 40ms target is genuinely difficult to achieve")
    print("5. This is why they needed PTX-level assembly optimizations!")
    print(f"{'='*80}")


def continuous_gpu_monitor():
    """Monitor GPU usage continuously while generation runs"""
    
    print("\n🖥️  CONTINUOUS GPU MONITORING TEST")
    print("Running sustained workload to show GPU utilization...")
    
    device = torch.device('cuda')
    
    # Create a moderately sized model for sustained testing
    model = GPUStressTestModel(channels=3, base_channels=192).to(device)
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Monitor GPU while running continuous generation
    batch_size = 2
    resolution = 64
    
    input_tensor = torch.randn(batch_size, 3, resolution, resolution, device=device)
    prev_frame = torch.randn(batch_size, 3, resolution, resolution, device=device)
    
    print("\nStarting continuous generation (watch GPU utilization)...")
    print("Run 'watch -n 1 nvidia-smi' in another terminal to monitor")
    
    model.eval()
    start_time = time.time()
    
    try:
        with torch.no_grad():
            for frame_idx in range(100):  # 100 frames
                
                # Generate with multiple timesteps (GPU intensive)
                for timestep in [40, 30, 20, 10, 0]:
                    output = model(input_tensor, timestep=timestep, previous_frame=prev_frame)
                    prev_frame = output
                
                if frame_idx % 10 == 0:
                    gpu_stats = get_gpu_stats()
                    elapsed = time.time() - start_time
                    fps = (frame_idx + 1) / elapsed
                    
                    print(f"Frame {frame_idx:3d}: {fps:.1f} FPS | "
                          f"GPU: {gpu_stats['gpu_util']:2d}% | "
                          f"Memory: {gpu_stats['memory_used']}MB "
                          f"({gpu_stats['memory_percent']:.1f}%)")
                    
                    # Check if we're actually using the GPU effectively
                    if gpu_stats['gpu_util'] < 30:
                        print("⚠️  Low GPU utilization - model might be too small or CPU bottlenecked")
                    elif gpu_stats['gpu_util'] > 80:
                        print("🔥 High GPU utilization - this is what production video diffusion looks like!")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    total_time = time.time() - start_time
    print(f"\nGenerated 100 frames in {total_time:.1f}s ({100/total_time:.1f} FPS average)")


if __name__ == '__main__':
    print("🚀 Starting REAL GPU-intensive video diffusion demonstration...")
    print("This will actually stress your GPU and show measurable utilization!")
    
    # Run the GPU stress test
    gpu_stress_test()
    
    # Ask user if they want continuous monitoring
    print("\n" + "="*60)
    print("Would you like to run continuous GPU monitoring?")
    print("This will show sustained GPU usage over time.")
    print("Run this and check 'nvidia-smi' in another terminal!")
    print("="*60)
    
    # Run continuous monitoring
    continuous_gpu_monitor()