"""
Tiny Visualization Model - Optimized for Real-Time Demo
Designed to show Tensor Core optimization wins visually

Features:
- Tiny U-Net with 1 attention block (fast but exercises all optimization paths)
- Head_dim=64 for optimal Flash Attention performance
- Tensor Core friendly dimensions (all multiples of 8)
- Switchable SDPA backends for visual FPS comparison
- Temporal conditioning for autoregressive stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import math


def sdpa_ctx(kind):
    """SDPA backend context manager"""
    if kind == 'flash':
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    elif kind == 'math':
        return sdpa_kernel(SDPBackend.MATH)
    else:
        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)


class TinyAttentionBlock(nn.Module):
    """Tiny attention block optimized for Tensor Cores"""
    
    def __init__(self, channels=256, num_heads=4, head_dim=64):
        super().__init__()
        
        # Ensure Tensor Core friendly dimensions
        self.channels = channels
        self.num_heads = num_heads  
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
        
        assert self.embed_dim % 8 == 0, f"Embed dim {self.embed_dim} not divisible by 8"
        assert head_dim % 8 == 0, f"Head dim {head_dim} not divisible by 8"
        
        # QKV projections
        self.qkv = nn.Linear(channels, self.embed_dim * 3, bias=False)
        self.proj = nn.Linear(self.embed_dim, channels)
        self.norm = nn.GroupNorm(8, channels)
        
        self.scale = head_dim ** -0.5
        
    def forward(self, x, sdpa_backend='flash'):
        """Forward with switchable SDPA backend"""
        B, C, H, W = x.shape
        
        # Normalize and flatten
        x_norm = self.norm(x)
        x_seq = x_norm.view(B, C, H * W).transpose(1, 2)  # B, HW, C
        
        # QKV projection
        qkv = self.qkv(x_seq).chunk(3, dim=-1)
        q, k, v = [tensor.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2) 
                   for tensor in qkv]
        
        # Ensure contiguous for SDPA
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Apply attention with specified backend
        with sdpa_ctx(sdpa_backend):
            attn_out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
        
        # Project and reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, H * W, self.embed_dim)
        attn_out = self.proj(attn_out)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + attn_out


class TinyVizUNet(nn.Module):
    """Tiny U-Net for visual demo - exercises conv + attention Tensor Core paths"""
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Ensure all dimensions are Tensor Core friendly (multiples of 8)
        base_channels = ((base_channels + 7) // 8) * 8
        in_channels = ((in_channels + 7) // 8) * 8
        
        print(f"🏗️ TinyVizUNet: {base_channels} base channels (Tensor Core optimized)")
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Input projection (handle RGB → padded channels)
        self.input_conv = nn.Conv2d(3, in_channels, 3, padding=1) if in_channels > 3 else nn.Identity()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels), 
            nn.SiLU(),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        
        # Attention in the middle (exercises attention Tensor Cores)
        self.attention = TinyAttentionBlock(
            channels=base_channels * 2,
            num_heads=4,
            head_dim=64  # Optimal for Flash Attention
        )
        
        # Decoder  
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),  # Skip connection
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x, prev=None, sdpa_backend='flash'):
        """Forward pass with temporal conditioning"""
        
        # Handle input channel padding if needed
        if x.size(1) == 3 and self.in_channels > 3:
            x = self.input_conv(x)
        
        # If temporal conditioning, concatenate with downsampled previous frame
        if prev is not None:
            # Downsample previous frame and concatenate
            prev_down = F.interpolate(prev, size=x.shape[2:], mode='bilinear', align_corners=False)
            if prev_down.size(1) != x.size(1):
                # Handle channel mismatch
                if prev_down.size(1) < x.size(1):
                    pad_channels = x.size(1) - prev_down.size(1)
                    prev_down = F.pad(prev_down, (0, 0, 0, 0, 0, pad_channels))
                else:
                    prev_down = prev_down[:, :x.size(1)]
            
            # Simple temporal blending instead of concatenation
            x = 0.7 * x + 0.3 * prev_down
        
        # Encoder
        x1 = self.enc1(x)           # 64 channels, full resolution
        x2 = self.enc2(x1)          # 128 channels, half resolution
        
        # Apply attention (exercises attention Tensor Cores)
        x2 = self.attention(x2, sdpa_backend=sdpa_backend)
        
        # Decoder with skip connection
        d2 = self.dec2(x2)          # Back to full resolution, 64 channels
        d1 = torch.cat([d2, x1], dim=1)  # Skip connection: 128 channels
        output = self.dec1(d1)      # Final output: 3 channels
        
        return output


class TinyVizModel:
    """Complete tiny model wrapper with optimization controls"""
    
    def __init__(self, base_channels=64, dtype='fp16', channels_last=True, sdpa_backend='flash'):
        self.dtype_name = dtype
        self.channels_last = channels_last
        self.sdpa_backend = sdpa_backend
        self.device = torch.device('cuda')
        
        # Set dtype
        self.dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32
        }[dtype]
        
        # Create model
        self.model = TinyVizUNet(
            in_channels=8,  # Padded from 3 for Tensor Core alignment
            out_channels=3,
            base_channels=base_channels
        ).to(self.device)
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Performance tracking
        self.frame_times = []
        self.memory_usage = []
        
        print(f"✅ TinyVizModel initialized:")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Dtype: {dtype}")
        print(f"   Channels last: {channels_last}")
        print(f"   SDPA backend: {sdpa_backend}")
    
    def _apply_optimizations(self):
        """Apply all Tensor Core optimizations"""
        
        # Enable fast math
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Convert model precision and memory format
        if self.dtype_name != 'fp32':
            self.model = self.model.to(dtype=self.dtype)
            
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        self.model.eval()
        
        # Verify optimization
        sample_param = next(self.model.parameters())
        print(f"   Model dtype: {sample_param.dtype}")
        
        if self.channels_last and len(sample_param.shape) == 4:
            is_channels_last = sample_param.is_contiguous(memory_format=torch.channels_last)
            print(f"   Channels last: {is_channels_last}")
    
    def process_frame(self, input_frame, prev_frame=None):
        """Process single frame with optimization tracking"""
        
        # Convert input to optimal format
        if self.dtype_name != 'fp32':
            input_frame = input_frame.to(dtype=self.dtype)
            if prev_frame is not None:
                prev_frame = prev_frame.to(dtype=self.dtype)
                
        if self.channels_last:
            input_frame = input_frame.contiguous(memory_format=torch.channels_last)
            if prev_frame is not None:
                prev_frame = prev_frame.contiguous(memory_format=torch.channels_last)
        
        # Track memory before processing
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() / 1024**2
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        # Process frame
        start_time.record()
        
        with torch.inference_mode():
            output = self.model(input_frame, prev=prev_frame, sdpa_backend=self.sdpa_backend)
        
        end_time.record()
        torch.cuda.synchronize()
        
        # Track performance
        frame_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        self.frame_times.append(frame_time)
        self.memory_usage.append(peak_memory)
        
        # Keep rolling average
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            self.memory_usage.pop(0)
        
        return output, {
            'frame_time': frame_time,
            'fps': 1.0 / frame_time if frame_time > 0 else 0,
            'memory_mb': peak_memory,
            'avg_fps': 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        }
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.frame_times:
            return {'avg_fps': 0, 'avg_memory': 0, 'frame_count': 0}
        
        return {
            'avg_fps': 1.0 / (sum(self.frame_times) / len(self.frame_times)),
            'avg_memory': sum(self.memory_usage) / len(self.memory_usage),
            'frame_count': len(self.frame_times),
            'latest_fps': 1.0 / self.frame_times[-1] if self.frame_times else 0
        }


if __name__ == '__main__':
    # Quick test of the tiny model
    print("🔬 Testing TinyVizModel...")
    
    # Test different configurations
    configs = [
        {'dtype': 'fp32', 'channels_last': False, 'sdpa_backend': 'math'},
        {'dtype': 'fp16', 'channels_last': True, 'sdpa_backend': 'flash'},
    ]
    
    for config in configs:
        print(f"\nTesting: {config}")
        
        model = TinyVizModel(base_channels=64, **config)
        
        # Create test input
        input_frame = torch.randn(1, 3, 288, 512, device='cuda')  # 512x288 as specified
        
        # Test processing
        for i in range(10):
            output, perf = model.process_frame(input_frame)
            if i == 0:
                print(f"   First frame: {perf['fps']:.1f} FPS, {perf['memory_mb']:.0f}MB")
        
        stats = model.get_performance_stats()
        print(f"   Average: {stats['avg_fps']:.1f} FPS")
        
        del model
        torch.cuda.empty_cache()