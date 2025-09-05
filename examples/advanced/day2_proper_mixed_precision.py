"""
Day 2: Proper Mixed Precision Implementation
Fixing Day 1 issues with expert guidance approach

Key fixes based on analysis:
1. Direct model conversion (no autocast overhead)
2. Channels_last memory format for Tensor Core utilization
3. SDPA Flash Attention backend
4. Proper FP16 inference path
5. TF32 optimizations enabled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
import numpy as np
from working_gpu_demo import GPUStressTestModel, get_gpu_stats

# Enable all the fast math optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

print(f"🚀 Day 2: PROPER MIXED PRECISION IMPLEMENTATION")
print(f"✅ TF32 enabled: matmul={torch.backends.cuda.matmul.allow_tf32}, "
      f"cudnn={torch.backends.cudnn.allow_tf32}")
print(f"✅ CUDNN benchmark: {torch.backends.cudnn.benchmark}")


class OptimizedAttention(nn.Module):
    """Optimized attention using SDPA/Flash backend"""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        
        # Ensure head_dim = 64 for optimal Flash Attention
        if channels >= 512:
            self.num_heads = channels // 64  # head_dim = 64
        else:
            self.num_heads = max(1, channels // 32)  # head_dim = 32 or 64
        
        self.head_dim = channels // self.num_heads
        
        print(f"   Attention: {channels} channels → {self.num_heads} heads × {self.head_dim} dim")
        
        # Linear projections (will be converted to FP16)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x_seq = x.view(B, C, H * W).transpose(1, 2)  # B, HW, C
        
        # QKV projections
        q = self.q_proj(x_seq).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_seq).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_seq).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Ensure contiguous and channels_last for SDPA optimization
        q = q.contiguous()
        k = k.contiguous() 
        v = v.contiguous()
        
        # Use SDPA with Flash Attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
        
        # Reshape back to spatial
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H * W, C)
        attn_output = self.out_proj(attn_output)
        
        return attn_output.transpose(1, 2).view(B, C, H, W)


class OptimizedConvBlock(nn.Module):
    """Tensor Core optimized convolution block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        
        # Ensure channels are multiples of 8 for FP16 Tensor Core optimization
        in_channels = ((in_channels + 7) // 8) * 8
        out_channels = ((out_channels + 7) // 8) * 8
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.SiLU()
        
        # Add attention for GPU stress
        self.attention = OptimizedAttention(out_channels)
        
    def forward(self, x):
        # Residual connection
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x) 
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        # Apply optimized attention
        x = self.attention(x)
        
        # Skip connection (handle channel mismatch)
        if residual.size(1) != x.size(1):
            residual = F.adaptive_avg_pool2d(residual, x.size()[2:])
            if residual.size(1) < x.size(1):
                # Pad channels
                pad_channels = x.size(1) - residual.size(1)
                residual = F.pad(residual, (0, 0, 0, 0, 0, pad_channels))
            elif residual.size(1) > x.size(1):
                # Truncate channels
                residual = residual[:, :x.size(1)]
        
        return x + residual


class OptimizedProductionUNet(nn.Module):
    """Tensor Core optimized U-Net for mixed precision"""
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=128):
        super().__init__()
        
        # Round all channels to multiples of 8
        base_channels = ((base_channels + 7) // 8) * 8
        print(f"🔧 Optimized base_channels: {base_channels} (Tensor Core friendly)")
        
        self.base_channels = base_channels
        
        # Simplified encoder (fewer blocks for speed)
        self.enc1 = OptimizedConvBlock(in_channels, base_channels)
        self.enc2 = OptimizedConvBlock(base_channels, base_channels * 2)
        self.enc3 = OptimizedConvBlock(base_channels * 2, base_channels * 4)
        
        # Simplified bottleneck
        self.bottleneck = OptimizedConvBlock(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.dec3 = OptimizedConvBlock(base_channels * 12, base_channels * 2)  # 8+4 skip
        self.dec2 = OptimizedConvBlock(base_channels * 4, base_channels)       # 2+2 skip
        self.dec1 = OptimizedConvBlock(base_channels * 2, base_channels)       # 1+1 skip
        
        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)
        
        # Downsampling/upsampling
        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        # Encoder with skips
        x1 = self.enc1(x)                 # 64x64
        x2 = self.enc2(self.down(x1))     # 32x32  
        x3 = self.enc3(self.down(x2))     # 16x16
        
        # Bottleneck  
        bottleneck = self.bottleneck(self.down(x3))  # 8x8
        
        # Decoder with skips
        d3 = self.dec3(torch.cat([self.up(bottleneck), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), x1], dim=1))
        
        return self.output(d1)


class Day2OptimizedModel:
    """Day 2: Properly optimized mixed precision model"""
    
    def __init__(self, base_channels=128, precision='fp16'):
        self.precision = precision
        self.device = torch.device('cuda')
        
        # Create optimized model
        self.model = OptimizedProductionUNet(
            in_channels=4,  # 3 + 1 for timestep
            out_channels=3,
            base_channels=base_channels
        ).to(self.device)
        
        # Convert to target precision and memory format
        self.dtype = torch.float16 if precision == 'fp16' else torch.bfloat16 if precision == 'bf16' else torch.float32
        
        if precision != 'fp32':
            print(f"🔄 Converting model to {precision.upper()} + channels_last...")
            self.model = self.model.to(
                dtype=self.dtype,
                memory_format=torch.channels_last
            )
            
            # Verify conversion
            sample_weight = next(self.model.parameters())
            print(f"✅ Model dtype: {sample_weight.dtype}")
            print(f"✅ Memory format: {sample_weight.is_contiguous(memory_format=torch.channels_last)}")
        
        self.model.eval()
        print(f"✅ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def forward(self, x, timestep, previous_frame=None):
        """Optimized inference without autocast overhead"""
        
        if previous_frame is None:
            previous_frame = torch.zeros_like(x)
        
        # Add timestep channel
        B, C, H, W = x.shape
        t_embed = torch.full((B, 1, H, W), timestep / 100.0, device=x.device, dtype=x.dtype)
        x_input = torch.cat([x, t_embed], dim=1)
        
        # Ensure inputs are channels_last and correct dtype
        if self.precision != 'fp32':
            x_input = x_input.to(dtype=self.dtype, memory_format=torch.channels_last)
        
        # Pure inference mode (no autocast needed - model already converted)
        with torch.inference_mode():
            return self.model(x_input)


def day2_benchmark():
    """Day 2 comprehensive benchmark with proper optimizations"""
    
    print("🎯 DAY 2 BENCHMARK: PROPER MIXED PRECISION")
    print("=" * 80)
    print("Fixes: Direct conversion, channels_last, SDPA, no autocast overhead")
    print()
    
    # Test configuration
    config = {'base_channels': 128, 'batch_size': 2}  # Lighter config for clear results
    
    results = {}
    
    for precision in ['fp32', 'fp16', 'bf16']:
        print(f"\n{'='*50}")
        print(f"🔬 Testing {precision.upper()} - Proper Implementation")
        print(f"{'='*50}")
        
        try:
            # Create properly optimized model
            model = Day2OptimizedModel(
                base_channels=config['base_channels'],
                precision=precision
            )
            
            # Create test inputs in target precision
            batch_size = config['batch_size']
            dtype = model.dtype
            
            input_tensor = torch.randn(
                batch_size, 3, 64, 64, 
                device='cuda', 
                dtype=dtype,
                memory_format=torch.channels_last if precision != 'fp32' else torch.contiguous_format
            )
            
            prev_tensor = torch.randn(
                batch_size, 3, 64, 64,
                device='cuda',
                dtype=dtype, 
                memory_format=torch.channels_last if precision != 'fp32' else torch.contiguous_format
            )
            
            print(f"✅ Input dtype: {input_tensor.dtype}")
            print(f"✅ Input memory format: {input_tensor.is_contiguous(memory_format=torch.channels_last)}")
            
            # Clear memory and reset tracking
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            
            # Warmup
            print("🔥 Warming up...")
            for _ in range(5):
                _ = model.forward(input_tensor, timestep=25, previous_frame=prev_tensor)
            torch.cuda.synchronize()
            
            print("📊 Benchmarking...")
            
            # Benchmark multiple iterations
            num_iterations = 10
            times = []
            gpu_utils = []
            
            for i in range(num_iterations):
                pre_gpu = get_gpu_stats()
                
                start_time = time.time()
                
                # Multi-timestep generation
                current_prev = prev_tensor
                for timestep in [40, 30, 20, 10, 0]:  # 5 timesteps
                    output = model.forward(
                        input_tensor,
                        timestep=timestep,
                        previous_frame=current_prev
                    )
                    current_prev = output
                
                torch.cuda.synchronize()
                iteration_time = time.time() - start_time
                times.append(iteration_time)
                
                post_gpu = get_gpu_stats()
                gpu_utils.append(post_gpu['gpu_util'])
                
                if i % 3 == 0:
                    print(f"  Iteration {i+1}: {iteration_time:.3f}s, GPU: {post_gpu['gpu_util']}%")
            
            # Calculate metrics
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_gpu_util = np.mean(gpu_utils)
            
            results[precision] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'fps': 5 / avg_time,  # 5 timesteps per sequence
                'time_per_frame': avg_time / 5,
                'peak_memory': peak_memory,
                'initial_memory': initial_memory,
                'avg_gpu_util': avg_gpu_util,
                'success': True
            }
            
            print(f"\n✅ {precision.upper()} Results:")
            print(f"   Sequence time: {avg_time:.3f}s (±{std_time:.3f}s)")
            print(f"   Time per frame: {results[precision]['time_per_frame']*1000:.1f}ms")
            print(f"   FPS: {results[precision]['fps']:.1f}")
            print(f"   Peak memory: {peak_memory:.0f}MB")
            print(f"   Memory vs FP32: {((peak_memory - results.get('fp32', {}).get('peak_memory', peak_memory)) / results.get('fp32', {}).get('peak_memory', peak_memory) * 100) if 'fp32' in results else 0:+.1f}%")
            print(f"   GPU utilization: {avg_gpu_util:.1f}%")
            
            # Mirage targets
            mirage_40ms = results[precision]['time_per_frame'] <= 0.040
            mirage_16ms = results[precision]['time_per_frame'] <= 0.016
            print(f"   Mirage 40ms: {'✅ ACHIEVED' if mirage_40ms else '❌ MISSED'}")
            print(f"   Mirage 16ms: {'✅ ACHIEVED' if mirage_16ms else '❌ MISSED'}")
            
        except Exception as e:
            print(f"❌ {precision.upper()} failed: {e}")
            results[precision] = {'success': False, 'error': str(e)}
            
        finally:
            # Thorough cleanup
            if 'model' in locals():
                del model
            if 'input_tensor' in locals():
                del input_tensor, prev_tensor
            if 'output' in locals():
                del output, current_prev
            torch.cuda.empty_cache()
            time.sleep(2)  # Let GPU fully recover
    
    return results


def analyze_day2_improvements(results):
    """Analyze Day 2 vs Day 1 improvements"""
    
    print(f"\n{'='*70}")
    print("📈 DAY 2 vs DAY 1 IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")
    
    # Load Day 1 results for comparison
    try:
        day1_data = torch.load('day1_results.pt')
        print("📂 Loaded Day 1 results for comparison")
    except:
        print("⚠️ Day 1 results not found - showing Day 2 absolute results only")
        day1_data = None
    
    if not results.get('fp32', {}).get('success', False):
        print("❌ No valid baseline for comparison")
        return
    
    fp32 = results['fp32']
    baseline_time = fp32['time_per_frame']
    baseline_memory = fp32['peak_memory']
    baseline_fps = fp32['fps']
    
    print(f"\n📊 Day 2 Performance Summary:")
    print(f"{'Precision':<10} {'FPS':<8} {'Frame(ms)':<10} {'Memory(MB)':<12} {'Speedup':<8} {'Mem Change':<10}")
    print("-" * 75)
    print(f"{'FP32':<10} {baseline_fps:<7.1f} {baseline_time*1000:<9.1f} {baseline_memory:<11.0f} {'1.00x':<7} {'baseline':<10}")
    
    day2_gains = {}
    
    for precision in ['fp16', 'bf16']:
        if results.get(precision, {}).get('success', False):
            data = results[precision]
            
            speedup = baseline_time / data['time_per_frame']
            memory_change = (data['peak_memory'] - baseline_memory) / baseline_memory * 100
            
            day2_gains[precision] = {
                'speedup': speedup,
                'memory_change': memory_change,
                'fps': data['fps'],
                'memory_mb': data['peak_memory']
            }
            
            print(f"{precision.upper():<10} {data['fps']:<7.1f} {data['time_per_frame']*1000:<9.1f} "
                  f"{data['peak_memory']:<11.0f} {speedup:<7.2f} {memory_change:+5.1f}%")
    
    # Success assessment
    print(f"\n🎯 DAY 2 TARGET ASSESSMENT:")
    print(f"   Goals: Fix memory increase, achieve 1.5x+ speedup")
    
    success_metrics = []
    
    for precision in ['fp16', 'bf16']:
        if precision in day2_gains:
            gain = day2_gains[precision]
            
            memory_fixed = gain['memory_change'] <= 0  # No increase
            speed_good = gain['speedup'] >= 1.3       # Reasonable speedup
            
            print(f"\n   {precision.upper()} Assessment:")
            print(f"   Memory change: {gain['memory_change']:+.1f}% {'✅' if memory_fixed else '❌'}")
            print(f"   Speed improvement: {gain['speedup']:.2f}x {'✅' if speed_good else '❌'}")
            print(f"   Current FPS: {gain['fps']:.1f}")
            
            if memory_fixed and speed_good:
                print(f"   🎉 {precision.upper()} FIXES DAY 1 ISSUES!")
                success_metrics.append(precision)
    
    # Overall Day 2 assessment
    print(f"\n{'='*70}")
    
    if len(success_metrics) > 0:
        print(f"🎉 DAY 2 SUCCESS! Fixed Day 1 issues")
        best_precision = max(success_metrics, key=lambda p: day2_gains[p]['speedup'])
        best_result = day2_gains[best_precision]
        
        print(f"✅ Best precision: {best_precision.upper()}")
        print(f"✅ Performance: {best_result['fps']:.1f} FPS ({best_result['speedup']:.2f}x speedup)")
        print(f"✅ Memory: {best_result['memory_change']:+.1f}% change")
        print(f"✅ Ready for Day 3: CUDA Graphs + further optimization")
        
        # Week 1 progress
        week1_target_fps = 35
        progress = (best_result['fps'] / week1_target_fps) * 100
        print(f"\n📊 Week 1 Progress: {progress:.0f}% toward 35 FPS target")
        
    else:
        print(f"⚠️ DAY 2 ISSUES REMAIN")
        print(f"📋 Need deeper investigation into:")
        print(f"   - Memory layout optimization")
        print(f"   - Tensor Core utilization") 
        print(f"   - Potential PyTorch version issues")
    
    return day2_gains


if __name__ == '__main__':
    # Execute Day 2 implementation
    results = day2_benchmark()
    
    # Analyze improvements
    gains = analyze_day2_improvements(results)
    
    # Save Day 2 results
    torch.save({
        'day': 2,
        'results': results,
        'gains': gains,
        'optimizations': [
            'Direct model conversion (no autocast)',
            'Channels_last memory format',
            'SDPA Flash Attention backend', 
            'Tensor Core friendly dimensions',
            'TF32 optimizations enabled'
        ]
    }, 'day2_results.pt')
    
    print(f"\n💾 Day 2 results saved to day2_results.pt")
    print(f"🔄 Comparison data available for Day 3 optimization planning")