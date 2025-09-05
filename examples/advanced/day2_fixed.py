"""
Day 2 Fixed: Proper Mixed Precision Implementation
Following expert guidance to fix Day 1 issues

Key fixes:
1. Proper tensor creation API
2. Direct model conversion (no autocast overhead)  
3. Channels_last memory format
4. SDPA with Flash Attention
5. TF32 optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  
torch.backends.cudnn.benchmark = True

print("✅ Optimizations enabled: TF32, CUDNN benchmark")

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False


def get_gpu_stats():
    """Get GPU utilization and memory stats"""
    if not NVML_AVAILABLE:
        return {"gpu_util": 0, "memory_used": 0}
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            "gpu_util": util.gpu,
            "memory_used": mem_info.used // 1024**2,
        }
    except:
        return {"gpu_util": 0, "memory_used": 0}


class SimpleOptimizedUNet(nn.Module):
    """Simplified U-Net optimized for mixed precision and Tensor Cores"""
    
    def __init__(self, in_channels=4, out_channels=3, base_channels=128):
        super().__init__()
        
        # Ensure Tensor Core friendly dimensions (multiples of 8)
        base_channels = ((base_channels + 7) // 8) * 8
        
        # Simplified architecture for clear performance measurement
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder  
        self.enc1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        
        # Bottleneck with attention
        self.bottleneck_conv = nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1)
        
        # Simple attention (head_dim = 64 for Flash Attention optimization)
        attn_channels = base_channels * 8
        self.num_heads = attn_channels // 64  # head_dim = 64
        self.attention = nn.MultiheadAttention(attn_channels, self.num_heads, batch_first=True)
        
        # Decoder
        self.dec2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        
        # Output
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        print(f"🏗️ Simplified U-Net: {base_channels} base channels")
        print(f"   Attention: {attn_channels} channels, {self.num_heads} heads, head_dim=64")
    
    def forward(self, x):
        # Encoder
        x = F.silu(self.conv_in(x))
        
        skip1 = x
        x = F.silu(self.enc1(x))  # 32x32
        
        skip2 = x  
        x = F.silu(self.enc2(x))  # 16x16
        
        # Bottleneck with attention
        x = F.silu(self.bottleneck_conv(x))  # 16x16
        
        # Apply attention
        B, C, H, W = x.shape
        x_attn = x.view(B, C, H * W).transpose(1, 2)  # B, HW, C
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x_attn.transpose(1, 2).view(B, C, H, W)
        
        # Decoder
        x = F.silu(self.dec2(x))  # 32x32
        x = x + skip2  # Skip connection
        
        x = F.silu(self.dec1(x))  # 64x64  
        x = x + skip1  # Skip connection
        
        return torch.tanh(self.conv_out(x))


class Day2Model:
    """Properly optimized model following expert guidance"""
    
    def __init__(self, base_channels=128, precision='fp16'):
        self.precision = precision
        self.device = torch.device('cuda')
        
        # Create model
        self.model = SimpleOptimizedUNet(
            in_channels=4,  # RGB + timestep
            out_channels=3,
            base_channels=base_channels
        ).to(self.device)
        
        # Convert to target precision + memory format
        if precision == 'fp16':
            self.dtype = torch.float16
            print("🔄 Converting to FP16 + channels_last...")
            self.model = self.model.to(dtype=torch.float16, memory_format=torch.channels_last)
        elif precision == 'bf16':
            self.dtype = torch.bfloat16  
            print("🔄 Converting to BF16 + channels_last...")
            self.model = self.model.to(dtype=torch.bfloat16, memory_format=torch.channels_last)
        else:
            self.dtype = torch.float32
            
        self.model.eval()
        
        # Verify conversion
        sample_param = next(self.model.parameters())
        print(f"✅ Model converted: {sample_param.dtype}")
        
        if precision != 'fp32':
            is_channels_last = sample_param.is_contiguous(memory_format=torch.channels_last)
            print(f"✅ Channels_last: {is_channels_last}")
    
    def forward(self, x, timestep, previous_frame=None):
        """Optimized inference - no autocast needed, model already converted"""
        
        # Create timestep embedding
        B, C, H, W = x.shape
        t_embed = torch.full((B, 1, H, W), timestep / 100.0, 
                           device=x.device, dtype=self.dtype)
        
        # Concatenate input + timestep
        x_input = torch.cat([x, t_embed], dim=1)
        
        # Ensure correct memory format for inputs
        if self.precision != 'fp32':
            x_input = x_input.contiguous(memory_format=torch.channels_last)
        
        # Pure inference (no autocast - model already converted)
        with torch.inference_mode():
            return self.model(x_input)


def comprehensive_day2_benchmark():
    """Comprehensive Day 2 benchmark showing the fixes"""
    
    print("🎯 DAY 2: COMPREHENSIVE BENCHMARK WITH EXPERT FIXES")
    print("=" * 80)
    
    # Lighter model for clearer results
    base_channels = 96  # Smaller for isolation of effects
    batch_size = 1
    
    results = {}
    
    for precision in ['fp32', 'fp16', 'bf16']:
        print(f"\n{'='*50}")
        print(f"Testing {precision.upper()} - Expert Optimized")
        print(f"{'='*50}")
        
        try:
            # Create optimized model
            model = Day2Model(base_channels=base_channels, precision=precision)
            
            # Create properly formatted inputs
            if precision == 'fp32':
                input_tensor = torch.randn(batch_size, 3, 64, 64, device='cuda')
                prev_tensor = torch.randn(batch_size, 3, 64, 64, device='cuda')
            else:
                dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
                
                # Create tensors in target format
                input_tensor = torch.randn(batch_size, 3, 64, 64, device='cuda', dtype=dtype)
                prev_tensor = torch.randn(batch_size, 3, 64, 64, device='cuda', dtype=dtype)
                
                # Convert to channels_last
                input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
                prev_tensor = prev_tensor.contiguous(memory_format=torch.channels_last)
            
            print(f"✅ Input tensor: {input_tensor.dtype}, channels_last: {input_tensor.is_contiguous(memory_format=torch.channels_last)}")
            
            # Memory tracking
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            
            # Warmup
            print("Warming up...")
            for _ in range(5):
                _ = model.forward(input_tensor, timestep=25, previous_frame=prev_tensor)
            torch.cuda.synchronize()
            
            print("Benchmarking...")
            
            # Benchmark
            num_iterations = 12
            times = []
            
            for i in range(num_iterations):
                start_time = time.time()
                
                # Generate sequence (multiple timesteps)
                current_prev = prev_tensor
                for timestep in [50, 40, 30, 20, 10, 0]:  # 6 timesteps
                    output = model.forward(input_tensor, timestep=timestep, previous_frame=current_prev)
                    current_prev = output
                
                torch.cuda.synchronize()
                iteration_time = time.time() - start_time
                times.append(iteration_time)
                
                if i % 4 == 0:
                    gpu_stats = get_gpu_stats()
                    print(f"  Iter {i+1}: {iteration_time:.3f}s, GPU: {gpu_stats['gpu_util']}%")
            
            # Results
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results[precision] = {
                'avg_time': avg_time,
                'std_time': std_time, 
                'fps': 6 / avg_time,  # 6 timesteps
                'time_per_frame': avg_time / 6,
                'peak_memory': peak_memory,
                'model_params': sum(p.numel() for p in model.model.parameters()),
                'success': True
            }
            
            print(f"\n📊 {precision.upper()} Results:")
            print(f"   Avg time: {avg_time:.3f}s (±{std_time:.3f}s)")
            print(f"   FPS: {results[precision]['fps']:.1f}")
            print(f"   Time per frame: {results[precision]['time_per_frame']*1000:.1f}ms")
            print(f"   Peak memory: {peak_memory:.0f}MB")
            print(f"   Model size: {results[precision]['model_params'] * 2 if precision != 'fp32' else results[precision]['model_params'] * 4:.0f} MB")
            
        except Exception as e:
            print(f"❌ {precision.upper()} failed: {e}")
            results[precision] = {'success': False, 'error': str(e)}
            
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            if 'input_tensor' in locals():
                del input_tensor, prev_tensor
            if 'output' in locals():
                del output, current_prev
            torch.cuda.empty_cache()
    
    return results


def analyze_day2_gains(results):
    """Analyze Day 2 performance gains"""
    
    print(f"\n{'='*70}")
    print("📊 DAY 2 PERFORMANCE ANALYSIS")  
    print(f"{'='*70}")
    
    if not results.get('fp32', {}).get('success', False):
        print("❌ No valid results to analyze")
        return
    
    fp32 = results['fp32']
    baseline_fps = fp32['fps']
    baseline_memory = fp32['peak_memory']
    baseline_time = fp32['time_per_frame']
    
    print(f"\n📈 Performance Comparison:")
    print(f"{'Precision':<10} {'FPS':<8} {'Frame(ms)':<10} {'Memory(MB)':<12} {'Speedup':<8} {'Mem Change':<10}")
    print("-" * 75)
    
    gains = {}
    
    for precision in ['fp32', 'fp16', 'bf16']:
        if results.get(precision, {}).get('success', False):
            data = results[precision]
            
            if precision == 'fp32':
                speedup_str = "1.00x"
                mem_change_str = "baseline"
            else:
                speedup = baseline_time / data['time_per_frame']
                memory_change = (data['peak_memory'] - baseline_memory) / baseline_memory * 100
                speedup_str = f"{speedup:.2f}x"
                mem_change_str = f"{memory_change:+.1f}%"
                
                gains[precision] = {
                    'speedup': speedup,
                    'memory_change': memory_change,
                    'fps_gain': (data['fps'] - baseline_fps) / baseline_fps * 100
                }
            
            print(f"{precision.upper():<10} {data['fps']:<7.1f} {data['time_per_frame']*1000:<9.1f} "
                  f"{data['peak_memory']:<11.0f} {speedup_str:<7} {mem_change_str:<10}")
    
    # Assessment
    print(f"\n🎯 Day 2 Success Assessment:")
    
    success_count = 0
    best_result = None
    
    for precision in ['fp16', 'bf16']:
        if precision in gains:
            gain = gains[precision]
            
            # Success criteria
            memory_improved = gain['memory_change'] <= 5  # Allow small increase
            speed_improved = gain['speedup'] >= 1.2       # Modest but real improvement
            
            print(f"\n   {precision.upper()}:")
            print(f"   Memory: {gain['memory_change']:+.1f}% {'✅' if memory_improved else '❌'}")
            print(f"   Speed: {gain['speedup']:.2f}x {'✅' if speed_improved else '❌'}")
            print(f"   FPS gain: {gain['fps_gain']:+.1f}%")
            
            if memory_improved and speed_improved:
                success_count += 1
                if best_result is None or gain['speedup'] > best_result['speedup']:
                    best_result = gain.copy()
                    best_result['precision'] = precision
    
    # Final assessment
    print(f"\n{'='*70}")
    
    if success_count > 0:
        print(f"🎉 DAY 2 SUCCESS! Fixed Day 1 memory issues")
        print(f"✅ Working mixed precision: {best_result['precision'].upper()}")
        print(f"✅ Speedup: {best_result['speedup']:.2f}x")
        print(f"✅ Memory: {best_result['memory_change']:+.1f}% change")
        
        # Progress toward Week 1 target
        current_fps = baseline_fps * best_result['speedup']
        week1_target = 35
        progress = min(100, (current_fps / week1_target) * 100)
        
        print(f"\n📈 Week 1 Progress:")
        print(f"   Current: {current_fps:.1f} FPS")
        print(f"   Target: {week1_target} FPS") 
        print(f"   Progress: {progress:.0f}%")
        
        if current_fps >= week1_target:
            print(f"🚀 WEEK 1 TARGET ACHIEVED EARLY!")
        else:
            remaining = week1_target / current_fps
            print(f"   Need {remaining:.1f}x more speedup for Week 1 goal")
    else:
        print(f"⚠️ Day 2 issues remain - need Day 3 investigation")
    
    return gains


if __name__ == '__main__':
    print("🚀 DAY 2: EXPERT-GUIDED MIXED PRECISION FIXES")
    print("=" * 80)
    print("Implementing: Direct conversion, channels_last, SDPA, TF32")
    print()
    
    # Run benchmark
    results = comprehensive_day2_benchmark()
    
    # Analyze gains
    gains = analyze_day2_gains(results)
    
    print(f"\n💾 Day 2 results complete!")
    print(f"🎯 Next steps based on results:")
    
    if any(g.get('speedup', 0) >= 1.2 for g in gains.values()):
        print(f"   ✅ Mixed precision working - proceed to Day 3 (CUDA Graphs)")
        print(f"   ✅ Focus: Eliminate launch overhead + static shapes")
    else:
        print(f"   ⚠️ Mixed precision needs deeper investigation")
        print(f"   🔍 Check: Tensor Core utilization, memory layout, API compatibility")