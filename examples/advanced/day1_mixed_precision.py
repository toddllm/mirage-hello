"""
Day 1 Mixed Precision Implementation
Core FP16/BF16 infrastructure with comprehensive benchmarking

This implements the Day 1 goals from WEEK1_MIXED_PRECISION_PLAN.md:
1. Basic FP16 implementation with autocast and GradScaler
2. Model conversion utilities  
3. Performance comparison (FP32 vs FP16 vs BF16)
4. Demonstrate 30-40% memory reduction
"""

import torch
import torch.nn as nn
import time
import numpy as np
from working_gpu_demo import GPUStressTestModel, get_gpu_stats

# Compatible autocast import
try:
    from torch.amp import autocast, GradScaler
    NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  
    NEW_API = False

print(f"🔧 PyTorch API: {'New (torch.amp)' if NEW_API else 'Legacy (torch.cuda.amp)'}")


class Day1MixedPrecisionModel:
    """Day 1: Core mixed precision implementation"""
    
    def __init__(self, base_model, precision='fp16'):
        self.base_model = base_model
        self.precision = precision
        self.device = next(base_model.parameters()).device
        
        # Set up precision-specific components
        if precision == 'fp16':
            self.dtype = torch.float16
            self.use_scaler = True
            if NEW_API:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        elif precision == 'bf16':
            self.dtype = torch.bfloat16
            self.use_scaler = False
            self.scaler = None
        else:  # fp32
            self.dtype = torch.float32
            self.use_scaler = False
            self.scaler = None
        
        print(f"✅ {precision.upper()} model initialized")
        print(f"   Gradient scaler: {'Yes' if self.use_scaler else 'No'}")
    
    def forward(self, x, timestep, previous_frame=None):
        """Forward pass with mixed precision"""
        
        if self.precision == 'fp32':
            # Standard FP32
            return self.base_model(x, timestep, previous_frame)
        
        # Mixed precision forward pass
        if NEW_API:
            with autocast('cuda', dtype=self.dtype):
                return self.base_model(x, timestep, previous_frame)
        else:
            # Legacy API - manually convert
            if self.dtype != torch.float32:
                x = x.to(self.dtype)
                if previous_frame is not None:
                    previous_frame = previous_frame.to(self.dtype)
            
            with autocast(enabled=True):
                output = self.base_model(x, timestep, previous_frame)
                if self.dtype != torch.float32:
                    output = output.to(self.dtype)
                return output


def benchmark_precision(model_config, num_iterations=10, num_timesteps=5):
    """Comprehensive precision benchmark with memory tracking"""
    
    results = {}
    
    for precision in ['fp32', 'fp16', 'bf16']:
        print(f"\n🔬 Testing {precision.upper()}")
        print("-" * 40)
        
        try:
            # Create base model
            base_model = GPUStressTestModel(
                channels=3,
                base_channels=model_config['base_channels']
            ).cuda()
            
            # Wrap with mixed precision
            model = Day1MixedPrecisionModel(base_model, precision)
            
            # Create test inputs
            batch_size = model_config['batch_size']
            input_tensor = torch.randn(batch_size, 3, 64, 64, device='cuda')
            prev_tensor = torch.randn(batch_size, 3, 64, 64, device='cuda')
            
            # Convert inputs to target precision
            if precision != 'fp32':
                dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
                input_tensor = input_tensor.to(dtype)
                prev_tensor = prev_tensor.to(dtype)
            
            # Clear GPU memory and reset stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model.forward(input_tensor, timestep=25, previous_frame=prev_tensor)
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            gpu_utils = []
            
            print(f"Running {num_iterations} iterations...")
            
            for i in range(num_iterations):
                # Pre-iteration stats
                pre_gpu = get_gpu_stats()
                
                start_time = time.time()
                
                with torch.no_grad():
                    # Multi-timestep generation like real usage
                    current_prev = prev_tensor
                    for timestep in range(num_timesteps-1, -1, -1):
                        output = model.forward(
                            input_tensor, 
                            timestep=timestep * 10, 
                            previous_frame=current_prev
                        )
                        current_prev = output
                
                torch.cuda.synchronize()
                iteration_time = time.time() - start_time
                times.append(iteration_time)
                
                # Post-iteration stats
                post_gpu = get_gpu_stats()
                gpu_utils.append(post_gpu['gpu_util'])
                
                if i % 3 == 0:
                    print(f"  Iter {i+1}: {iteration_time:.3f}s, GPU: {post_gpu['gpu_util']}%")
            
            # Memory stats
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            # Calculate results
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_gpu_util = np.mean(gpu_utils)
            
            results[precision] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'fps': num_timesteps / avg_time,
                'time_per_frame': avg_time / num_timesteps,
                'peak_memory': peak_memory,
                'avg_gpu_util': avg_gpu_util,
                'success': True
            }
            
            print(f"✅ Results:")
            print(f"   Average time: {avg_time:.3f}s (±{std_time:.3f}s)")
            print(f"   FPS: {results[precision]['fps']:.1f}")
            print(f"   Time per frame: {results[precision]['time_per_frame']*1000:.1f}ms")
            print(f"   Peak memory: {peak_memory:.0f}MB")
            print(f"   GPU utilization: {avg_gpu_util:.1f}%")
            
            # Mirage targets
            mirage_40ms = results[precision]['time_per_frame'] <= 0.040
            mirage_16ms = results[precision]['time_per_frame'] <= 0.016
            print(f"   Mirage 40ms: {'✅' if mirage_40ms else '❌'}")
            print(f"   Mirage 16ms: {'✅' if mirage_16ms else '❌'}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[precision] = {'success': False, 'error': str(e)}
        
        finally:
            # Cleanup
            if 'base_model' in locals():
                del base_model
            if 'model' in locals():
                del model
            if 'input_tensor' in locals():
                del input_tensor, prev_tensor
            if 'output' in locals():
                del output
            torch.cuda.empty_cache()
            time.sleep(1)  # Let GPU recover
    
    return results


def analyze_day1_results(results):
    """Analyze Day 1 results and calculate gains"""
    
    print(f"\n{'='*60}")
    print("📊 DAY 1 RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    if not results.get('fp32', {}).get('success', False):
        print("❌ FP32 baseline failed - cannot compute gains")
        return
    
    fp32 = results['fp32']
    
    print(f"\n📈 Performance Comparison vs FP32 Baseline:")
    print(f"{'Precision':<10} {'Time(s)':<8} {'FPS':<6} {'Memory(MB)':<12} {'Speedup':<8} {'Mem Save':<8}")
    print("-" * 70)
    
    baseline_time = fp32['time_per_frame']
    baseline_memory = fp32['peak_memory']
    baseline_fps = fp32['fps']
    
    print(f"{'FP32':<10} {baseline_time:.3f}    {baseline_fps:5.1f}  {baseline_memory:8.0f}      1.00x      0%")
    
    gains = {}
    
    for precision in ['fp16', 'bf16']:
        if results.get(precision, {}).get('success', False):
            data = results[precision]
            
            speedup = baseline_time / data['time_per_frame']
            memory_save = (baseline_memory - data['peak_memory']) / baseline_memory * 100
            
            gains[precision] = {
                'speedup': speedup,
                'memory_save': memory_save,
                'fps_improvement': (data['fps'] - baseline_fps) / baseline_fps * 100
            }
            
            print(f"{precision.upper():<10} {data['time_per_frame']:.3f}    {data['fps']:5.1f}  {data['peak_memory']:8.0f}      {speedup:.2f}x    {memory_save:5.1f}%")
    
    # Day 1 success assessment
    print(f"\n🎯 DAY 1 SUCCESS METRICS:")
    print(f"   Target: 30-40% memory reduction, 1.5-1.8x speedup")
    
    success_count = 0
    
    for precision in ['fp16', 'bf16']:
        if precision in gains:
            gain = gains[precision]
            memory_target = gain['memory_save'] >= 30  # 30% target
            speed_target = gain['speedup'] >= 1.5      # 1.5x target
            
            print(f"\n   {precision.upper()} Results:")
            print(f"   Memory reduction: {gain['memory_save']:.1f}% {'✅' if memory_target else '❌'}")
            print(f"   Speed improvement: {gain['speedup']:.2f}x {'✅' if speed_target else '❌'}")
            print(f"   FPS improvement: {gain['fps_improvement']:+.1f}%")
            
            if memory_target and speed_target:
                success_count += 1
                print(f"   🎉 {precision.upper()} MEETS DAY 1 TARGETS!")
    
    # Overall assessment
    print(f"\n{'='*60}")
    if success_count > 0:
        print(f"🎉 DAY 1 SUCCESS! {success_count} precision(s) meet targets")
        print(f"✅ Mixed precision infrastructure working")
        print(f"✅ Significant performance gains demonstrated") 
        print(f"✅ Ready for Day 2: BF16 optimization & runtime selection")
    else:
        print(f"⚠️ DAY 1 PARTIAL SUCCESS")
        print(f"📋 Issues to address in Day 2 optimization")
    
    return gains


def main():
    """Execute Day 1 mixed precision implementation"""
    
    print("🚀 DAY 1: MIXED PRECISION CORE IMPLEMENTATION")
    print("=" * 80)
    print("Goals: FP16/BF16 infrastructure, 30-40% memory reduction, 1.5x+ speedup")
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - mixed precision requires GPU")
        return
    
    # Check GPU capabilities
    device_props = torch.cuda.get_device_properties(0)
    print(f"🔧 GPU: {device_props.name}")
    print(f"   Compute capability: {device_props.major}.{device_props.minor}")
    print(f"   Memory: {device_props.total_memory / 1024**3:.1f} GB")
    
    # Determine optimal test configuration
    if device_props.total_memory >= 20 * 1024**3:  # 20GB+
        test_config = {'name': 'Medium', 'base_channels': 192, 'batch_size': 2}
    else:
        test_config = {'name': 'Lightweight', 'base_channels': 128, 'batch_size': 1}
    
    print(f"   Test config: {test_config['name']} ({test_config['base_channels']} channels)")
    print()
    
    # Run comprehensive benchmark
    results = benchmark_precision(test_config, num_iterations=8, num_timesteps=5)
    
    # Analyze results and calculate gains
    gains = analyze_day1_results(results)
    
    # Save results for progress tracking
    torch.save({
        'day': 1,
        'config': test_config, 
        'results': results,
        'gains': gains,
        'gpu_info': {
            'name': device_props.name,
            'memory_gb': device_props.total_memory / 1024**3,
            'compute_capability': f"{device_props.major}.{device_props.minor}"
        }
    }, 'day1_results.pt')
    
    print(f"\n💾 Results saved to day1_results.pt")
    print(f"🎯 Next: Day 2 implementation (BF16 optimization + automatic selection)")


if __name__ == '__main__':
    main()