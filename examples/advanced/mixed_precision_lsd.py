"""
Mixed Precision Live Stream Diffusion - Week 1 Implementation
Based on detailed research and memory profiling

This is the Day 1 starter implementation showing:
- FP16/BF16 support with automatic hardware detection
- Memory optimization techniques
- Performance benchmarking integration
- Gradient scaling for stability
"""

import torch
import torch.nn as nn
# Support both old and new PyTorch API
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
from working_gpu_demo import GPUStressTestModel, ProductionScaleUNet, get_gpu_stats


def get_optimal_precision():
    """Auto-detect best precision for current hardware"""
    if not torch.cuda.is_available():
        return 'fp32'
    
    # Check GPU generation
    device_capability = torch.cuda.get_device_properties(0)
    
    if device_capability.major >= 8:  # RTX 30/40 series (Ampere+)
        return 'bf16'  # Best numerical stability
    elif device_capability.major >= 7:  # RTX 20 series (Turing+)  
        return 'fp16'  # Good Tensor Core support
    else:
        return 'fp32'  # Older GPUs


def make_tensor_core_friendly(channels):
    """Round channels to optimal Tensor Core dimensions (multiples of 8)"""
    return ((channels + 7) // 8) * 8


class AdaptiveGradScaler:
    """Enhanced gradient scaler with stability monitoring"""
    
    def __init__(self, init_scale=2**15, growth_factor=2.0, backoff_factor=0.5):
        # Support both old and new PyTorch API
        try:
            self.scaler = GradScaler('cuda',
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=200  # Conservative growth
            )
        except TypeError:
            # Fallback to older API
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=200  # Conservative growth
            )
        self.overflow_count = 0
        
    def scale_and_step(self, optimizer, loss):
        """Scale loss, backward pass, and optimizer step with monitoring"""
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Unscale gradients for clipping
        self.scaler.unscale_(optimizer)
        
        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']], 
            max_norm=1.0
        )
        
        # Check for overflow
        if torch.isfinite(grad_norm):
            self.scaler.step(optimizer)
        else:
            self.overflow_count += 1
            print(f"⚠️ Gradient overflow detected (count: {self.overflow_count})")
        
        self.scaler.update()
        return grad_norm


class MemoryPool:
    """Pre-allocated tensor pool to reduce memory fragmentation"""
    
    def __init__(self, precision='fp16', pool_sizes=None):
        self.precision = precision
        self.dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16, 
            'fp32': torch.float32
        }[precision]
        
        self.pools = {}
        
        # Pre-allocate common tensor sizes
        if pool_sizes is None:
            pool_sizes = [
                (1, 3, 64, 64),    # Single batch input
                (2, 3, 64, 64),    # Batch size 2 input
                (4, 3, 64, 64),    # Batch size 4 input
                (2, 192, 32, 32),  # Intermediate features
                (2, 384, 16, 16),  # Deeper features
                (2, 768, 8, 8),    # Bottleneck features
            ]
        
        print(f"🏊 Initializing memory pool ({precision})...")
        for shape in pool_sizes:
            key = tuple(shape)
            self.pools[key] = torch.empty(shape, dtype=self.dtype, device='cuda')
            print(f"   Pool {shape}: {torch.numel(self.pools[key]) * self.dtype.itemsize / 1024**2:.1f}MB")
    
    def get_tensor(self, shape):
        """Get pre-allocated tensor or create new one"""
        key = tuple(shape)
        if key in self.pools:
            return self.pools[key]
        else:
            return torch.empty(shape, dtype=self.dtype, device='cuda')


class MixedPrecisionWrapper:
    """Wrapper to add mixed precision support to any model"""
    
    def __init__(self, model, precision='auto', use_memory_pool=True):
        self.model = model
        self.precision = get_optimal_precision() if precision == 'auto' else precision
        self.use_autocast = self.precision in ['fp16', 'bf16']
        self.dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32
        }[self.precision]
        
        # Gradient scaler for FP16
        self.scaler = AdaptiveGradScaler() if self.precision == 'fp16' else None
        
        # Memory pool for efficiency  
        self.memory_pool = MemoryPool(self.precision) if use_memory_pool else None
        
        # Optimize model for Tensor Cores
        self._optimize_model()
        
        print(f"🔧 Mixed Precision: {self.precision.upper()}")
        print(f"   Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"   Memory pool: {'Enabled' if use_memory_pool else 'Disabled'}")
        print(f"   Gradient scaling: {'Enabled' if self.scaler else 'Disabled'}")
    
    def _optimize_model(self):
        """Apply model optimizations for mixed precision"""
        
        # Enable CUDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Convert model to target precision for inference
        if self.precision != 'fp32':
            self.model = self.model.to(self.dtype)
        
        # Optimize memory format for convolutions
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use channels_last for better memory access patterns
                if module.weight.size(1) >= 4:  # Only for channels >= 4
                    module.weight.data = module.weight.data.contiguous(
                        memory_format=torch.channels_last
                    )
    
    def forward(self, *args, **kwargs):
        """Forward pass with mixed precision"""
        if self.use_autocast:
            # Support both old and new PyTorch API
            try:
                with autocast('cuda', dtype=self.dtype):
                    return self.model(*args, **kwargs)
            except TypeError:
                # Fallback to older API
                with autocast(enabled=True):
                    return self.model(*args, **kwargs).to(self.dtype)
        else:
            return self.model(*args, **kwargs)
    
    def training_step(self, optimizer, loss_fn, *args, **kwargs):
        """Complete training step with mixed precision"""
        if not self.use_autocast:
            # Standard FP32 training
            optimizer.zero_grad()
            output = self.forward(*args, **kwargs)
            loss = loss_fn(output)
            loss.backward()
            optimizer.step()
            return output, loss
        
        # Mixed precision training
        optimizer.zero_grad()
        
        # Support both old and new PyTorch API
        try:
            with autocast('cuda', dtype=self.dtype):
                output = self.model(*args, **kwargs)
                loss = loss_fn(output)
        except TypeError:
            # Fallback to older API
            with autocast(enabled=True):
                output = self.model(*args, **kwargs)
                loss = loss_fn(output)
        
        if self.scaler:
            # FP16 with gradient scaling
            grad_norm = self.scaler.scale_and_step(optimizer, loss)
            return output, loss, grad_norm
        else:
            # BF16 - no scaling needed
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 
                max_norm=1.0
            )
            optimizer.step()
            return output, loss, grad_norm


class OptimizedMixedPrecisionLSD:
    """Complete optimized LSD system with mixed precision"""
    
    def __init__(self, base_channels=192, precision='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = get_optimal_precision() if precision == 'auto' else precision
        
        # Create optimized model
        optimized_channels = make_tensor_core_friendly(base_channels)
        if optimized_channels != base_channels:
            print(f"🔧 Tensor Core optimization: {base_channels} → {optimized_channels} channels")
        
        self.base_model = GPUStressTestModel(
            channels=3,
            base_channels=optimized_channels
        ).to(self.device)
        
        # Wrap with mixed precision
        self.model = MixedPrecisionWrapper(
            self.base_model, 
            precision=self.precision,
            use_memory_pool=True
        )
        
        # Performance tracking
        self.performance_history = []
    
    def generate_sequence(self, input_tensor, previous_frame=None, num_timesteps=5):
        """Generate video sequence with mixed precision"""
        
        if previous_frame is None:
            previous_frame = torch.zeros_like(input_tensor)
        
        start_time = time.time()
        
        # Track GPU memory
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Generate sequence
        with torch.inference_mode():  # Faster than no_grad for inference
            for timestep in range(num_timesteps-1, -1, -1):  # Reverse timesteps
                timestep_val = timestep * 10  # Scale to model's expected range
                
                output = self.model.forward(
                    input_tensor, 
                    timestep=timestep_val, 
                    previous_frame=previous_frame
                )
                previous_frame = output
        
        # Performance metrics
        total_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        gpu_stats = get_gpu_stats()
        
        performance = {
            'total_time': total_time,
            'fps': num_timesteps / total_time,
            'time_per_frame': total_time / num_timesteps,
            'peak_memory': peak_memory,
            'memory_efficiency': initial_memory / peak_memory if peak_memory > 0 else 0,
            'gpu_utilization': gpu_stats.get('gpu_util', 0),
            'precision': self.precision
        }
        
        self.performance_history.append(performance)
        
        return output, performance


def mixed_precision_benchmark():
    """Comprehensive mixed precision benchmark"""
    
    print("🚀 MIXED PRECISION BENCHMARK - WEEK 1 IMPLEMENTATION")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - mixed precision requires GPU")
        return
    
    # Test configurations  
    configs = [
        {'name': 'Lightweight', 'base_channels': 128, 'batch_size': 1},
        {'name': 'Medium', 'base_channels': 192, 'batch_size': 2},
        {'name': 'Heavy', 'base_channels': 256, 'batch_size': 2},
    ]
    
    precisions = ['fp32', 'fp16', 'bf16']
    
    results = {}
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']} Model ({config['base_channels']} channels)")
        print("-" * 60)
        
        for precision in precisions:
            print(f"\n📊 Precision: {precision.upper()}")
            
            try:
                # Create optimized model
                lsd = OptimizedMixedPrecisionLSD(
                    base_channels=config['base_channels'],
                    precision=precision
                )
                
                # Create test input
                batch_size = config['batch_size']
                input_tensor = torch.randn(batch_size, 3, 64, 64, device='cuda')
                
                # Warmup
                for _ in range(3):
                    _, _ = lsd.generate_sequence(input_tensor, num_timesteps=3)
                
                # Benchmark
                num_runs = 5
                performances = []
                
                for run in range(num_runs):
                    output, perf = lsd.generate_sequence(input_tensor, num_timesteps=5)
                    performances.append(perf)
                    
                    if run == 0:  # First run details
                        print(f"   Time: {perf['time_per_frame']*1000:.1f}ms/frame")
                        print(f"   Memory: {perf['peak_memory']:.0f}MB") 
                        print(f"   GPU: {perf['gpu_utilization']}%")
                
                # Average results
                avg_perf = {
                    'time_per_frame': np.mean([p['time_per_frame'] for p in performances]),
                    'peak_memory': np.mean([p['peak_memory'] for p in performances]),
                    'fps': np.mean([p['fps'] for p in performances])
                }
                
                results[(config['name'], precision)] = avg_perf
                
                print(f"   📈 Average: {avg_perf['time_per_frame']*1000:.1f}ms/frame, "
                      f"{avg_perf['fps']:.1f} FPS, {avg_perf['peak_memory']:.0f}MB")
                
                # Mirage targets
                mirage_40ms = avg_perf['time_per_frame'] <= 0.040
                mirage_16ms = avg_perf['time_per_frame'] <= 0.016
                
                print(f"   🎯 Mirage 40ms: {'✅' if mirage_40ms else '❌'}")
                print(f"   🎯 Mirage 16ms: {'✅' if mirage_16ms else '❌'}")
                
                del lsd
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[(config['name'], precision)] = {'error': str(e)}
    
    # Results summary
    print(f"\n{'='*80}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    for config in configs:
        print(f"\n{config['name']} Model:")
        
        for precision in precisions:
            key = (config['name'], precision)
            if key in results and 'error' not in results[key]:
                perf = results[key]
                improvement = ""
                
                # Calculate improvement vs FP32
                fp32_key = (config['name'], 'fp32')
                if fp32_key in results and precision != 'fp32':
                    fp32_time = results[fp32_key]['time_per_frame']
                    speedup = fp32_time / perf['time_per_frame']
                    fp32_memory = results[fp32_key]['peak_memory']
                    memory_reduction = (fp32_memory - perf['peak_memory']) / fp32_memory * 100
                    improvement = f" ({speedup:.1f}x faster, {memory_reduction:.0f}% less memory)"
                
                print(f"  {precision.upper():4s}: {perf['time_per_frame']*1000:5.1f}ms/frame, "
                      f"{perf['peak_memory']:5.0f}MB{improvement}")
    
    print(f"\n🎯 Week 1 Success Metrics:")
    print(f"   Target: 35+ FPS (≤28.6ms/frame) with 30%+ memory reduction")
    
    # Check if we hit Week 1 targets
    medium_fp16 = results.get(('Medium', 'fp16'))
    if medium_fp16 and 'error' not in medium_fp16:
        success = medium_fp16['time_per_frame'] <= 0.0286  # 35 FPS
        memory_target = results.get(('Medium', 'fp32'), {}).get('peak_memory', 0) * 0.7
        memory_success = medium_fp16['peak_memory'] <= memory_target if memory_target else False
        
        print(f"   Week 1 Status: {'✅ SUCCESS' if success and memory_success else '⚠️ IN PROGRESS'}")
        
        if success:
            print(f"   🚀 Performance target achieved: {medium_fp16['fps']:.1f} FPS")
        if memory_success:  
            print(f"   💾 Memory target achieved: {medium_fp16['peak_memory']:.0f}MB")
    
    return results


if __name__ == '__main__':
    # Run the mixed precision benchmark
    results = mixed_precision_benchmark()
    
    print(f"\n🎉 Mixed precision implementation complete!")
    print(f"   Next: Day 2-7 optimizations per WEEK1_MIXED_PRECISION_PLAN.md")
    print(f"   Goal: 35+ FPS by end of Week 1")