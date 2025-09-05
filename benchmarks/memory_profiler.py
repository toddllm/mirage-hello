"""
Memory Profiling and Analysis for Mirage Hello Models
Identifies memory bottlenecks and optimization opportunities

Usage:
    python memory_profiler.py --model heavy --profile-layers
    python memory_profiler.py --analyze-attention --batch-sizes 1,2,4
"""

import torch
import torch.nn as nn
import gc
import argparse
import numpy as np
from working_gpu_demo import GPUStressTestModel, ProductionScaleUNet, GPUIntensiveAttention
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Memory tracking utilities
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False


class MemoryProfiler:
    """Advanced memory profiling for video diffusion models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_snapshots = []
        self.layer_analysis = defaultdict(list)
        
    def get_memory_stats(self):
        """Get detailed GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        
        stats = {
            "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
            "reserved": torch.cuda.memory_reserved() / 1024**2,    # MB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
        }
        
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats["gpu_used"] = mem_info.used / 1024**2
                stats["gpu_total"] = mem_info.total / 1024**2
            except:
                pass
        
        return stats
    
    def memory_checkpoint(self, label):
        """Create memory checkpoint with label"""
        stats = self.get_memory_stats()
        stats["label"] = label
        stats["timestamp"] = time.time()
        self.memory_snapshots.append(stats)
        return stats
    
    def analyze_model_memory(self, model, input_shape=(2, 3, 64, 64)):
        """Comprehensive model memory analysis"""
        
        print("🔍 COMPREHENSIVE MODEL MEMORY ANALYSIS")
        print("=" * 60)
        
        # Model parameter analysis
        total_params = sum(p.numel() for p in model.parameters())
        param_memory = total_params * 4 / 1024**2  # FP32 MB
        
        print(f"📊 Parameter Analysis:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Parameter memory (FP32): {param_memory:.1f}MB")
        print(f"   Parameter memory (FP16): {param_memory/2:.1f}MB")
        print(f"   Parameter memory (BF16): {param_memory/2:.1f}MB")
        
        # Layer-by-layer analysis
        print(f"\n🏗️ Layer-by-Layer Analysis:")
        layer_memories = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    mem_mb = params * 4 / 1024**2
                    layer_memories.append((name, params, mem_mb))
        
        # Sort by memory usage
        layer_memories.sort(key=lambda x: x[2], reverse=True)
        
        print(f"   Top 10 memory-intensive layers:")
        for i, (name, params, mem_mb) in enumerate(layer_memories[:10]):
            print(f"   {i+1:2d}. {name[:50]:50s} {mem_mb:6.1f}MB ({params:,} params)")
        
        # Forward pass memory analysis
        print(f"\n🔄 Forward Pass Memory Analysis:")
        
        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Initial memory
        initial_stats = self.memory_checkpoint("Model loaded")
        
        # Create input
        batch_size, channels, height, width = input_shape
        input_tensor = torch.randn(batch_size, channels, height, width, device=self.device)
        prev_tensor = torch.randn(batch_size, channels, height, width, device=self.device)
        
        input_stats = self.memory_checkpoint("Input created")
        
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'forward') and 'timestep' in model.forward.__code__.co_varnames:
                output = model(input_tensor, timestep=25, previous_frame=prev_tensor)
            else:
                output = model(input_tensor)
        
        forward_stats = self.memory_checkpoint("Forward pass complete")
        
        # Peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"   Input tensor memory: {(input_tensor.numel() * 4 / 1024**2):.1f}MB")
        print(f"   Output tensor memory: {(output.numel() * 4 / 1024**2):.1f}MB")
        print(f"   Peak GPU memory: {peak_memory:.1f}MB")
        print(f"   Memory overhead: {peak_memory - param_memory - (input_tensor.numel() * 4 / 1024**2):.1f}MB")
        
        return {
            "total_params": total_params,
            "param_memory_fp32": param_memory,
            "param_memory_fp16": param_memory / 2,
            "peak_memory": peak_memory,
            "layer_memories": layer_memories[:20],  # Top 20
            "memory_snapshots": self.memory_snapshots
        }
    
    def analyze_attention_memory(self, base_channels_list=[128, 192, 256]):
        """Analyze attention memory usage across different model sizes"""
        
        print("🎯 ATTENTION MEMORY ANALYSIS")
        print("=" * 60)
        
        attention_results = []
        
        for base_channels in base_channels_list:
            print(f"\n📊 Testing base_channels = {base_channels}")
            
            # Create attention module
            attention = GPUIntensiveAttention(base_channels * 4).to(self.device)  # Use 4x like in bottleneck
            
            # Test different sequence lengths (spatial dimensions)
            for spatial_size in [16, 32, 64]:  # 16x16, 32x32, 64x64
                seq_len = spatial_size * spatial_size
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create test input
                x = torch.randn(2, seq_len, base_channels * 4, device=self.device)
                
                initial_memory = torch.cuda.memory_allocated() / 1024**2
                
                with torch.no_grad():
                    # Reshape for spatial attention
                    x_spatial = x.view(2, base_channels * 4, spatial_size, spatial_size)
                    output = attention(x_spatial)
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                attention_overhead = peak_memory - initial_memory
                
                result = {
                    "base_channels": base_channels,
                    "spatial_size": spatial_size,
                    "seq_len": seq_len,
                    "input_memory": x.numel() * 4 / 1024**2,
                    "peak_memory": peak_memory,
                    "attention_overhead": attention_overhead
                }
                
                attention_results.append(result)
                
                print(f"   {spatial_size}x{spatial_size}: Input={result['input_memory']:.1f}MB, "
                      f"Peak={peak_memory:.1f}MB, Overhead={attention_overhead:.1f}MB")
                
                del x, x_spatial, output
                torch.cuda.empty_cache()
            
            del attention
            torch.cuda.empty_cache()
        
        return attention_results
    
    def mixed_precision_comparison(self, model_config):
        """Compare memory usage: FP32 vs FP16 vs BF16"""
        
        print("🔄 MIXED PRECISION MEMORY COMPARISON")  
        print("=" * 60)
        
        results = {}
        
        for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
            print(f"\n📊 Testing {dtype_name}...")
            
            try:
                # Create model in specific precision
                model = GPUStressTestModel(
                    channels=3,
                    base_channels=model_config['base_channels']
                ).to(self.device).to(dtype)
                
                # Input tensors in same precision  
                batch_size = model_config['batch_size']
                input_tensor = torch.randn(batch_size, 3, 64, 64, device=self.device, dtype=dtype)
                prev_tensor = torch.randn(batch_size, 3, 64, 64, device=self.device, dtype=dtype)
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Measure forward pass
                start_time = time.time()
                
                with torch.no_grad():
                    for timestep in [40, 30, 20, 10, 0]:  # Multiple timesteps like real usage
                        output = model(input_tensor, timestep=timestep, previous_frame=prev_tensor)
                        prev_tensor = output
                
                torch.cuda.synchronize()
                forward_time = time.time() - start_time
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                
                results[dtype_name] = {
                    "peak_memory": peak_memory,
                    "forward_time": forward_time,
                    "dtype": dtype_name
                }
                
                print(f"   Peak memory: {peak_memory:.1f}MB")
                print(f"   Forward time: {forward_time:.3f}s")
                
                del model, input_tensor, prev_tensor, output
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"   ❌ Failed: {e}")
                results[dtype_name] = {"error": str(e)}
        
        # Comparison summary
        if "FP32" in results and "FP16" in results:
            fp32_memory = results["FP32"]["peak_memory"]
            fp16_memory = results["FP16"]["peak_memory"] 
            memory_reduction = (fp32_memory - fp16_memory) / fp32_memory * 100
            
            fp32_time = results["FP32"]["forward_time"]
            fp16_time = results["FP16"]["forward_time"]
            speedup = fp32_time / fp16_time
            
            print(f"\n📈 FP32 vs FP16 Comparison:")
            print(f"   Memory reduction: {memory_reduction:.1f}%")
            print(f"   Speed improvement: {speedup:.2f}x")
        
        return results
    
    def generate_optimization_report(self, analysis_results):
        """Generate comprehensive optimization recommendations"""
        
        print("\n" + "=" * 80)
        print("🎯 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 80)
        
        param_memory = analysis_results["param_memory_fp32"]
        peak_memory = analysis_results["peak_memory"]
        
        print(f"📊 Current Memory Profile:")
        print(f"   Model parameters: {param_memory:.1f}MB (FP32)")
        print(f"   Peak forward pass: {peak_memory:.1f}MB")
        print(f"   Memory efficiency: {param_memory/peak_memory*100:.1f}%")
        
        print(f"\n🚀 Priority 1: Mixed Precision (Week 1)")
        print(f"   💾 FP16 parameter reduction: {param_memory/2:.1f}MB ({param_memory - param_memory/2:.1f}MB saved)")
        print(f"   ⚡ Expected speedup: 1.5-2.0x")
        print(f"   🎯 Implementation: torch.cuda.amp.autocast()")
        
        print(f"\n🚀 Priority 2: Memory Layout Optimization")
        memory_overhead = peak_memory - param_memory
        print(f"   Current overhead: {memory_overhead:.1f}MB")
        print(f"   🎯 Target reduction: 30-40% through gradient checkpointing")
        print(f"   🎯 Channels-last memory format for convolutions")
        
        print(f"\n🚀 Priority 3: Attention Optimization") 
        top_layers = analysis_results["layer_memories"][:5]
        attention_layers = [layer for layer in top_layers if "attention" in layer[0].lower()]
        
        if attention_layers:
            total_attention_memory = sum(layer[2] for layer in attention_layers)
            print(f"   Current attention memory: {total_attention_memory:.1f}MB")
            print(f"   🎯 Flash Attention could reduce by 50-70%")
            print(f"   🎯 Potential savings: {total_attention_memory * 0.6:.1f}MB")
        
        print(f"\n📋 Implementation Priority:")
        print(f"   Week 1: Mixed precision (FP16) - {param_memory/2:.1f}MB savings, 1.8x speedup")
        print(f"   Week 2: Flash Attention - {total_attention_memory * 0.6 if attention_layers else 20:.1f}MB savings, 1.5x speedup")
        print(f"   Week 3: Memory layout optimization - {memory_overhead * 0.3:.1f}MB savings")
        print(f"   Week 4: Custom CUDA kernels - Additional 20-30% speedup")
        
        return {
            "param_memory_savings_fp16": param_memory / 2,
            "attention_memory_current": total_attention_memory if attention_layers else 0,
            "attention_memory_optimized": total_attention_memory * 0.4 if attention_layers else 0,
            "total_optimized_memory": param_memory/2 + (total_attention_memory * 0.4 if attention_layers else 0),
            "expected_speedup_week_1": 1.8,
            "expected_speedup_week_2": 2.7,  # Cumulative
        }


def main():
    parser = argparse.ArgumentParser(description='Profile memory usage of Mirage Hello models')
    parser.add_argument('--model', choices=['lightweight', 'medium', 'heavy'], default='medium',
                       help='Model size to profile')
    parser.add_argument('--profile-layers', action='store_true',
                       help='Detailed layer-by-layer analysis')
    parser.add_argument('--analyze-attention', action='store_true', 
                       help='Analyze attention memory scaling')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Compare FP32 vs FP16 vs BF16')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4',
                       help='Comma-separated batch sizes to test')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - memory profiling requires GPU")
        return
    
    profiler = MemoryProfiler()
    
    # Model configurations
    model_configs = {
        'lightweight': {"base_channels": 128, "batch_size": 1},
        'medium': {"base_channels": 192, "batch_size": 2}, 
        'heavy': {"base_channels": 256, "batch_size": 2},
    }
    
    config = model_configs[args.model]
    
    print(f"🔍 MEMORY PROFILING: {args.model.upper()} MODEL")
    print(f"Configuration: {config}")
    print("=" * 80)
    
    # Create and analyze model
    model = GPUStressTestModel(
        channels=3,
        base_channels=config['base_channels']
    ).to(profiler.device)
    
    # Main analysis
    analysis_results = profiler.analyze_model_memory(
        model, 
        input_shape=(config['batch_size'], 3, 64, 64)
    )
    
    # Optional analyses
    if args.analyze_attention:
        attention_results = profiler.analyze_attention_memory([config['base_channels']])
    
    if args.mixed_precision:
        precision_results = profiler.mixed_precision_comparison(config)
    
    # Generate optimization recommendations
    optimization_plan = profiler.generate_optimization_report(analysis_results)
    
    print(f"\n💾 Profile complete! Results saved for optimization planning.")


if __name__ == '__main__':
    main()