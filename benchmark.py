"""
Automated Benchmarking Infrastructure for Mirage Hello
Tracks performance improvements and prevents regressions

Usage:
    python benchmark.py --model heavy --duration 60    # Run heavy model for 60 seconds
    python benchmark.py --compare baseline.json        # Compare against baseline
    python benchmark.py --all                          # Run all benchmarks
"""

import torch
import time
import json
import argparse
import os
import platform
import subprocess
from datetime import datetime
from collections import defaultdict
import numpy as np

# Try to import GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False

from working_gpu_demo import GPUStressTestModel, get_gpu_stats


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for video diffusion models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.system_info = self.get_system_info()
        
    def get_system_info(self):
        """Collect system information for benchmark reproducibility"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory // 1024**2,  # MB
                "gpu_compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
            })
        
        return info
    
    def benchmark_model(self, model_config, duration=30, warmup=5):
        """Benchmark a specific model configuration"""
        
        print(f"🔥 Benchmarking: {model_config['name']}")
        print(f"   Duration: {duration}s, Warmup: {warmup}s")
        
        # Create model
        model = GPUStressTestModel(
            channels=3,
            base_channels=model_config['base_channels'],
            num_timesteps=50
        ).to(self.device)
        
        model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / 1024**2  # Assuming FP32
        
        # Create test data
        batch_size = model_config['batch_size']
        resolution = model_config['resolution']
        
        input_tensor = torch.randn(batch_size, 3, resolution, resolution, device=self.device)
        prev_tensor = torch.randn(batch_size, 3, resolution, resolution, device=self.device)
        
        # Get initial GPU state
        initial_gpu = get_gpu_stats()
        
        # Warmup
        print(f"   Warming up for {warmup}s...")
        warmup_start = time.time()
        with torch.no_grad():
            while time.time() - warmup_start < warmup:
                _ = model(input_tensor, timestep=25, previous_frame=prev_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"   Running benchmark for {duration}s...")
        
        frame_times = []
        gpu_utils = []
        memory_usage = []
        power_usage = []
        temperatures = []
        
        start_time = time.time()
        frame_count = 0
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                # Single frame timing
                frame_start = time.time()
                
                # Generate with multiple timesteps (like real usage)
                for timestep in [40, 30, 20, 10, 0]:
                    output = model(input_tensor, timestep=timestep, previous_frame=prev_tensor)
                    prev_tensor = output
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                
                # Collect GPU stats
                gpu_stats = get_gpu_stats()
                gpu_utils.append(gpu_stats['gpu_util'])
                memory_usage.append(gpu_stats['memory_used'])
                
                # Additional GPU metrics if available
                if NVML_AVAILABLE:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        power_usage.append(power)
                        temperatures.append(temp)
                    except:
                        pass
                
                frame_count += 1
        
        total_duration = time.time() - start_time
        
        # Calculate metrics
        results = {
            "model_config": model_config,
            "system_info": self.system_info,
            "model_stats": {
                "parameters": total_params,
                "model_size_mb": model_size_mb,
            },
            "performance": {
                "total_duration": total_duration,
                "frames_generated": frame_count,
                "avg_fps": frame_count / total_duration,
                "avg_frame_time": np.mean(frame_times),
                "min_frame_time": np.min(frame_times),
                "max_frame_time": np.max(frame_times),
                "frame_time_std": np.std(frame_times),
                "frame_time_p95": np.percentile(frame_times, 95),
                "frame_time_p99": np.percentile(frame_times, 99),
            },
            "gpu_metrics": {
                "avg_gpu_util": np.mean(gpu_utils),
                "max_gpu_util": np.max(gpu_utils),
                "avg_memory_usage": np.mean(memory_usage),
                "max_memory_usage": np.max(memory_usage),
                "memory_efficiency": np.mean(memory_usage) / self.system_info.get('gpu_memory_total', 1),
            },
            "targets": {
                "mirage_current_ms": 40,
                "mirage_next_gen_ms": 16, 
                "current_ms": np.mean(frame_times) * 1000,
                "vs_mirage_current": (np.mean(frame_times) * 1000) / 40,
                "vs_mirage_next_gen": (np.mean(frame_times) * 1000) / 16,
            }
        }
        
        if power_usage:
            results["gpu_metrics"].update({
                "avg_power_usage": np.mean(power_usage),
                "max_power_usage": np.max(power_usage),
            })
        
        if temperatures:
            results["gpu_metrics"].update({
                "avg_temperature": np.mean(temperatures),
                "max_temperature": np.max(temperatures),
            })
        
        # Cleanup
        del model, input_tensor, prev_tensor, output
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return results
    
    def run_benchmark_suite(self, duration=30):
        """Run complete benchmark suite"""
        
        print("🚀 Running Mirage Hello Benchmark Suite")
        print(f"System: {self.system_info['platform']}")
        print(f"GPU: {self.system_info.get('gpu_name', 'CPU only')}")
        print("=" * 80)
        
        # Model configurations to benchmark
        configs = [
            {
                "name": "Lightweight", 
                "base_channels": 96, 
                "batch_size": 1, 
                "resolution": 64,
                "description": "Fast model for real-time applications"
            },
            {
                "name": "Medium", 
                "base_channels": 128, 
                "batch_size": 2, 
                "resolution": 64,
                "description": "Balanced speed/quality model"
            },
            {
                "name": "Heavy", 
                "base_channels": 192, 
                "batch_size": 2, 
                "resolution": 64,
                "description": "High quality model (current best)"
            },
        ]
        
        suite_results = {
            "suite_info": {
                "timestamp": datetime.now().isoformat(),
                "duration_per_model": duration,
                "total_models": len(configs),
            },
            "system_info": self.system_info,
            "model_results": {}
        }
        
        for config in configs:
            try:
                result = self.benchmark_model(config, duration=duration)
                suite_results["model_results"][config["name"]] = result
                
                # Print summary
                perf = result["performance"]
                targets = result["targets"]
                gpu = result["gpu_metrics"]
                
                print(f"\n📊 {config['name']} Results:")
                print(f"   FPS: {perf['avg_fps']:.1f} (±{perf['frame_time_std']*perf['avg_fps']:.1f})")
                print(f"   Frame Time: {perf['avg_frame_time']*1000:.1f}ms (p95: {perf['frame_time_p95']*1000:.1f}ms)")
                print(f"   GPU Util: {gpu['avg_gpu_util']:.1f}% (max: {gpu['max_gpu_util']:.1f}%)")
                print(f"   Memory: {gpu['avg_memory_usage']:.0f}MB ({gpu['memory_efficiency']*100:.1f}% of total)")
                print(f"   vs Mirage 40ms: {targets['vs_mirage_current']:.1f}x {'faster' if targets['vs_mirage_current'] < 1 else 'slower'}")
                print(f"   vs Mirage 16ms: {targets['vs_mirage_next_gen']:.1f}x {'faster' if targets['vs_mirage_next_gen'] < 1 else 'slower'}")
                
                if targets['vs_mirage_current'] <= 1.0:
                    print("   🎯 MIRAGE CURRENT TARGET ACHIEVED!")
                if targets['vs_mirage_next_gen'] <= 1.0:
                    print("   🚀 MIRAGE NEXT-GEN TARGET ACHIEVED!")
                
            except Exception as e:
                print(f"   ❌ Benchmark failed: {e}")
                suite_results["model_results"][config["name"]] = {"error": str(e)}
        
        return suite_results
    
    def save_results(self, results, filename=None):
        """Save benchmark results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def compare_results(self, current_file, baseline_file):
        """Compare current results against baseline"""
        
        print(f"📈 Comparing {current_file} vs {baseline_file}")
        
        with open(current_file, 'r') as f:
            current = json.load(f)
        
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        
        for model_name in current["model_results"]:
            if model_name not in baseline["model_results"]:
                continue
            
            curr = current["model_results"][model_name]
            base = baseline["model_results"][model_name]
            
            if "error" in curr or "error" in base:
                continue
            
            curr_fps = curr["performance"]["avg_fps"]
            base_fps = base["performance"]["avg_fps"]
            fps_change = (curr_fps - base_fps) / base_fps * 100
            
            curr_memory = curr["gpu_metrics"]["avg_memory_usage"]
            base_memory = base["gpu_metrics"]["avg_memory_usage"]
            memory_change = (curr_memory - base_memory) / base_memory * 100
            
            print(f"\n🔄 {model_name} Model:")
            print(f"   FPS: {base_fps:.1f} → {curr_fps:.1f} ({fps_change:+.1f}%)")
            print(f"   Memory: {base_memory:.0f}MB → {curr_memory:.0f}MB ({memory_change:+.1f}%)")
            
            if fps_change > 5:
                print("   🚀 PERFORMANCE IMPROVED!")
            elif fps_change < -5:
                print("   ⚠️ PERFORMANCE REGRESSION!")
            else:
                print("   ➡️ Performance stable")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Mirage Hello performance')
    parser.add_argument('--model', choices=['lightweight', 'medium', 'heavy'], 
                       help='Benchmark specific model')
    parser.add_argument('--duration', type=int, default=30,
                       help='Benchmark duration in seconds')
    parser.add_argument('--all', action='store_true',
                       help='Run full benchmark suite')
    parser.add_argument('--compare', type=str,
                       help='Compare against baseline file')
    parser.add_argument('--save', type=str,
                       help='Save results to specific filename')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not available - benchmarks will be CPU-only")
    
    if args.model:
        # Single model benchmark
        config_map = {
            'lightweight': {"name": "Lightweight", "base_channels": 96, "batch_size": 1, "resolution": 64},
            'medium': {"name": "Medium", "base_channels": 128, "batch_size": 2, "resolution": 64},
            'heavy': {"name": "Heavy", "base_channels": 192, "batch_size": 2, "resolution": 64},
        }
        
        config = config_map[args.model]
        result = benchmark.benchmark_model(config, duration=args.duration)
        
        results = {
            "system_info": benchmark.system_info,
            "model_results": {config["name"]: result}
        }
        
    else:
        # Full benchmark suite
        results = benchmark.run_benchmark_suite(duration=args.duration)
    
    # Save results
    filename = benchmark.save_results(results, args.save)
    
    # Compare if requested
    if args.compare:
        benchmark.compare_results(filename, args.compare)


if __name__ == '__main__':
    main()