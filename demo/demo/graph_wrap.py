"""
CUDA Graphs Wrapper for Static Shape Inference
Eliminates Python/launch overhead for real-time performance

Usage:
    graph_runner = GraphRunner(model, example_input)
    output = graph_runner(new_input)  # Much faster than direct model call
"""

import torch
import time


class GraphRunner:
    """CUDA Graph wrapper for static shape inference"""
    
    def __init__(self, model, example_input, warmup_steps=10, enable_timing=True):
        """
        Initialize CUDA Graph for static model inference
        
        Args:
            model: PyTorch model to wrap
            example_input: Example input tensor (defines static shape)
            warmup_steps: Number of warmup iterations before capture
            enable_timing: Whether to track timing performance
        """
        self.model = model
        self.device = example_input.device
        self.enable_timing = enable_timing
        
        # Create static input/output buffers
        self.static_input = example_input.clone()
        
        # Warmup the model
        print(f"🔥 CUDA Graphs: Warming up {warmup_steps} steps...")
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(warmup_steps):
                example_output = self.model(example_input)
        
        torch.cuda.synchronize()
        
        # Create static output buffer
        self.static_output = torch.empty_like(example_output)
        
        # Capture the graph
        print(f"📸 CUDA Graphs: Capturing computation graph...")
        self.graph = torch.cuda.CUDAGraph()
        
        torch.cuda.synchronize()
        
        # Capture phase
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)
        
        torch.cuda.synchronize()
        
        print(f"✅ CUDA Graph captured successfully")
        print(f"   Input shape: {self.static_input.shape}")
        print(f"   Output shape: {self.static_output.shape}")
        
        # Performance tracking
        self.call_count = 0
        self.total_time = 0
        
    @torch.inference_mode()
    def __call__(self, x):
        """Fast inference using CUDA Graph replay"""
        
        # Validate input shape matches captured graph
        if x.shape != self.static_input.shape:
            raise ValueError(f"Input shape {x.shape} doesn't match captured shape {self.static_input.shape}")
        
        # Timing (optional)
        if self.enable_timing:
            start_time = time.time()
        
        # Copy input to static buffer and replay graph
        self.static_input.copy_(x, non_blocking=True)
        self.graph.replay()
        
        if self.enable_timing:
            torch.cuda.synchronize()  # Only for timing
            self.total_time += time.time() - start_time
            self.call_count += 1
        
        # Return copy of output (static buffer will be overwritten)
        return self.static_output.clone()
    
    def get_performance_stats(self):
        """Get CUDA Graph performance statistics"""
        if self.call_count == 0:
            return {'avg_time': 0, 'avg_fps': 0, 'call_count': 0}
        
        avg_time = self.total_time / self.call_count
        return {
            'avg_time': avg_time,
            'avg_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'call_count': self.call_count,
            'total_time': self.total_time
        }


class AdaptiveGraphRunner:
    """Adaptive CUDA Graphs for multiple static shapes"""
    
    def __init__(self, model, enable_timing=True):
        self.model = model
        self.enable_timing = enable_timing
        self.graphs = {}  # Shape -> GraphRunner mapping
        self.current_graph = None
        
        print("🔧 Adaptive CUDA Graphs initialized")
    
    def get_or_create_graph(self, x):
        """Get existing graph or create new one for this shape"""
        shape_key = tuple(x.shape)
        
        if shape_key not in self.graphs:
            print(f"📸 Creating new CUDA Graph for shape {x.shape}")
            self.graphs[shape_key] = GraphRunner(
                self.model, 
                x, 
                warmup_steps=5,
                enable_timing=self.enable_timing
            )
        
        return self.graphs[shape_key]
    
    @torch.inference_mode()
    def __call__(self, x):
        """Adaptive inference with CUDA Graph optimization"""
        graph_runner = self.get_or_create_graph(x)
        return graph_runner(x)
    
    def get_performance_summary(self):
        """Get performance summary across all graphs"""
        total_calls = 0
        total_time = 0
        
        for shape, graph in self.graphs.items():
            stats = graph.get_performance_stats()
            total_calls += stats['call_count']
            total_time += stats['total_time']
        
        return {
            'total_graphs': len(self.graphs),
            'total_calls': total_calls,
            'avg_fps': total_calls / total_time if total_time > 0 else 0,
            'shapes_cached': list(self.graphs.keys())
        }


def benchmark_cuda_graphs():
    """Benchmark CUDA Graphs vs standard inference"""
    
    print("🚀 CUDA GRAPHS BENCHMARK")
    print("=" * 50)
    
    from model_tiny_viz import TinyVizModel
    
    # Create test model
    model = TinyVizModel(base_channels=64, dtype='fp16', channels_last=True)
    
    # Test data
    batch_size = 2
    input_shape = (batch_size, 3, 288, 512)  # 512x288 as specified
    test_input = torch.randn(input_shape, device='cuda', dtype=torch.float16)
    test_input = test_input.contiguous(memory_format=torch.channels_last)
    
    num_iterations = 50
    
    # Test 1: Standard PyTorch inference
    print("\n📊 Standard PyTorch Inference:")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model.process_frame(test_input)
    
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    pytorch_fps = num_iterations / pytorch_time
    
    print(f"   Total time: {pytorch_time:.3f}s")
    print(f"   Average FPS: {pytorch_fps:.1f}")
    print(f"   Time per frame: {pytorch_time/num_iterations*1000:.2f}ms")
    
    # Test 2: CUDA Graphs inference  
    print("\n📊 CUDA Graphs Inference:")
    
    graph_runner = GraphRunner(model.model, test_input, warmup_steps=5)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = graph_runner(test_input)
    
    torch.cuda.synchronize()
    graph_time = time.time() - start_time
    graph_fps = num_iterations / graph_time
    
    print(f"   Total time: {graph_time:.3f}s")
    print(f"   Average FPS: {graph_fps:.1f}")
    print(f"   Time per frame: {graph_time/num_iterations*1000:.2f}ms")
    
    # Comparison
    speedup = pytorch_time / graph_time
    overhead_reduction = (pytorch_time - graph_time) / pytorch_time * 100
    
    print(f"\n📈 CUDA Graphs Impact:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Overhead reduction: {overhead_reduction:.1f}%")
    print(f"   Time saved per frame: {(pytorch_time - graph_time)/num_iterations*1000:.2f}ms")
    
    if speedup >= 1.2:
        print("   ✅ Significant CUDA Graphs speedup achieved!")
    elif speedup >= 1.1:
        print("   ✅ Moderate CUDA Graphs speedup")
    else:
        print("   ⚠️ Limited CUDA Graphs impact - model may be too simple")
    
    return {
        'pytorch_fps': pytorch_fps,
        'graph_fps': graph_fps,
        'speedup': speedup,
        'overhead_reduction': overhead_reduction
    }


if __name__ == '__main__':
    benchmark_cuda_graphs()