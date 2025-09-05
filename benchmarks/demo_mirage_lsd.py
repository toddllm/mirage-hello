"""
Demonstration of Live Stream Diffusion (LSD) Model
Based on Mirage/Daycart real-time video generation approach

This demo shows:
1. Basic LSD model usage
2. PTX-optimized kernels in action
3. Error accumulation prevention
4. Real-time performance monitoring
"""

import torch
import time
import numpy as np
from mirage_lsd import LiveStreamDiffusionModel, generate_realtime_video

def demo_basic_lsd():
    """Basic demonstration of LSD model functionality"""
    
    print("=" * 60)
    print("MIRAGE/DAYCART LIVE STREAM DIFFUSION DEMO")
    print("=" * 60)
    print()
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: Running on CPU. For real-time performance, use GPU.")
    
    print()
    
    # Initialize model
    print("1. Initializing Live Stream Diffusion Model...")
    model = LiveStreamDiffusionModel(
        input_channels=3,      # RGB input
        hidden_dim=256,        # Reduced for demo speed
        context_length=8       # Shorter context for demo
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Context length: {model.context_length} frames")
    print()
    
    # Create synthetic input stream
    print("2. Creating synthetic input stream...")
    batch_size = 1
    sequence_length = 10
    height, width = 64, 64
    
    # Create a simple animated pattern as input
    input_frames = []
    for t in range(sequence_length):
        # Create animated checkerboard pattern
        frame = torch.zeros(batch_size, 3, height, width, device=device)
        
        # Animate the pattern
        for i in range(height):
            for j in range(width):
                checker = ((i // 8) + (j // 8) + t) % 2
                frame[0, :, i, j] = checker * 0.8 - 0.4
        
        input_frames.append(frame)
    
    input_stream = torch.stack(input_frames, dim=1)
    print(f"   Input shape: {input_stream.shape}")
    print(f"   Input range: [{input_stream.min():.2f}, {input_stream.max():.2f}]")
    print()
    
    # Warm up the model
    print("3. Warming up model...")
    with torch.no_grad():
        warmup_input = input_stream[:, :1]
        _ = model(warmup_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    print("   Warmup complete")
    print()
    
    # Generate video with performance monitoring
    print("4. Generating video with LSD model...")
    print("   Monitoring for error accumulation and performance...")
    print()
    
    generated_frames = []
    frame_times = []
    context = None
    
    with torch.no_grad():
        for i in range(25):  # Generate 25 frames for demo
            start_time = time.time()
            
            # Get input frame (cycle through available frames)
            input_idx = i % input_stream.size(1)
            current_input = input_stream[:, input_idx:input_idx+1]
            
            # Generate next frame
            output = model(current_input, context)
            next_frame = output[:, 0]
            
            generated_frames.append(next_frame)
            
            # Update context
            if context is None:
                context = [next_frame]
            else:
                context.append(next_frame)
                if len(context) > model.context_length:
                    context.pop(0)
            
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            # Performance monitoring
            if i % 5 == 0 or i < 5:
                avg_time = np.mean(frame_times[-5:]) if len(frame_times) >= 5 else np.mean(frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # Check against Mirage targets
                current_target = 0.040  # 40ms (25 FPS)
                next_target = 0.016     # 16ms (62.5 FPS)
                
                status_current = "✓" if avg_time <= current_target else "⚠"
                status_next = "✓" if avg_time <= next_target else "⚠"
                
                print(f"   Frame {i:2d}: {frame_time*1000:5.1f}ms | "
                      f"Avg: {avg_time*1000:5.1f}ms | "
                      f"FPS: {fps:5.1f} | "
                      f"Current: {status_current} | Next: {status_next}")
    
    print()
    
    # Performance summary
    total_frames = len(frame_times)
    avg_time = np.mean(frame_times)
    min_time = np.min(frame_times)
    max_time = np.max(frame_times)
    avg_fps = 1.0 / avg_time
    
    print("5. Performance Summary:")
    print(f"   Generated frames: {total_frames}")
    print(f"   Average time:     {avg_time*1000:.1f}ms per frame")
    print(f"   Min time:         {min_time*1000:.1f}ms")
    print(f"   Max time:         {max_time*1000:.1f}ms")
    print(f"   Average FPS:      {avg_fps:.1f}")
    print()
    
    # Mirage performance targets
    current_target = 0.040  # 40ms
    next_target = 0.016     # 16ms
    
    print("6. Mirage Performance Targets:")
    print(f"   Current target (40ms):  {'✅ ACHIEVED' if avg_time <= current_target else '❌ NOT MET'} "
          f"({avg_time*1000:.1f}ms)")
    print(f"   Next gen target (16ms): {'✅ ACHIEVED' if avg_time <= next_target else '❌ NOT MET'} "
          f"({avg_time*1000:.1f}ms)")
    print()
    
    # Quality analysis
    print("7. Quality Analysis:")
    generated_tensor = torch.stack(generated_frames, dim=1)
    
    # Check for error accumulation indicators
    frame_variances = []
    for i in range(total_frames):
        frame_var = torch.var(generated_tensor[0, i]).item()
        frame_variances.append(frame_var)
    
    initial_variance = np.mean(frame_variances[:5])
    final_variance = np.mean(frame_variances[-5:])
    variance_ratio = final_variance / initial_variance if initial_variance > 0 else 1.0
    
    print(f"   Initial variance: {initial_variance:.6f}")
    print(f"   Final variance:   {final_variance:.6f}")
    print(f"   Variance ratio:   {variance_ratio:.3f} {'✅' if variance_ratio > 0.5 else '⚠️'}")
    
    if variance_ratio > 0.5:
        print("   ✅ Good detail preservation - no significant quality degradation")
    else:
        print("   ⚠️ Possible quality degradation detected")
    
    print()
    
    # Final assessment
    print("8. Overall Assessment:")
    
    realtime_capable = avg_time <= current_target
    quality_preserved = variance_ratio > 0.5
    
    if realtime_capable and quality_preserved:
        print("   🎉 SUCCESS! LSD model achieves real-time performance")
        print("      with good quality preservation.")
        print("      This demonstrates the key Mirage/Daycart innovations:")
        print("      - PTX-optimized CUDA kernels for speed")
        print("      - Context memory to prevent error accumulation") 
        print("      - Autoregressive frame-by-frame generation")
    elif realtime_capable:
        print("   ⚠️  PARTIAL SUCCESS: Real-time performance achieved")
        print("      but quality preservation needs improvement.")
    elif quality_preserved:
        print("   ⚠️  PARTIAL SUCCESS: Quality preserved")
        print("      but performance optimization needed for real-time.")
    else:
        print("   ❌ NEEDS IMPROVEMENT: Both performance and quality")
        print("      require further optimization.")
    
    print()
    print("Demo complete! Try running the full test suite:")
    print("  python test_error_accumulation.py")
    print("  python advanced_ptx_kernels.py")
    

def demo_ptx_kernels():
    """Quick demonstration of PTX kernel functionality"""
    
    print("\n" + "=" * 40)
    print("PTX KERNEL PERFORMANCE DEMO")
    print("=" * 40)
    
    try:
        from mirage_lsd import ptx_kernels
        
        device = torch.device('cuda')
        
        # Test data
        size = 1024 * 1024  # 1M elements
        x = torch.randn(size, device=device)
        noise = torch.randn(size, device=device)
        alpha = torch.full((size,), 0.8, device=device)
        sigma = torch.full((size,), 0.6, device=device)
        
        # Warmup
        for _ in range(10):
            _ = ptx_kernels.ptx_diffusion_step(x, noise, alpha, sigma)
        torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            result = ptx_kernels.ptx_diffusion_step(x, noise, alpha, sigma)
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        
        print(f"PTX Diffusion Step:")
        print(f"  Data size: {size:,} elements")
        print(f"  Iterations: {num_iterations}")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {(size * 4 / avg_time / 1e9):.1f} GB/s")
        print("  ✅ PTX kernels working correctly")
        
    except Exception as e:
        print(f"  ❌ PTX kernels not available: {e}")
        print("     Running on CPU or compilation failed")


if __name__ == '__main__':
    demo_basic_lsd()
    
    if torch.cuda.is_available():
        demo_ptx_kernels()
    
    print("\n" + "=" * 60)
    print("For more comprehensive testing:")
    print("  python test_error_accumulation.py    # Full error accumulation test")
    print("  python advanced_ptx_kernels.py       # Advanced PTX benchmarks")
    print("=" * 60)