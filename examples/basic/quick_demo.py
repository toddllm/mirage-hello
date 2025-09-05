"""
🚀 Quick Demo - See Mirage Hello Working in 30 seconds

This is the simplest entry point to see our real-time video diffusion working.
Perfect for first-time users who want to quickly understand what we've built.

Usage:
    python examples/basic/quick_demo.py

What you'll see:
- GPU utilization and memory usage 
- Real-time frame generation
- Performance vs Mirage targets
- Error accumulation prevention in action
"""

import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from simplified_lsd import LiveStreamDiffusionModel
    print("✅ Using optimized simplified LSD model")
except ImportError:
    print("⚠️ Using basic demo model (simplified_lsd.py not found)")
    
    # Fallback basic model
    class QuickDemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(6, 64, 3, padding=1)  # input + previous
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
            
        def forward(self, x, prev):
            x = torch.cat([x, prev], dim=1)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return torch.tanh(self.conv3(x))


def quick_demo():
    """30-second demo of real-time video diffusion"""
    
    print("🎬 MIRAGE HELLO - QUICK DEMO")
    print("=" * 50)
    print("⚡ Demonstrating real-time video diffusion in 30 seconds")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️ Running on CPU - for real performance, use GPU")
    
    # Create simple model
    try:
        model = LiveStreamDiffusionModel().to(device)
        model_name = "Optimized LSD Model"
    except:
        model = QuickDemoModel().to(device)
        model_name = "Basic Demo Model"
    
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {model_name} ({params:,} parameters)")
    
    # Create animated test input
    print(f"🎨 Creating animated test pattern...")
    frames = []
    for t in range(10):
        frame = torch.zeros(1, 3, 64, 64, device=device)
        
        # Create moving sine wave pattern
        x = torch.linspace(0, 4*np.pi, 64, device=device)
        y = torch.linspace(0, 4*np.pi, 64, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        pattern = torch.sin(X + t * 0.5) * torch.cos(Y + t * 0.3) * 0.5
        frame[0, 0] = pattern
        frame[0, 1] = torch.roll(pattern, shifts=t*3, dims=0) 
        frame[0, 2] = torch.roll(pattern, shifts=-t*3, dims=1)
        
        frames.append(frame)
    
    input_stream = torch.stack(frames, dim=1)
    print(f"✅ Input stream: {input_stream.shape}")
    
    # Demo generation
    print(f"\n🎬 Generating real-time video (30 frames)...")
    print("   Watch for consistent performance and no error accumulation")
    
    generated_frames = []
    times = []
    
    # Initialize 
    if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 3:
        # LSD model with context
        context = None
        
        with torch.no_grad():
            for i in range(30):
                start = time.time()
                
                # Get input frame (cycle through available)
                input_idx = i % input_stream.size(1)
                current_input = input_stream[:, input_idx:input_idx+1]
                
                # Generate with context
                if hasattr(model, '__call__'):
                    output, context = model(current_input, context)
                    next_frame = output[:, 0]
                else:
                    next_frame = model(current_input[:, 0], frames[-1] if generated_frames else current_input[:, 0])
                
                generated_frames.append(next_frame)
                times.append(time.time() - start)
                
                if i % 5 == 0:
                    avg_time = np.mean(times[-5:]) if len(times) >= 5 else np.mean(times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"   Frame {i:2d}: {times[-1]*1000:4.1f}ms | Avg: {avg_time*1000:4.1f}ms | FPS: {fps:6.1f}")
    else:
        # Simple model 
        previous_frame = frames[0]
        
        with torch.no_grad():
            for i in range(30):
                start = time.time()
                
                input_idx = i % len(frames)
                current_input = frames[input_idx]
                
                next_frame = model(current_input, previous_frame)
                generated_frames.append(next_frame)
                previous_frame = next_frame
                
                times.append(time.time() - start)
                
                if i % 5 == 0:
                    avg_time = np.mean(times[-5:]) if len(times) >= 5 else np.mean(times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"   Frame {i:2d}: {times[-1]*1000:4.1f}ms | Avg: {avg_time*1000:4.1f}ms | FPS: {fps:6.1f}")
    
    # Results summary
    total_time = sum(times)
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n📊 DEMO RESULTS:")
    print(f"   Total time: {total_time:.2f}s for 30 frames")
    print(f"   Average FPS: {fps:.1f}")
    print(f"   Time per frame: {avg_time*1000:.1f}ms")
    
    # Compare to Mirage
    mirage_target = 0.040  # 40ms
    if avg_time <= mirage_target:
        print(f"   🎯 Mirage target: ✅ ACHIEVED ({avg_time*1000:.1f}ms ≤ 40ms)")
    else:
        print(f"   🎯 Mirage target: ⚠️ {avg_time/mirage_target:.1f}x slower than 40ms")
    
    # Error accumulation check
    if len(generated_frames) >= 10:
        early_var = torch.var(torch.stack(generated_frames[:10])).item()
        late_var = torch.var(torch.stack(generated_frames[-10:])).item()
        stability = late_var / early_var if early_var > 0 else 1.0
        
        print(f"   🔄 Error accumulation: {'✅ Stable' if stability > 0.5 else '⚠️ Degrading'} (ratio: {stability:.2f})")
    
    print(f"\n🎉 Demo complete! This shows the core concepts working.")
    print(f"   For optimization details: docs/intermediate/")
    print(f"   For contributing: CONTRIBUTING.md")
    print(f"   For benchmarking: benchmarks/run_benchmarks.py")


if __name__ == '__main__':
    quick_demo()