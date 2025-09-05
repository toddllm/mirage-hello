"""
Comprehensive test for error accumulation prevention in autoregressive video generation.

This test addresses the key challenge mentioned in the Mirage/Daycart transcript:
"That same problem that LLMs dealt with a few years ago comes back when you try to do 
auto regressive video models... the model gets stuck in this loop until it just gets 
stuck on a single color and your entire screen just becomes reds or blue or green"

Tests:
1. Long sequence generation (500+ frames)
2. Error accumulation metrics
3. Context memory effectiveness  
4. Temporal consistency verification
5. Comparison with/without error prevention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
from collections import deque

# Import our LSD implementation
from mirage_lsd import LiveStreamDiffusionModel, ContextMemoryBank, ptx_kernels

def calculate_error_metrics(frames):
    """Calculate various error accumulation metrics"""
    frames_np = frames.detach().cpu().numpy()
    B, T, C, H, W = frames_np.shape
    
    metrics = {
        'frame_variance': [],           # Variance within each frame
        'temporal_variance': [],        # Variance across time for each pixel
        'color_drift': [],              # Drift in mean color values
        'repetition_score': [],         # Similarity to recent frames
        'detail_preservation': [],      # High-frequency content preservation
        'color_collapse_score': []      # Measure of single-color collapse
    }
    
    for t in range(T):
        frame = frames_np[0, t]  # Take first batch element
        
        # Frame variance (measure of detail preservation)
        frame_var = np.var(frame)
        metrics['frame_variance'].append(frame_var)
        
        # Detail preservation (high-frequency content)
        # Use simple gradient-based edge detection
        grad_x = np.abs(np.diff(frame, axis=2)).mean()
        grad_y = np.abs(np.diff(frame, axis=1)).mean()
        detail_score = grad_x + grad_y
        metrics['detail_preservation'].append(detail_score)
        
        # Color collapse detection
        # If the model collapses, all pixels become similar
        pixel_std = np.std(frame.reshape(C, -1), axis=1).mean()
        metrics['color_collapse_score'].append(pixel_std)
        
        if t > 0:
            # Color drift (change in mean values)
            prev_mean = np.mean(frames_np[0, t-1], axis=(1, 2))
            curr_mean = np.mean(frame, axis=(1, 2))
            drift = np.linalg.norm(curr_mean - prev_mean)
            metrics['color_drift'].append(drift)
            
            # Temporal variance for pixels
            temporal_var = np.var(frames_np[0, max(0, t-10):t+1], axis=0).mean()
            metrics['temporal_variance'].append(temporal_var)
            
            # Repetition score (similarity to recent frames)
            if t >= 5:
                recent_frames = frames_np[0, t-5:t]
                similarities = []
                for prev_frame in recent_frames:
                    # Cosine similarity
                    frame_flat = frame.flatten()
                    prev_flat = prev_frame.flatten()
                    similarity = np.dot(frame_flat, prev_flat) / (
                        np.linalg.norm(frame_flat) * np.linalg.norm(prev_flat) + 1e-8
                    )
                    similarities.append(similarity)
                metrics['repetition_score'].append(np.max(similarities))
            else:
                metrics['repetition_score'].append(0.0)
    
    return metrics


class BaselineARModel(nn.Module):
    """Baseline autoregressive model WITHOUT error accumulation prevention"""
    
    def __init__(self, channels=3, hidden_dim=256):
        super().__init__()
        self.channels = channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//4, channels, 3, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        # Add some noise to simulate imperfect prediction
        noise = 0.05 * torch.randn_like(out)
        return out + noise


def test_error_accumulation(num_frames=500, save_results=True):
    """Main test function for error accumulation"""
    
    print("=" * 80)
    print("TESTING ERROR ACCUMULATION IN AUTOREGRESSIVE VIDEO GENERATION")
    print("=" * 80)
    print(f"Generating {num_frames} frames to test for error accumulation...")
    print("This addresses the core challenge from Mirage/Daycart transcript.")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 1
    channels = 3
    height, width = 64, 64
    
    # Create models
    print("\n1. Creating models...")
    lsd_model = LiveStreamDiffusionModel(
        input_channels=channels,
        hidden_dim=512,
        context_length=16
    ).to(device)
    
    baseline_model = BaselineARModel(channels=channels).to(device)
    
    # Initial frame
    initial_frame = torch.randn(batch_size, channels, height, width, device=device)
    
    print("\n2. Testing LSD Model (WITH error accumulation prevention)...")
    
    # Test LSD model
    lsd_frames = []
    lsd_context = None
    lsd_times = []
    
    with torch.no_grad():
        for i in range(num_frames):
            start_time = time.time()
            
            # For LSD, we need input stream format
            if i == 0:
                input_frame = initial_frame.unsqueeze(1)
            else:
                # Use previous generated frame as input (simulating live stream)
                input_frame = lsd_frames[-1].unsqueeze(1)
            
            # Generate next frame
            output = lsd_model(input_frame, lsd_context)
            next_frame = output[:, 0]  # Take first frame from sequence
            
            lsd_frames.append(next_frame)
            
            # Update context for LSD
            if lsd_context is None:
                lsd_context = [next_frame]
            else:
                lsd_context.append(next_frame)
                if len(lsd_context) > lsd_model.context_length:
                    lsd_context.pop(0)
            
            frame_time = time.time() - start_time
            lsd_times.append(frame_time)
            
            if i % 50 == 0:
                avg_time = np.mean(lsd_times[-10:]) if len(lsd_times) >= 10 else np.mean(lsd_times)
                print(f"  Frame {i:3d}: {frame_time*1000:5.1f}ms (avg: {avg_time*1000:5.1f}ms)")
    
    lsd_frames_tensor = torch.stack(lsd_frames).unsqueeze(0).transpose(0, 1)
    
    print("\n3. Testing Baseline Model (WITHOUT error accumulation prevention)...")
    
    # Test baseline model
    baseline_frames = []
    baseline_times = []
    
    with torch.no_grad():
        current_frame = initial_frame
        for i in range(num_frames):
            start_time = time.time()
            
            next_frame = baseline_model(current_frame)
            baseline_frames.append(next_frame)
            current_frame = next_frame  # Pure autoregressive
            
            frame_time = time.time() - start_time
            baseline_times.append(frame_time)
            
            if i % 50 == 0:
                avg_time = np.mean(baseline_times[-10:]) if len(baseline_times) >= 10 else np.mean(baseline_times)
                print(f"  Frame {i:3d}: {frame_time*1000:5.1f}ms (avg: {avg_time*1000:5.1f}ms)")
    
    baseline_frames_tensor = torch.stack(baseline_frames).unsqueeze(0).transpose(0, 1)
    
    print("\n4. Calculating error metrics...")
    
    # Calculate metrics
    lsd_metrics = calculate_error_metrics(lsd_frames_tensor)
    baseline_metrics = calculate_error_metrics(baseline_frames_tensor)
    
    print("\n5. Analysis Results:")
    print("=" * 60)
    
    # Compare final metrics
    lsd_final_variance = lsd_metrics['frame_variance'][-1]
    baseline_final_variance = baseline_metrics['frame_variance'][-1]
    
    lsd_color_collapse = np.mean(lsd_metrics['color_collapse_score'][-50:])  # Last 50 frames
    baseline_color_collapse = np.mean(baseline_metrics['color_collapse_score'][-50:])
    
    lsd_avg_repetition = np.mean(lsd_metrics['repetition_score'][-50:])
    baseline_avg_repetition = np.mean(baseline_metrics['repetition_score'][-50:])
    
    print(f"Frame Variance (detail preservation):")
    print(f"  LSD Model:      {lsd_final_variance:.6f}")
    print(f"  Baseline Model: {baseline_final_variance:.6f}")
    print(f"  Improvement:    {((lsd_final_variance - baseline_final_variance) / baseline_final_variance * 100):+.1f}%")
    print()
    
    print(f"Color Collapse Resistance (higher = better):")
    print(f"  LSD Model:      {lsd_color_collapse:.6f}")
    print(f"  Baseline Model: {baseline_color_collapse:.6f}")
    print(f"  Improvement:    {((lsd_color_collapse - baseline_color_collapse) / baseline_color_collapse * 100):+.1f}%")
    print()
    
    print(f"Repetition Score (lower = better):")
    print(f"  LSD Model:      {lsd_avg_repetition:.6f}")
    print(f"  Baseline Model: {baseline_avg_repetition:.6f}")
    print(f"  Improvement:    {((baseline_avg_repetition - lsd_avg_repetition) / baseline_avg_repetition * 100):+.1f}%")
    print()
    
    # Performance comparison
    lsd_avg_time = np.mean(lsd_times)
    baseline_avg_time = np.mean(baseline_times)
    
    print(f"Performance Comparison:")
    print(f"  LSD Model:      {lsd_avg_time*1000:.1f}ms per frame ({1/lsd_avg_time:.1f} FPS)")
    print(f"  Baseline Model: {baseline_avg_time*1000:.1f}ms per frame ({1/baseline_avg_time:.1f} FPS)")
    print(f"  Overhead:       {((lsd_avg_time - baseline_avg_time) / baseline_avg_time * 100):+.1f}%")
    print()
    
    # Mirage performance targets
    mirage_current_target = 0.040  # 40ms (25 FPS)
    mirage_next_target = 0.016     # 16ms (62.5 FPS)
    
    print(f"Mirage Performance Targets:")
    print(f"  Current (40ms):  {'✓' if lsd_avg_time <= mirage_current_target else '✗'} "
          f"({'ON TARGET' if lsd_avg_time <= mirage_current_target else f'{(lsd_avg_time/mirage_current_target-1)*100:.0f}% SLOWER'})")
    print(f"  Next Gen (16ms): {'✓' if lsd_avg_time <= mirage_next_target else '✗'} "
          f"({'ON TARGET' if lsd_avg_time <= mirage_next_target else f'{(lsd_avg_time/mirage_next_target-1)*100:.0f}% SLOWER'})")
    
    if save_results:
        print("\n6. Saving visualization...")
        save_error_analysis_plots(lsd_metrics, baseline_metrics, num_frames)
        save_sample_frames(lsd_frames_tensor, baseline_frames_tensor)
    
    # Final assessment
    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT:")
    
    error_accumulation_prevented = (
        lsd_color_collapse > baseline_color_collapse * 1.1 and  # 10% better color preservation
        lsd_avg_repetition < baseline_avg_repetition * 0.9       # 10% less repetition
    )
    
    if error_accumulation_prevented:
        print("✅ ERROR ACCUMULATION SUCCESSFULLY PREVENTED!")
        print("   The LSD model with context memory and temporal consistency")
        print("   prevents the color collapse and repetition issues that")
        print("   plagued early autoregressive video models.")
    else:
        print("❌ ERROR ACCUMULATION PREVENTION NEEDS IMPROVEMENT")
        print("   The current implementation may not be sufficient to")
        print("   prevent long-term degradation in video quality.")
    
    realtime_capable = lsd_avg_time <= mirage_current_target
    
    if realtime_capable:
        print("✅ REAL-TIME PERFORMANCE ACHIEVED!")
        print(f"   Average generation time: {lsd_avg_time*1000:.1f}ms")
        print("   Meets Mirage's 40ms target for real-time generation.")
    else:
        print("⚠️  REAL-TIME PERFORMANCE NEEDS OPTIMIZATION")
        print(f"   Current: {lsd_avg_time*1000:.1f}ms (target: {mirage_current_target*1000:.1f}ms)")
        print("   Consider further PTX optimizations or model reduction.")
    
    return lsd_metrics, baseline_metrics


def save_error_analysis_plots(lsd_metrics, baseline_metrics, num_frames):
    """Save comprehensive error analysis plots"""
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Error Accumulation Analysis: LSD vs Baseline', fontsize=16, color='white')
    
    frames_axis = range(num_frames)
    
    # Plot 1: Frame Variance (Detail Preservation)
    axes[0, 0].plot(frames_axis, lsd_metrics['frame_variance'], 
                   label='LSD Model', color='#00ff00', linewidth=2)
    axes[0, 0].plot(frames_axis, baseline_metrics['frame_variance'], 
                   label='Baseline', color='#ff4444', linewidth=2)
    axes[0, 0].set_title('Frame Variance (Detail Preservation)', color='white')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Variance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Color Collapse Score
    axes[0, 1].plot(frames_axis, lsd_metrics['color_collapse_score'], 
                   label='LSD Model', color='#00ff00', linewidth=2)
    axes[0, 1].plot(frames_axis, baseline_metrics['color_collapse_score'], 
                   label='Baseline', color='#ff4444', linewidth=2)
    axes[0, 1].set_title('Color Collapse Resistance', color='white')
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Pixel Std (higher = better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Color Drift
    if len(lsd_metrics['color_drift']) > 0:
        axes[0, 2].plot(range(1, num_frames), lsd_metrics['color_drift'], 
                       label='LSD Model', color='#00ff00', linewidth=2)
        axes[0, 2].plot(range(1, num_frames), baseline_metrics['color_drift'], 
                       label='Baseline', color='#ff4444', linewidth=2)
    axes[0, 2].set_title('Color Drift', color='white')
    axes[0, 2].set_xlabel('Frame Number')
    axes[0, 2].set_ylabel('Color Change')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Repetition Score  
    axes[1, 0].plot(frames_axis, lsd_metrics['repetition_score'], 
                   label='LSD Model', color='#00ff00', linewidth=2)
    axes[1, 0].plot(frames_axis, baseline_metrics['repetition_score'], 
                   label='Baseline', color='#ff4444', linewidth=2)
    axes[1, 0].set_title('Repetition Score (lower = better)', color='white')
    axes[1, 0].set_xlabel('Frame Number')
    axes[1, 0].set_ylabel('Max Similarity to Recent Frames')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Detail Preservation
    axes[1, 1].plot(frames_axis, lsd_metrics['detail_preservation'], 
                   label='LSD Model', color='#00ff00', linewidth=2)
    axes[1, 1].plot(frames_axis, baseline_metrics['detail_preservation'], 
                   label='Baseline', color='#ff4444', linewidth=2)
    axes[1, 1].set_title('Detail Preservation', color='white')
    axes[1, 1].set_xlabel('Frame Number')
    axes[1, 1].set_ylabel('Edge Content')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Temporal Variance
    if len(lsd_metrics['temporal_variance']) > 0:
        axes[1, 2].plot(range(1, num_frames), lsd_metrics['temporal_variance'], 
                       label='LSD Model', color='#00ff00', linewidth=2)
        axes[1, 2].plot(range(1, num_frames), baseline_metrics['temporal_variance'], 
                       label='Baseline', color='#ff4444', linewidth=2)
    axes[1, 2].set_title('Temporal Variance', color='white')
    axes[1, 2].set_xlabel('Frame Number')
    axes[1, 2].set_ylabel('Variance Across Time')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_accumulation_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='black')
    print("  Saved: error_accumulation_analysis.png")
    plt.close()


def save_sample_frames(lsd_frames, baseline_frames, sample_indices=[0, 100, 250, 499]):
    """Save sample frames for visual comparison"""
    
    fig, axes = plt.subplots(2, len(sample_indices), figsize=(20, 8))
    fig.suptitle('Sample Frame Comparison', fontsize=16, color='white')
    plt.style.use('dark_background')
    
    for i, frame_idx in enumerate(sample_indices):
        if frame_idx >= lsd_frames.size(1):
            continue
            
        # LSD frames (top row)
        lsd_frame = lsd_frames[0, frame_idx].detach().cpu().numpy()
        lsd_frame = lsd_frame.transpose(1, 2, 0)  # CHW to HWC
        lsd_frame = (lsd_frame + 1) / 2  # Assuming tanh output [-1,1] -> [0,1]
        lsd_frame = np.clip(lsd_frame, 0, 1)
        
        axes[0, i].imshow(lsd_frame)
        axes[0, i].set_title(f'LSD Frame {frame_idx}', color='white')
        axes[0, i].axis('off')
        
        # Baseline frames (bottom row)
        baseline_frame = baseline_frames[0, frame_idx].detach().cpu().numpy()
        baseline_frame = baseline_frame.transpose(1, 2, 0)  # CHW to HWC
        baseline_frame = (baseline_frame + 1) / 2  # Assuming tanh output
        baseline_frame = np.clip(baseline_frame, 0, 1)
        
        axes[1, i].imshow(baseline_frame)
        axes[1, i].set_title(f'Baseline Frame {frame_idx}', color='white')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_frames_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='black')
    print("  Saved: sample_frames_comparison.png")
    plt.close()


if __name__ == '__main__':
    import time
    
    # Run the comprehensive test
    start_time = time.time()
    lsd_metrics, baseline_metrics = test_error_accumulation(
        num_frames=300,  # Reduced for faster testing, increase for thorough evaluation
        save_results=True
    )
    total_time = time.time() - start_time
    
    print(f"\nTest completed in {total_time:.1f} seconds")
    print("\nKey files generated:")
    print("  - error_accumulation_analysis.png")  
    print("  - sample_frames_comparison.png")
    print("\nTest addresses the core Mirage challenge of preventing")
    print("autoregressive video models from getting 'stuck in loops'")
    print("and degrading to single colors over long sequences.")