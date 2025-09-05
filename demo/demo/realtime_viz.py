"""
Real-Time Visual Demo - Mirage Hello Optimization Showcase
Live GPU Stylizer with fast-path toggles

This is the visual "hello world" that demonstrates:
- Real-time webcam → GPU processing → display
- Visual FPS jumps when optimizations are enabled
- Side-by-side comparison of optimization impacts
- HUD showing exact performance metrics and settings

CLI Usage:
    python demo/realtime_viz.py --source webcam
    python demo/realtime_viz.py --source video.mp4 --encode output.mp4
    python demo/realtime_viz.py --res 512x288 --dtype fp16 --sdpa flash --graphs 1
"""

import torch
import cv2
import numpy as np
import argparse
import time
from typing import Optional
import threading
import queue

# Import our demo components
from model_tiny_viz import TinyVizModel
from gpu_io import GPUVideoProcessor, VideoCapture, VideoEncoder
from graph_wrap import GraphRunner, AdaptiveGraphRunner  
from hud import PerformanceHUD

# Try to get GPU stats
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False


def get_gpu_stats():
    """Get current GPU utilization"""
    if not NVML_AVAILABLE:
        return {'gpu_util': 0, 'memory_used': 0}
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'gpu_util': util.gpu,
            'memory_used': mem_info.used // 1024**2
        }
    except:
        return {'gpu_util': 0, 'memory_used': 0}


class RealtimeVizApp:
    """Main real-time visualization application"""
    
    def __init__(self, args):
        self.args = args
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Parse resolution
        if 'x' in args.res:
            width, height = map(int, args.res.split('x'))
            self.target_size = (width, height)
        else:
            self.target_size = (512, 288)
        
        print(f"🎬 MIRAGE HELLO - REAL-TIME VISUAL DEMO")
        print(f"=" * 60)
        print(f"Resolution: {self.target_size[0]}x{self.target_size[1]}")
        print(f"Source: {args.source}")
        
        # Initialize components
        self._init_model()
        self._init_video_pipeline()
        self._init_hud()
        
        # Performance comparison data
        self.comparison_results = {}
        
    def _init_model(self):
        """Initialize the tiny visualization model"""
        print(f"\n🧠 Initializing model...")
        
        self.model = TinyVizModel(
            base_channels=64,  # Small but Tensor Core friendly
            dtype=self.args.dtype,
            channels_last=self.args.channels_last == 1,
            sdpa_backend=self.args.sdpa
        )
        
        # Wrap with CUDA Graphs if enabled
        if self.args.graphs == 1:
            print(f"📸 Enabling CUDA Graphs...")
            
            # Create example input for graph capture
            example_input = torch.randn(
                1, 3, self.target_size[1], self.target_size[0],  # BCHW
                device='cuda',
                dtype=torch.float16 if self.args.dtype == 'fp16' else torch.float32
            )
            
            if self.args.channels_last == 1:
                example_input = example_input.contiguous(memory_format=torch.channels_last)
            
            self.graph_runner = GraphRunner(self.model.model, example_input)
            self.use_graphs = True
        else:
            self.use_graphs = False
            
        print(f"✅ Model ready with optimizations")
    
    def _init_video_pipeline(self):
        """Initialize video I/O pipeline"""
        print(f"\n📹 Initializing video pipeline...")
        
        # Video processor
        self.video_processor = GPUVideoProcessor(
            target_size=self.target_size,
            dtype=torch.float16 if self.args.dtype == 'fp16' else torch.float32,
            channels_last=self.args.channels_last == 1
        )
        
        # Video capture
        if self.args.source == 'webcam':
            self.capture = VideoCapture(source=0, target_fps=30)
        else:
            self.capture = VideoCapture(source=self.args.source)
        
        # Video encoder (if output specified)
        if self.args.encode:
            self.encoder = VideoEncoder(
                output_path=self.args.encode,
                fps=25,
                size=self.target_size
            )
            print(f"🎥 Video encoding to: {self.args.encode}")
        else:
            self.encoder = None
        
        print(f"✅ Video pipeline ready")
    
    def _init_hud(self):
        """Initialize HUD overlay"""
        self.hud = PerformanceHUD(target_fps=25)  # Mirage target
        
        # Settings for HUD display
        self.settings = {
            'dtype': self.args.dtype,
            'channels_last': bool(self.args.channels_last),
            'sdpa': self.args.sdpa,
            'graphs': bool(self.args.graphs),
            'temporal': bool(self.args.temporal)
        }
    
    def process_frame(self, input_frame, prev_output=None):
        """Process single frame through complete pipeline"""
        
        # GPU preprocessing
        frame_tensor = self.video_processor.preprocess_frame_gpu(input_frame)
        
        # Model inference
        if self.use_graphs:
            # CUDA Graphs path
            if self.args.temporal and prev_output is not None:
                # For temporal mode, we'd need separate graphs for different prev inputs
                # Simplified: just use current frame  
                output_tensor = self.graph_runner(frame_tensor)
            else:
                output_tensor = self.graph_runner(frame_tensor)
        else:
            # Standard PyTorch path
            output_tensor, perf = self.model.process_frame(
                frame_tensor, 
                prev_frame=prev_output if self.args.temporal else None
            )
        
        # GPU postprocessing
        output_frame = self.video_processor.postprocess_frame_gpu(output_tensor)
        
        return output_frame, output_tensor
    
    def run_performance_comparison(self):
        """Run quick performance comparison across settings"""
        
        print(f"\n📊 Running performance comparison...")
        
        # Test configurations
        test_configs = [
            {'name': 'FP32+Math', 'dtype': 'fp32', 'channels_last': 0, 'sdpa': 'math', 'graphs': 0},
            {'name': 'FP16', 'dtype': 'fp16', 'channels_last': 0, 'sdpa': 'math', 'graphs': 0},
            {'name': 'FP16+Flash', 'dtype': 'fp16', 'channels_last': 1, 'sdpa': 'flash', 'graphs': 0},
            {'name': 'FP16+Flash+Graphs', 'dtype': 'fp16', 'channels_last': 1, 'sdpa': 'flash', 'graphs': 1},
        ]
        
        # Get test frame
        test_frame = self.capture.read_frame()
        if test_frame is None:
            print("❌ No test frame available")
            return {}
        
        results = {}
        
        for config in test_configs:
            print(f"   Testing: {config['name']}")
            
            # Create model with config
            test_model = TinyVizModel(
                base_channels=64,
                dtype=config['dtype'],
                channels_last=config['channels_last'] == 1,
                sdpa_backend=config['sdpa']
            )
            
            # Setup CUDA Graphs if enabled
            if config['graphs'] == 1:
                example = self.video_processor.preprocess_frame_gpu(test_frame)
                graph_runner = GraphRunner(test_model.model, example, warmup_steps=3)
            
            # Benchmark
            times = []
            for _ in range(10):  # Quick test
                frame_tensor = self.video_processor.preprocess_frame_gpu(test_frame)
                
                start_time = time.time()
                if config['graphs'] == 1:
                    _ = graph_runner(frame_tensor)
                else:
                    _, _ = test_model.process_frame(frame_tensor)
                torch.cuda.synchronize()
                
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            results[config['name']] = fps
            
            print(f"      {fps:.1f} FPS")
            
            # Cleanup
            del test_model
            if config['graphs'] == 1:
                del graph_runner
            torch.cuda.empty_cache()
        
        self.comparison_results = results
        return results
    
    def run_live_demo(self):
        """Run the live real-time demo"""
        
        print(f"\n🎬 Starting live demo...")
        print(f"Controls:")
        print(f"   'q' - Quit")
        print(f"   'c' - Run comparison benchmark")
        print(f"   's' - Save current frame")
        print(f"   SPACE - Pause/resume")
        
        self.running = True
        frame_count = 0
        prev_output = None
        paused = False
        
        try:
            while self.running:
                # Read frame
                input_frame = self.capture.read_frame()
                if input_frame is None:
                    print("📹 No more frames")
                    break
                
                if not paused:
                    # Process frame
                    start_time = time.time()
                    output_frame, output_tensor = self.process_frame(input_frame, prev_output)
                    process_time = time.time() - start_time
                    
                    # Update temporal conditioning
                    if self.args.temporal:
                        prev_output = output_tensor
                    
                    # Get performance metrics
                    gpu_stats = get_gpu_stats()
                    performance = {
                        'fps': 1.0 / process_time if process_time > 0 else 0,
                        'memory_mb': gpu_stats['memory_used'],
                        'gpu_util': gpu_stats['gpu_util'],
                        'frame_time': process_time
                    }
                    
                    # Create side-by-side display
                    display_frame = self._create_display(input_frame, output_frame, performance)
                    
                    # Encode if requested
                    if self.encoder:
                        self.encoder.write_frame(display_frame)
                    
                    frame_count += 1
                    
                    if frame_count % 30 == 0:  # Print stats every 30 frames
                        stats = self.model.get_performance_stats()
                        print(f"Frame {frame_count}: {stats['avg_fps']:.1f} FPS avg, "
                              f"{performance['memory_mb']}MB, GPU {performance['gpu_util']}%")
                else:
                    display_frame = input_frame
                
                # Display
                cv2.imshow('Mirage Hello - Real-Time Demo', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("🔄 Running comparison...")
                    comparison = self.run_performance_comparison()
                    print("📊 Comparison results:")
                    for name, fps in comparison.items():
                        print(f"   {name}: {fps:.1f} FPS")
                elif key == ord('s'):
                    cv2.imwrite(f'demo_frame_{frame_count}.png', display_frame)
                    print(f"💾 Saved frame {frame_count}")
                elif key == ord(' '):
                    paused = not paused
                    print(f"⏸️ {'Paused' if paused else 'Resumed'}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Demo stopped by user")
        
        finally:
            self.cleanup()
        
        # Final performance report
        self._print_final_stats(frame_count)
    
    def _create_display(self, input_frame, output_frame, performance):
        """Create side-by-side display with HUD overlay"""
        
        # Resize frames to target size for display
        input_resized = cv2.resize(input_frame, self.target_size)
        output_resized = cv2.resize(output_frame, self.target_size)
        
        # Create side-by-side layout
        display_width = self.target_size[0] * 2 + 20  # Gap between frames
        display_height = self.target_size[1] + 100    # Extra space for HUD
        
        display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Place input and output frames
        display[10:10+self.target_size[1], 10:10+self.target_size[0]] = input_resized
        display[10:10+self.target_size[1], 20+self.target_size[0]:20+self.target_size[0]*2] = output_resized
        
        # Labels
        cv2.putText(display, "Input", (10, self.target_size[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Mirage Hello Output", (20 + self.target_size[0], self.target_size[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add HUD overlay
        display = self.hud.draw_overlay(display, self.settings, performance)
        
        # Add comparison results if available
        if self.comparison_results:
            display = self.hud.draw_comparison(display, self.comparison_results)
        
        return display
    
    def _print_final_stats(self, frame_count):
        """Print final performance statistics"""
        
        print(f"\n📊 DEMO COMPLETE - FINAL STATS")
        print(f"=" * 50)
        
        model_stats = self.model.get_performance_stats()
        io_stats = self.video_processor.get_io_stats()
        
        print(f"Frames processed: {frame_count}")
        print(f"Model performance:")
        print(f"   Average FPS: {model_stats['avg_fps']:.1f}")
        print(f"   Average memory: {model_stats['avg_memory']:.0f}MB")
        
        print(f"I/O performance:")  
        print(f"   I/O FPS: {io_stats['io_fps']:.1f}")
        print(f"   I/O time: {io_stats['avg_io_time']*1000:.1f}ms")
        
        # Mirage target assessment
        mirage_target = 25  # 40ms = 25 FPS
        if model_stats['avg_fps'] >= mirage_target:
            print(f"🎯 Mirage target: ✅ ACHIEVED ({model_stats['avg_fps']:.1f} ≥ {mirage_target} FPS)")
        else:
            print(f"🎯 Mirage target: ⚠️ {mirage_target / model_stats['avg_fps']:.1f}x slower than target")
        
        # Settings summary
        print(f"\nOptimization settings:")
        for setting, value in self.settings.items():
            print(f"   {setting}: {value}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        cv2.destroyAllWindows()
        
        if self.encoder:
            self.encoder.close()
            print(f"💾 Video saved to: {self.args.encode}")


def create_sample_video():
    """Create a sample video for testing when no webcam available"""
    
    print("🎨 Creating sample test video...")
    
    # Create animated pattern
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('demo/sample_test.mp4', fourcc, 25, (640, 480))
    
    for frame_num in range(150):  # 6 seconds at 25 FPS
        # Create animated test pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving sine wave pattern
        for y in range(480):
            for x in range(640):
                # Create RGB wave pattern
                r = int(127 * (1 + np.sin(0.02 * x + 0.1 * frame_num)))
                g = int(127 * (1 + np.cos(0.02 * y + 0.1 * frame_num)))
                b = int(127 * (1 + np.sin(0.02 * (x + y) + 0.15 * frame_num)))
                
                frame[y, x] = [b, g, r]  # BGR for OpenCV
        
        writer.write(frame)
    
    writer.release()
    print("✅ Sample video created: demo/sample_test.mp4")


def main():
    parser = argparse.ArgumentParser(description='Mirage Hello Real-Time Visual Demo')
    
    # Video source options
    parser.add_argument('--source', default='webcam', help='Video source: "webcam" or path to video file')
    parser.add_argument('--res', default='512x288', help='Target resolution (WxH)')
    
    # Optimization toggles
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16',
                       help='Model precision')
    parser.add_argument('--channels-last', type=int, choices=[0, 1], default=1,
                       help='Use channels_last memory format')
    parser.add_argument('--sdpa', choices=['math', 'flash', 'efficient'], default='flash',
                       help='SDPA backend for attention')
    parser.add_argument('--graphs', type=int, choices=[0, 1], default=1,
                       help='Enable CUDA Graphs')
    parser.add_argument('--temporal', type=int, choices=[0, 1], default=0,
                       help='Enable temporal conditioning (autoregressive)')
    
    # Output options
    parser.add_argument('--encode', type=str, help='Output video path for encoding')
    parser.add_argument('--comparison-only', action='store_true',
                       help='Run comparison benchmark only (no live demo)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - real-time demo requires GPU")
        return
    
    # Create sample video if needed
    if args.source != 'webcam' and not os.path.exists(args.source):
        if args.source == 'demo/sample_test.mp4':
            create_sample_video()
        else:
            print(f"❌ Video file not found: {args.source}")
            return
    
    try:
        # Initialize and run demo
        app = RealtimeVizApp(args)
        
        if args.comparison_only:
            # Just run comparison
            results = app.run_performance_comparison()
            print("\n📊 Performance Comparison Results:")
            for name, fps in results.items():
                print(f"   {name}: {fps:.1f} FPS")
        else:
            # Run full live demo
            app.run_live_demo()
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import os
    main()