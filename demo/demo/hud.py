"""
HUD Overlay for Real-Time Performance Monitoring
Shows optimization settings and performance metrics in real-time

Features:
- Real-time FPS, memory usage, GPU utilization
- Optimization settings display (dtype, backend, etc.)
- Visual performance comparison bars
- Color-coded status indicators
"""

import cv2
import numpy as np
import time
from collections import deque


class PerformanceHUD:
    """Real-time performance overlay for video demo"""
    
    def __init__(self, target_fps=25):
        self.target_fps = target_fps
        self.fps_history = deque(maxlen=30)  # 1-second average at 30 FPS
        self.memory_history = deque(maxlen=30)
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'good': (0, 255, 0),      # Green
            'warning': (0, 255, 255),  # Yellow  
            'bad': (0, 0, 255),        # Red
            'text': (255, 255, 255),   # White
            'bg': (0, 0, 0),          # Black
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def update(self, fps, memory_mb, gpu_util=0):
        """Update performance metrics"""
        self.fps_history.append(fps)
        self.memory_history.append(memory_mb)
        
    def get_avg_fps(self):
        """Get smoothed FPS average"""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
    def get_avg_memory(self):
        """Get smoothed memory average"""
        return sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
    
    def get_performance_color(self, fps):
        """Get color based on performance vs targets"""
        if fps >= self.target_fps:
            return self.colors['good']
        elif fps >= self.target_fps * 0.8:
            return self.colors['warning'] 
        else:
            return self.colors['bad']
    
    def draw_overlay(self, frame, settings, performance):
        """Draw complete HUD overlay on frame"""
        
        # Get current metrics
        current_fps = performance.get('fps', 0)
        avg_fps = self.get_avg_fps()
        memory_mb = performance.get('memory_mb', 0)
        
        self.update(current_fps, memory_mb, performance.get('gpu_util', 0))
        
        # Overlay background
        overlay = frame.copy()
        
        # Top-left: Performance metrics
        y_offset = 30
        
        # FPS display
        fps_color = self.get_performance_color(avg_fps)
        fps_text = f"FPS: {avg_fps:.1f} (target: {self.target_fps})"
        cv2.putText(overlay, fps_text, (10, y_offset), self.font, self.font_scale, fps_color, self.thickness)
        y_offset += 25
        
        # Memory display  
        memory_text = f"GPU Memory: {memory_mb:.0f}MB"
        cv2.putText(overlay, memory_text, (10, y_offset), self.font, self.font_scale, self.colors['text'], self.thickness)
        y_offset += 25
        
        # GPU utilization (if available)
        if 'gpu_util' in performance:
            gpu_text = f"GPU Util: {performance['gpu_util']}%"
            gpu_color = self.colors['good'] if performance['gpu_util'] > 70 else self.colors['warning']
            cv2.putText(overlay, gpu_text, (10, y_offset), self.font, self.font_scale, gpu_color, self.thickness)
            y_offset += 35
        
        # Optimization settings
        cv2.putText(overlay, "Optimizations:", (10, y_offset), self.font, self.font_scale, self.colors['text'], self.thickness)
        y_offset += 20
        
        # Settings display
        for setting, value in settings.items():
            color = self.colors['good'] if value in [True, 'fp16', 'bf16', 'flash'] else self.colors['warning']
            setting_text = f"  {setting}: {value}"
            cv2.putText(overlay, setting_text, (10, y_offset), self.font, 0.5, color, 1)
            y_offset += 18
        
        # Performance bar (top-right)
        bar_width = 200
        bar_height = 20
        bar_x = frame.shape[1] - bar_width - 10
        bar_y = 10
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), self.colors['bg'], -1)
        
        # Performance bar fill
        fill_ratio = min(1.0, avg_fps / (self.target_fps * 1.5))  # Show up to 1.5x target
        fill_width = int(bar_width * fill_ratio)
        bar_color = self.get_performance_color(avg_fps)
        
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
        
        # Bar labels
        cv2.putText(overlay, f"{avg_fps:.0f} FPS", (bar_x, bar_y - 5), self.font, 0.5, self.colors['text'], 1)
        
        # Mirage target indicator
        mirage_x = bar_x + int(bar_width * (25 / (self.target_fps * 1.5)))
        cv2.line(overlay, (mirage_x, bar_y), (mirage_x, bar_y + bar_height), (255, 0, 255), 2)  # Magenta line
        cv2.putText(overlay, "Mirage", (mirage_x - 20, bar_y + bar_height + 15), self.font, 0.4, (255, 0, 255), 1)
        
        return overlay
    
    def draw_comparison(self, frame, results_dict):
        """Draw comparison of different optimization settings"""
        
        # Bottom section: optimization comparison
        y_start = frame.shape[0] - 100
        
        # Background
        cv2.rectangle(frame, (10, y_start), (frame.shape[1] - 10, frame.shape[0] - 10), 
                     (0, 0, 0, 128), -1)
        
        # Title
        cv2.putText(frame, "Optimization Comparison:", (15, y_start + 20), 
                   self.font, 0.6, self.colors['text'], self.thickness)
        
        # Results table
        x_offset = 15
        y_offset = y_start + 45
        
        for config_name, fps in results_dict.items():
            color = self.get_performance_color(fps)
            result_text = f"{config_name}: {fps:.1f} FPS"
            cv2.putText(frame, result_text, (x_offset, y_offset), self.font, 0.5, color, 1)
            y_offset += 18
        
        return frame


def test_hud():
    """Test HUD overlay functionality"""
    
    print("🖥️ Testing HUD overlay...")
    
    # Create test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Create HUD
    hud = PerformanceHUD(target_fps=25)
    
    # Test different performance scenarios
    test_scenarios = [
        {'fps': 15, 'memory_mb': 4500, 'gpu_util': 45, 'name': 'Baseline'},
        {'fps': 28, 'memory_mb': 2200, 'gpu_util': 85, 'name': 'Optimized'},
        {'fps': 42, 'memory_mb': 1800, 'gpu_util': 95, 'name': 'Target Exceeded'},
    ]
    
    settings = {
        'dtype': 'fp16',
        'channels_last': True,
        'sdpa': 'flash',
        'graphs': True,
        'temporal': False
    }
    
    for i, scenario in enumerate(test_scenarios):
        test_frame = frame.copy()
        
        # Draw HUD
        hud_frame = hud.draw_overlay(test_frame, settings, scenario)
        
        # Save test image
        cv2.imwrite(f'hud_test_{scenario["name"].lower()}.png', hud_frame)
        print(f"   Saved: hud_test_{scenario['name'].lower()}.png")
    
    # Test comparison display
    comparison_results = {
        'FP32+Math': 18.5,
        'FP16': 24.2, 
        'FP16+Flash': 28.7,
        'FP16+Flash+Graphs': 32.1
    }
    
    comparison_frame = frame.copy()
    comparison_frame = hud.draw_comparison(comparison_frame, comparison_results)
    cv2.imwrite('hud_comparison_test.png', comparison_frame)
    print(f"   Saved: hud_comparison_test.png")
    
    print("✅ HUD test complete!")


if __name__ == '__main__':
    test_hud()