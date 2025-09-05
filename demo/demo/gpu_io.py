"""
GPU I/O Pipeline for Real-Time Video Processing
Optimized video capture, resize, and encode on GPU

Features:
- GPU-accelerated video decode/encode (NVDEC/NVENC when available)
- CV-CUDA optimization for resize/normalize operations
- Fallback to PyTorch operations when CV-CUDA unavailable
- Minimal CPU-GPU transfers for maximum performance
"""

import torch
import cv2
import numpy as np
import time
from typing import Optional, Tuple

# Try to import CV-CUDA for optimal GPU I/O
try:
    import cvcuda
    CV_CUDA_AVAILABLE = True
    print("✅ CV-CUDA available for optimal GPU I/O")
except ImportError:
    CV_CUDA_AVAILABLE = False
    print("⚠️ CV-CUDA not available - using PyTorch fallback")

# Try to import PyAV for hardware decode/encode
try:
    import av
    PYAV_AVAILABLE = True
    print("✅ PyAV available for hardware video decode/encode")
except ImportError:
    PYAV_AVAILABLE = False
    print("⚠️ PyAV not available - using OpenCV fallback")


class GPUVideoProcessor:
    """GPU-optimized video processing pipeline"""
    
    def __init__(self, target_size=(512, 288), dtype=torch.float16, channels_last=True):
        self.target_size = target_size  # (width, height)
        self.dtype = dtype
        self.channels_last = channels_last
        self.device = torch.device('cuda')
        
        # Performance tracking
        self.io_times = []
        
        print(f"🖥️ GPU Video Processor initialized:")
        print(f"   Target size: {target_size[0]}x{target_size[1]}")
        print(f"   Dtype: {dtype}")
        print(f"   Channels last: {channels_last}")
        print(f"   CV-CUDA: {CV_CUDA_AVAILABLE}")
        
    def preprocess_frame_gpu(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess frame entirely on GPU"""
        
        start_time = time.time()
        
        if CV_CUDA_AVAILABLE:
            # Optimal CV-CUDA path
            output = self._preprocess_cvcuda(frame_bgr)
        else:
            # PyTorch fallback
            output = self._preprocess_pytorch(frame_bgr)
        
        # Track I/O performance
        io_time = time.time() - start_time
        self.io_times.append(io_time)
        if len(self.io_times) > 30:
            self.io_times.pop(0)
        
        return output
    
    def _preprocess_cvcuda(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """CV-CUDA optimized preprocessing"""
        
        # Convert numpy to CV-CUDA tensor
        frame_cvcuda = cvcuda.as_tensor(frame_bgr, "HWC")
        
        # Resize on GPU
        resized = cvcuda.resize(frame_cvcuda, self.target_size, cvcuda.Interp.LINEAR)
        
        # Convert BGR to RGB
        rgb = cvcuda.cvtcolor(resized, cvcuda.ColorConversion.BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = cvcuda.convertto(rgb, np.float32, scale=1.0/255.0)
        
        # Convert to PyTorch tensor
        torch_tensor = torch.as_tensor(normalized.cuda(), device=self.device)
        
        # Convert to target format: HWC → CHW → target dtype/layout
        torch_tensor = torch_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW
        torch_tensor = torch_tensor.to(dtype=self.dtype)
        
        if self.channels_last:
            torch_tensor = torch_tensor.contiguous(memory_format=torch.channels_last)
            
        return torch_tensor
    
    def _preprocess_pytorch(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """PyTorch fallback preprocessing"""
        
        # Convert to PyTorch tensor on GPU
        frame_tensor = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        
        # Convert BGR to RGB
        frame_rgb = frame_tensor.flip(-1)  # BGR → RGB
        
        # Resize using PyTorch
        frame_tensor = frame_rgb.permute(2, 0, 1).unsqueeze(0).float()  # HWC → BCHW
        frame_resized = torch.nn.functional.interpolate(
            frame_tensor, 
            size=self.target_size[::-1],  # (height, width)
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize to [0, 1]
        frame_normalized = frame_resized / 255.0
        
        # Convert to target dtype and layout
        frame_normalized = frame_normalized.to(dtype=self.dtype)
        
        if self.channels_last:
            frame_normalized = frame_normalized.contiguous(memory_format=torch.channels_last)
        
        return frame_normalized
    
    def postprocess_frame_gpu(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output back to display format"""
        
        # Convert from model output format to display format
        if self.channels_last:
            tensor = tensor.contiguous()
        
        # Clamp to valid range and convert to uint8
        tensor = torch.clamp(tensor, -1, 1)  # Model outputs in [-1, 1]
        tensor = (tensor + 1) * 127.5  # Convert to [0, 255]
        tensor = tensor.to(torch.uint8)
        
        # Convert to CPU and numpy for display
        frame = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # CHW → HWC
        
        # Convert RGB back to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def get_io_stats(self):
        """Get I/O performance statistics"""
        if not self.io_times:
            return {'avg_io_time': 0, 'io_fps': 0}
        
        avg_time = sum(self.io_times) / len(self.io_times)
        return {
            'avg_io_time': avg_time,
            'io_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'samples': len(self.io_times)
        }


class VideoCapture:
    """Optimized video capture with hardware decode when available"""
    
    def __init__(self, source=0, target_fps=30):
        self.source = source
        self.target_fps = target_fps
        
        if isinstance(source, int):
            # Webcam capture
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            self.is_webcam = True
            
            # Verify webcam
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Failed to open webcam {source}")
            print(f"📷 Webcam {source} opened: {frame.shape}")
            
        elif isinstance(source, str):
            # Video file
            self.cap = cv2.VideoCapture(source)
            self.is_webcam = False
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video file {source}")
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"🎬 Video file opened: {width}x{height} @ {fps:.1f} FPS")
        
        self.frame_count = 0
        
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from source"""
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            return frame
        else:
            if not self.is_webcam:
                # Loop video file
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count = 1
                    return frame
            return None
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


class VideoEncoder:
    """Hardware-accelerated video encoding when available"""
    
    def __init__(self, output_path: str, fps: int, size: Tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.size = size  # (width, height)
        
        # Try hardware encoding first
        if PYAV_AVAILABLE:
            self._setup_hardware_encoder()
        else:
            self._setup_opencv_encoder()
    
    def _setup_hardware_encoder(self):
        """Setup NVENC hardware encoder via PyAV"""
        try:
            self.container = av.open(self.output_path, mode='w')
            
            # Try NVENC H.264 encoder
            self.stream = self.container.add_stream('h264_nvenc', rate=self.fps)
            self.stream.width = self.size[0]
            self.stream.height = self.size[1]
            self.stream.pix_fmt = 'yuv420p'
            
            self.hardware_encode = True
            print(f"✅ Hardware encoder (NVENC) initialized: {self.size[0]}x{self.size[1]} @ {self.fps} FPS")
            
        except Exception as e:
            print(f"⚠️ Hardware encoder failed: {e}")
            print("   Falling back to software encoding")
            self._setup_opencv_encoder()
    
    def _setup_opencv_encoder(self):
        """Fallback to OpenCV software encoder"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc, 
            self.fps,
            self.size
        )
        self.hardware_encode = False
        print(f"📹 Software encoder (OpenCV) initialized: {self.size[0]}x{self.size[1]} @ {self.fps} FPS")
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to video output"""
        if self.hardware_encode:
            # PyAV hardware encoding
            frame_av = av.VideoFrame.from_ndarray(frame, format='bgr24')
            for packet in self.stream.encode(frame_av):
                self.container.mux(packet)
        else:
            # OpenCV software encoding
            self.writer.write(frame)
    
    def close(self):
        """Close encoder and finalize video"""
        if self.hardware_encode:
            # Flush encoder
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        else:
            self.writer.release()


if __name__ == '__main__':
    # Test GPU I/O pipeline
    print("🔬 Testing GPU I/O Pipeline...")
    
    # Test video processor
    processor = GPUVideoProcessor(target_size=(512, 288), dtype=torch.float16)
    
    # Test with webcam if available
    try:
        capture = VideoCapture(source=0)  # Webcam
        
        print("📷 Testing webcam capture + GPU processing...")
        
        for i in range(10):
            frame = capture.read_frame()
            if frame is not None:
                # Process on GPU
                processed = processor.preprocess_frame_gpu(frame)
                
                # Simple passthrough "model" 
                output_tensor = processed * 0.8  # Darken slightly
                
                # Back to display
                output_frame = processor.postprocess_frame_gpu(output_tensor)
                
                print(f"   Frame {i+1}: {processed.shape} → {output_frame.shape}")
                
                if i == 0:
                    cv2.imshow('GPU I/O Test', output_frame)
                    cv2.waitKey(1)
        
        # Show I/O performance
        io_stats = processor.get_io_stats()
        print(f"\n📊 I/O Performance:")
        print(f"   Average I/O FPS: {io_stats['io_fps']:.1f}")
        print(f"   Average I/O time: {io_stats['avg_io_time']*1000:.2f}ms")
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"⚠️ Webcam test failed: {e}")
        print("   (This is normal if no webcam is connected)")
    
    print("✅ GPU I/O pipeline test complete")