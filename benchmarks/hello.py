import os
import time
import torch
from torch.utils.cpp_extension import load_inline

# Set CUDA arch to match RTX 3090 (compute_86/sm_86)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

# C++ sources: Declarations and PyBind11 bindings
cpp_sources = """
#include <torch/extension.h>

extern torch::Tensor fused_scale_add(torch::Tensor input, torch::Tensor noise, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scale_add", &fused_scale_add);
}
"""

# CUDA sources: Kernel and wrapper implementation
cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void ptx_fused_scale_add(float* input, float* noise, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float reg;
        asm volatile(
            "ld.global.f32 %0, [%1]; \\n\\t"  // Load input into %0
            "{ .reg .f32 temp; \\n\\t"
            "ld.global.f32 temp, [%2]; \\n\\t"  // Load noise into temp
            "fma.rn.f32 %0, %0, %3, temp; \\n\\t"  // Fused: %0 = %0 * scale + temp
            "} \\n\\t"
            "st.global.f32 [%4], %0; \\n\\t"  // Store %0 to output
            : "=f"(reg)
            : "l"(input + idx), "l"(noise + idx), "f"(scale), "l"(output + idx)
            : "memory"
        );
    }
}

torch::Tensor fused_scale_add(torch::Tensor input, torch::Tensor noise, float scale) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(noise.device().is_cuda(), "noise must be a CUDA tensor");
    TORCH_CHECK(input.sizes() == noise.sizes(), "input and noise must have the same size");

    auto output = torch::empty_like(input);
    int size = input.numel();
    dim3 blocks((size + 255) / 256);
    dim3 threads(256);
    ptx_fused_scale_add<<<blocks, threads>>>(
        input.data_ptr<float>(), noise.data_ptr<float>(), output.data_ptr<float>(), scale, size
    );
    return output;
}
"""

# Load the custom kernel inline
custom_kernel = load_inline(
    name='custom_ptx',
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    verbose=True
)

# Toy Autoregressive Model (simple linear for next-frame prediction, made larger)
class ToyARModel(torch.nn.Module):
    def __init__(self, frame_size):
        super().__init__()
        self.linear = torch.nn.Linear(frame_size, frame_size)  # Predict next from current

    def forward(self, prev_frame):
        # Flatten, predict, reshape back to "frame"
        flat = prev_frame.view(-1)
        pred = self.linear(flat)
        # Use custom PTX op: Add some "noise" and scale (simulates diffusion-like step)
        noise = torch.randn_like(pred, device='cuda')
        pred_optimized = custom_kernel.fused_scale_add(pred, noise, 0.5)  # PTX here!
        return pred_optimized.view_as(prev_frame)

# Generate sequence autoregressively, with incremental timing
def generate_video(num_frames=100, frame_shape=(3, 64, 64)):  # Larger: RGB 64x64 frames
    frame_size = frame_shape[0] * frame_shape[1] * frame_shape[2]
    model = ToyARModel(frame_size=frame_size).cuda()
    initial_frame = torch.randn(frame_shape, device='cuda')  # Start with random "frame"
    video = [initial_frame]
    
    total_start = time.time()
    for i in range(1, num_frames):
        frame_start = time.time()
        next_frame = model(video[-1])  # Autoregressive: Predict next from last
        video.append(next_frame)
        frame_time = time.time() - frame_start
        if i % 10 == 0:  # Print every 10 frames for incremental feedback
            print(f"Generated frame {i}/{num_frames-1} in {frame_time:.4f} seconds (on GPU: {next_frame.device})")
    
    total_time = time.time() - total_start
    print(f"Total time for {num_frames} frames: {total_time:.2f} seconds ({total_time / num_frames:.4f} s/frame)")
    return torch.stack(video)  # "Video" tensor: (num_frames, 3, 64, 64)

# Run it
if __name__ == '__main__':
    video = generate_video()
    print("Generated 'video' shape:", video.shape)
    # To "visualize" a sample frame (requires matplotlib and numpy; install if needed: pip install numpy matplotlib)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        frame_np = video[0].cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
        frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())  # Normalize for display
        plt.imshow(frame_np)
        plt.title("Sample Frame (Random Noise Pattern)")
        plt.show()
    except ImportError:
        print("Install numpy and matplotlib to visualize frames: pip install numpy matplotlib")
