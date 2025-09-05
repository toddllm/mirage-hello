# ⚡ CUDA Development Guide

## 🎯 **Expert-Level Optimization for Real-Time Video Diffusion**

This guide covers advanced CUDA optimization techniques needed to match Mirage's PTX-level performance. If you're comfortable with GPU architecture and low-level optimization, this is where the real performance gains happen.

---

## 🔥 **Current Performance Bottlenecks**

### **Profiling Results (Production Scale)**
```
Model: 880M parameters, Batch Size: 2, Resolution: 64x64
Current Performance: 22.3 FPS (44.8ms per frame)
Mirage Target: 25 FPS (40ms per frame)
Gap: 1.12x too slow (need 12% speedup)

Memory Bandwidth Utilization: ~85% (memory-bound)
Compute Utilization: ~100% (saturated)
Bottleneck: Memory bandwidth + kernel launch overhead
```

**Why Custom CUDA is Needed:**
- PyTorch overhead: ~5-10ms per frame (significant for 40ms budget)
- Kernel fusion opportunities: 3-5 separate ops could be fused
- Memory access pattern optimization: 20-30% bandwidth improvement possible

---

## 🏗️ **CUDA Optimization Strategy**

### **Phase 1: Kernel Fusion (Week 4)**

**Target Operations for Fusion:**
1. **Conv + BatchNorm + Activation** (appears 15+ times in U-Net)
2. **Attention QKV Projection** (3 separate matmuls → 1 fused)  
3. **Diffusion Denoising Step** (multiple element-wise ops)
4. **Temporal Blending** (interpolation + consistency check)

**Expected Gains:**
- Kernel launch overhead: -3ms per frame
- Memory bandwidth: +25% efficiency  
- Total speedup: 1.3-1.5x

### **Phase 2: Memory Access Optimization (Week 5)**

**Optimization Targets:**
1. **Memory Coalescing**: Ensure 128-byte aligned accesses
2. **Shared Memory Usage**: Cache frequently accessed data
3. **Register Optimization**: Minimize register spilling  
4. **Occupancy Optimization**: Maximize warp utilization

**Expected Gains:**
- Memory bandwidth: +30% utilization
- Latency hiding: Better instruction overlap
- Total speedup: 1.2-1.4x (cumulative 1.8x)

### **Phase 3: PTX Assembly (Week 6+)**

**Hand-Optimization Targets:**
1. **Vectorized Operations**: float4 processing for 4x throughput
2. **Tensor Core Assembly**: Direct PTX for maximum utilization
3. **Warp-Level Primitives**: Custom reduce/scan operations
4. **Memory Layout Control**: Fine-grained data movement

**Expected Gains:**
- Instruction-level optimization: +40% throughput
- Perfect memory patterns: +50% bandwidth
- Total speedup: 2.0x+ (cumulative 3.6x → **11ms per frame**)

---

## 🛠️ **Implementation Roadmap**

### **Week 4: Custom CUDA Kernels**

#### **Day 1-2: Fused Convolution Kernel**
```cuda
// Target: Replace Conv2d + BatchNorm2d + SiLU chain
__global__ void fused_conv_bn_silu_kernel(
    const float* input,           // NCHW or NHWC input
    const float* weight,          // Convolution weights
    const float* bn_weight,       // BatchNorm gamma
    const float* bn_bias,         // BatchNorm beta  
    const float* bn_mean,         // BatchNorm running mean
    const float* bn_var,          // BatchNorm running variance
    float* output,                // Output tensor
    int batch_size, int channels, int height, int width,
    int out_channels, int kernel_size
) {
    // Fused computation:
    // 1. Convolution
    // 2. BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    // 3. SiLU activation: x * sigmoid(x)
    // All in one kernel launch
}
```

**Implementation Steps:**
1. Profile existing PyTorch conv+bn+silu chain
2. Write fused CUDA kernel
3. Add PyTorch C++ extension wrapper
4. Benchmark vs PyTorch implementation
5. Integration testing with full model

**Expected Gain:** 15-25% speedup on conv-heavy operations

#### **Day 3-4: Flash Attention Implementation**
```cuda
// Target: Memory-efficient attention with tiling
__global__ void flash_attention_kernel(
    const half* Q, const half* K, const half* V,  // FP16 for Tensor Cores
    half* O,                                      // Output
    float* l,                                     // Logsumexp  
    int batch_size, int num_heads, int seq_len, int head_dim,
    int block_size_q, int block_size_k           // Tiling parameters
) {
    // Tiled computation to fit in shared memory:
    // 1. Load Q, K, V tiles into shared memory
    // 2. Compute attention for tile
    // 3. Update output and statistics
    // 4. Move to next tile
}
```

**Implementation Steps:**
1. Study Flash Attention algorithm and tiling strategy
2. Implement basic tiled attention kernel  
3. Optimize for RTX 30/40 series shared memory (48-100KB)
4. Add multi-head and batch processing
5. Benchmark vs PyTorch SDPA Flash backend

**Expected Gain:** 30-50% attention speedup, 40% memory reduction

#### **Day 5-6: Diffusion Step Fusion**
```cuda  
// Target: Fuse multiple diffusion operations
__global__ void fused_diffusion_step_kernel(
    const half* x_noisy,          // Noisy input
    const half* noise_pred,       // Predicted noise
    const half* alpha_schedule,   // Diffusion schedule
    const half* prev_frame,       // Previous generated frame  
    half* output,                 // Denoised output
    float blend_weight,           // Temporal blending weight
    int numel                     // Number of elements
) {
    // Fused operations:
    // 1. Diffusion denoising: (x_noisy - sqrt_alpha_bar * noise_pred) / alpha
    // 2. Temporal blending: blend_weight * denoised + (1-weight) * prev_frame  
    // 3. Clamping/normalization if needed
}
```

**Expected Gain:** 20-30% speedup on diffusion sampling loop

#### **Day 7: Integration and Validation**
- Integrate all custom kernels into production model
- Comprehensive performance validation
- Memory usage analysis
- Quality assurance testing

**Week 4 Target:** 1.3-1.5x cumulative speedup → **30+ FPS on production model**

---

## 🔧 **Development Tools & Setup**

### **CUDA Development Environment**
```bash
# Install CUDA Toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# Verify installation
nvcc --version
```

### **PyTorch C++ Extension Setup**
```python
from torch.utils.cpp_extension import load_inline

# Inline compilation for rapid prototyping
cuda_kernel = load_inline(
    name='custom_kernel',
    cpp_sources=cpp_source_code,
    cuda_sources=cuda_source_code,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)
```

### **Profiling Tools**
```bash
# NVIDIA profilers (essential for optimization)
sudo apt-get install nvidia-nsight-systems nvidia-nsight-compute

# Profile Python script
nsys profile --force-overwrite=true -o profile python script.py

# Profile CUDA kernels 
ncu --set full -o kernel_profile python script.py
```

---

## 🎯 **Custom Kernel Development Process**

### **1. Identify Bottlenecks**
```bash
# Use PyTorch profiler
python -m torch.utils.bottleneck script.py

# Use NVIDIA Nsight Systems
nsys profile --force-overwrite=true -o analysis python script.py
nsys-ui analysis.qdrep  # Visual analysis
```

**Look for:**
- **High-frequency kernels** (called many times per frame)
- **Memory-bound operations** (low arithmetic intensity)
- **Kernel launch overhead** (many small kernels)

### **2. Implement Custom Kernel**

**Basic Template:**
```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_kernel(
    const float* input,
    float* output, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Your optimized computation here
        output[idx] = some_operation(input[idx]);
    }
}

torch::Tensor custom_operation(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    int size = input.numel();
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    custom_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_operation", &custom_operation);
}
```

### **3. Optimization Techniques**

**Memory Coalescing:**
```cuda
// Bad: Non-coalesced access
float val = input[idx * stride + offset];

// Good: Coalesced access  
float val = input[threadIdx.x + blockIdx.x * blockDim.x];
```

**Shared Memory Usage:**
```cuda
__shared__ float shared_data[256];

// Load collaboratively
if (threadIdx.x < data_size) {
    shared_data[threadIdx.x] = global_data[blockIdx.x * data_size + threadIdx.x];
}
__syncthreads();

// Use shared data for computation
```

**Tensor Core Utilization:**
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Use Tensor Core operations for matrix multiply
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;  
wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

wmma::load_matrix_sync(a_frag, a, 16);
wmma::load_matrix_sync(b_frag, b, 16);
wmma::fill_fragment(c_frag, 0.0f);

wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
```

---

## 📊 **Performance Measurement**

### **CUDA Event Timing** 
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
custom_kernel<<<grid, block>>>(args...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

### **Memory Bandwidth Analysis**
```python
def analyze_kernel_bandwidth(kernel_func, tensor_size, iterations=100):
    # Measure kernel execution time
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        result = kernel_func()
        torch.cuda.synchronize() 
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    
    # Calculate theoretical vs achieved bandwidth
    bytes_transferred = tensor_size * 4 * 2  # Input + output, FP32
    achieved_bandwidth = bytes_transferred / avg_time / 1e9  # GB/s
    
    rtx_3090_bandwidth = 936  # GB/s theoretical
    efficiency = achieved_bandwidth / rtx_3090_bandwidth * 100
    
    print(f"Kernel bandwidth: {achieved_bandwidth:.1f} GB/s ({efficiency:.1f}% of peak)")
```

---

## 🎯 **PTX Assembly (Advanced)**

### **When to Use PTX**
- **After CUDA optimization** - PTX is the final level of optimization
- **Critical hot paths** - Operations called thousands of times per frame  
- **Tensor Core utilization** - When you need perfect instruction scheduling
- **Vectorization** - float4/half8 operations for maximum throughput

### **PTX Development Process**
```cuda
// Example: Vectorized FP16 operation
asm volatile(
    "ld.global.v4.f16 {%0, %1, %2, %3}, [%4];     \\n\\t"  // Load 4 FP16 values
    "add.rn.f16 %0, %0, %5;                       \\n\\t"  // Add bias
    "add.rn.f16 %1, %1, %5;                       \\n\\t"
    "add.rn.f16 %2, %2, %5;                       \\n\\t"
    "add.rn.f16 %3, %3, %5;                       \\n\\t"
    "st.global.v4.f16 [%6], {%0, %1, %2, %3};     \\n\\t"  // Store 4 values
    : "=h"(out0), "=h"(out1), "=h"(out2), "=h"(out3)
    : "l"(input_ptr), "h"(bias_val), "l"(output_ptr)
    : "memory"
);
```

### **PTX Resources**
- [NVIDIA PTX ISA Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Tensor Core PTX Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions)
- [PTX Optimization Examples](examples/expert/ptx_examples/)

---

## 🚀 **Current Development Priorities**

### **High-Impact Kernels (Week 4 Focus)**

**1. Fused Convolution (Priority 1)**
```
Current: PyTorch Conv2d + BatchNorm + SiLU (3 kernel launches)
Target: Single fused kernel with optimal memory access
Expected: 20-30% speedup on conv-heavy operations (60% of model)
```

**2. Attention Optimization (Priority 2)**  
```
Current: PyTorch MultiheadAttention or basic SDPA
Target: Hand-optimized Flash Attention with register tiling
Expected: 40-60% attention speedup (25% of model computation)
```

**3. Diffusion Step Fusion (Priority 3)**
```
Current: Multiple element-wise PyTorch operations  
Target: Fused denoising + blending + normalization
Expected: 15-25% speedup on diffusion sampling
```

### **Development Workflow**

**1. Kernel Development**
```bash
# Create kernel in examples/expert/kernels/
# Test with: python examples/expert/test_kernel.py --kernel conv_fusion

# Template structure:
examples/expert/kernels/
├── conv_fusion.cu           # CUDA kernel implementation
├── conv_fusion.cpp          # PyTorch wrapper  
├── test_conv_fusion.py      # Standalone testing
└── benchmark_conv.py        # Performance validation
```

**2. Integration Testing**
```bash
# Test kernel in isolation
python examples/expert/test_kernel.py --kernel conv_fusion --validate

# Integration with full model
python examples/advanced/production_scale_cuda.py --use-custom-kernels
```

**3. Performance Validation**
```bash
# Benchmark against PyTorch baseline
python benchmarks/kernel_benchmark.py --kernel conv_fusion

# Full model comparison
python benchmarks/run_benchmarks.py --custom-kernels --compare baseline.json
```

---

## 🔬 **Kernel Development Examples**

### **Example 1: Element-wise Fusion**
```cuda
// Fuse multiple element-wise operations
__global__ void fused_elementwise_kernel(
    const half* a, const half* b, const half* c,
    half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Fuse: output = silu(a + b * c)
        half temp = __hadd(a[idx], __hmul(b[idx], c[idx]));
        half sigmoid_val = __hdiv(__float2half(1.0f), 
                                 __hadd(__float2half(1.0f), hexp(__hneg(temp))));
        output[idx] = __hmul(temp, sigmoid_val);
    }
}
```

### **Example 2: Shared Memory Optimization**
```cuda
__global__ void optimized_attention_kernel(
    const half* Q, const half* K, const half* V,
    half* O, int seq_len, int head_dim
) {
    __shared__ half shared_K[64 * 64];  // Tile size for RTX 3090
    __shared__ half shared_V[64 * 64];
    
    // Collaborative loading into shared memory
    int tid = threadIdx.x;
    if (tid < head_dim) {
        for (int i = 0; i < seq_len; i++) {
            shared_K[i * head_dim + tid] = K[blockIdx.x * seq_len * head_dim + i * head_dim + tid];
            shared_V[i * head_dim + tid] = V[blockIdx.x * seq_len * head_dim + i * head_dim + tid];
        }
    }
    __syncthreads();
    
    // Compute attention using shared memory (much faster)
    // ... attention computation using shared_K, shared_V
}
```

---

## 🎯 **Contributing CUDA Optimizations**

### **Getting Started**
1. **Pick a kernel** from the priority list above
2. **Profile the baseline** PyTorch implementation  
3. **Implement CUDA version** following templates
4. **Benchmark thoroughly** against baseline
5. **Submit PR** with performance comparison

### **Code Organization**
```
examples/expert/
├── kernels/                 # CUDA kernel implementations
│   ├── conv_fusion/         # Fused convolution kernels
│   ├── attention/           # Flash Attention implementations  
│   └── diffusion/           # Diffusion-specific kernels
├── profiling/               # Advanced profiling tools
└── assembly/                # PTX assembly implementations (future)
```

### **Contribution Guidelines**
- **Document performance gains** with before/after benchmarks
- **Include fallback** to PyTorch for compatibility
- **Add comprehensive tests** for correctness validation
- **Follow CUDA best practices** for memory access and occupancy

### **Quality Standards**
- **Correctness**: Pass all numerical accuracy tests
- **Performance**: Show measurable improvement over PyTorch
- **Compatibility**: Work across RTX 20/30/40 series GPUs  
- **Maintainability**: Clear code with documentation

---

## 📈 **Expected Timeline to Mirage Parity**

```
Week 4 (Custom CUDA):     30+ FPS  (custom kernels)
Week 5 (Memory Opt):      35+ FPS  (memory access optimization)  
Week 6 (PTX Assembly):    40+ FPS  (hand-optimized assembly)
Week 7+ (Novel Techniques): 50+ FPS  (surpass Mirage targets)
```

**Mirage Targets:**
- ✅ **Week 6**: Match 40ms current target  
- ✅ **Week 7+**: Exceed 16ms next-gen target

---

## 💡 **Advanced Optimization Ideas**

### **Novel Techniques to Explore**
1. **Persistent Kernels**: Keep kernels resident for ultra-low latency
2. **Multi-GPU Streaming**: Pipeline across multiple GPUs
3. **Temporal Caching**: Cache and reuse temporal features
4. **Adaptive Precision**: Dynamic FP16/INT8 based on content
5. **Hardware-Specific**: RTX 40 series Ada architecture optimizations

### **Research Opportunities**
- **New Attention Patterns**: Beyond Flash Attention efficiency
- **Diffusion Sampling**: Novel fast sampling techniques
- **Memory Hierarchies**: Optimal use of L1/L2/HBM
- **Compiler Integration**: MLIR/XLA custom passes

---

**🎯 Ready to write some blazing fast CUDA code? Pick a kernel and let's unlock that final performance tier!**

**Next:** Check [`examples/expert/kernels/`](../../examples/expert/kernels/) for implementation templates and current development.