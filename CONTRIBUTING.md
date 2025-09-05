# 🤝 Contributing to Mirage Hello

Thank you for your interest in contributing to **Mirage Hello**! This project is a community effort to build open-source real-time video diffusion, and we welcome contributors of all skill levels.

## 🎯 Quick Start

1. **Star & Fork** the repository
2. **Clone your fork**: `git clone git@github.com:yourusername/mirage-hello.git`
3. **Run the demo**: `python working_gpu_demo.py` (requires CUDA GPU)
4. **Pick an issue** labeled `good first issue` or `help wanted`
5. **Submit a PR** with your improvements

## 🚀 Areas Where We Need Help

### 🔥 **High Priority (Week 1-2 Focus)**

**Performance Optimization:**
- [ ] Mixed precision (FP16/BF16) implementation
- [ ] Flash Attention integration
- [ ] Memory usage optimization
- [ ] CPU-GPU transfer bottleneck elimination

**Infrastructure:**
- [ ] Automated benchmarking CI/CD
- [ ] Quality regression testing
- [ ] Documentation improvements
- [ ] Error handling and logging

### 💡 **Medium Priority**

**Model Improvements:**
- [ ] Architecture optimization (smaller, faster models)
- [ ] Advanced sampling methods (DDIM, DPM-Solver)
- [ ] Knowledge distillation
- [ ] Quantization support

**Features:**
- [ ] Web interface for demos
- [ ] API endpoints
- [ ] Docker containerization
- [ ] Multi-GPU support

### 🔬 **Research & Advanced**

**CUDA Development:**
- [ ] Custom CUDA kernels
- [ ] PTX assembly optimization
- [ ] Memory coalescing improvements
- [ ] Tensor Core utilization

**Novel Techniques:**
- [ ] New architectures
- [ ] Improved temporal consistency
- [ ] Better conditioning mechanisms
- [ ] Novel loss functions

## 🛠️ Development Setup

### Prerequisites
```bash
# GPU required for meaningful testing
nvidia-smi  # Should show your GPU

# Python 3.8+
python --version

# CUDA Toolkit (optional but recommended)
nvcc --version
```

### Installation
```bash
# Clone your fork
git clone git@github.com:yourusername/mirage-hello.git
cd mirage-hello

# Install dependencies
pip install torch torchvision pynvml psutil

# Verify installation
python working_gpu_demo.py --help
```

### Development Tools
```bash
# Install dev dependencies
pip install pytest black flake8 mypy

# Install CUDA profiling tools (optional)
pip install py-spy line_profiler memory_profiler
```

## 📊 Before You Start: Run Benchmarks

**Always benchmark before making changes** so you can measure impact:

```bash
# Run baseline benchmark (takes ~3 minutes)
python benchmark.py --all --duration 30

# This creates benchmark_YYYYMMDD_HHMMSS.json
# Keep this file to compare your improvements!
```

## 🔄 Development Workflow

### 1. **Choose Your Contribution**
- Check [Issues](https://github.com/toddllm/mirage-hello/issues) for tasks
- Look for `good first issue`, `help wanted`, or `optimization` labels
- Comment on the issue to avoid duplicate work

### 2. **Create a Branch**
```bash
git checkout -b feature/your-improvement-name
# or
git checkout -b fix/issue-number
```

### 3. **Make Changes**
- Follow our [Code Style](#code-style) guidelines
- Write tests for new functionality
- Keep changes focused and atomic

### 4. **Test Your Changes**
```bash
# Run the basic demo
python working_gpu_demo.py

# Run benchmarks to measure impact
python benchmark.py --all --duration 30

# Compare against your baseline
python benchmark.py --compare your_baseline.json
```

### 5. **Commit & Push**
```bash
git add .
git commit -m "optimization: implement mixed precision training

- Add FP16 support using torch.cuda.amp
- Reduces memory usage by 40% 
- Improves performance by 1.8x on RTX GPUs
- Maintains quality within 1% of FP32

Fixes #123"

git push origin feature/your-improvement-name
```

### 6. **Create Pull Request**
- Use our [PR Template](#pull-request-template)
- Include benchmark results
- Explain the impact of your changes
- Link to relevant issues

## 📝 Code Style

### Python Style
- **Black formatting**: `black .` before committing
- **Import order**: `isort .` for consistent imports
- **Type hints**: Add type annotations for new functions
- **Docstrings**: Use Google-style docstrings

```python
def optimize_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor,
    use_flash_attention: bool = True
) -> torch.Tensor:
    """Optimized multi-head attention computation.
    
    Args:
        query: Query tensor of shape (B, N, D)
        key: Key tensor of shape (B, N, D) 
        value: Value tensor of shape (B, N, D)
        use_flash_attention: Whether to use Flash Attention optimization
        
    Returns:
        Attention output of shape (B, N, D)
    """
```

### Performance Guidelines
- **Profile first**: Use profilers to identify bottlenecks
- **Measure impact**: Include before/after benchmarks in PRs
- **Memory conscious**: Minimize GPU memory usage
- **Batch-friendly**: Ensure operations work with different batch sizes

### GPU Code Guidelines
- **Error handling**: Check CUDA errors properly
- **Memory management**: Clean up GPU memory
- **Compatibility**: Support multiple GPU architectures
- **Fallbacks**: Provide CPU fallbacks for CUDA operations

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_models.py
pytest tests/test_performance.py

# Run with coverage
pytest --cov=. tests/
```

### Writing Tests
- **Unit tests** for individual functions
- **Integration tests** for model components
- **Benchmark tests** for performance regressions
- **Quality tests** for output validation

### Performance Testing
```bash
# Quick performance check (30s)
python benchmark.py --model heavy --duration 30

# Full benchmark suite (90s)  
python benchmark.py --all

# Regression testing
python benchmark.py --compare baseline.json
```

## 📋 Pull Request Template

When creating a PR, please include:

```markdown
## 🎯 What This Changes
Brief description of your improvement.

## 🚀 Performance Impact
- **Before**: X.X FPS, Y.Y GB memory
- **After**: X.X FPS, Y.Y GB memory  
- **Improvement**: Zx faster, W% less memory

## 🧪 Testing
- [ ] Ran `python working_gpu_demo.py` successfully
- [ ] Ran benchmark suite and compared results
- [ ] Added/updated tests as needed
- [ ] Verified no quality regression

## 📊 Benchmark Results
```
Paste benchmark comparison output here
```

## 🔗 Related Issues
Fixes #123, Related to #456

## ✅ Checklist
- [ ] Code follows style guidelines (ran `black .`)
- [ ] Added type hints and docstrings  
- [ ] Tests pass locally
- [ ] Performance benchmarks included
- [ ] Documentation updated if needed
```

## 🏷️ Issue Guidelines

### Reporting Bugs
```markdown
**Bug Description**: Brief description

**Environment**:
- GPU: RTX 3090 24GB
- CUDA: 12.1  
- PyTorch: 2.1.0
- Python: 3.11

**To Reproduce**:
1. Run `python working_gpu_demo.py`
2. Observe error message

**Expected vs Actual**:
Expected: X, Got: Y

**Additional Context**:
Any relevant logs or screenshots
```

### Requesting Features
```markdown
**Feature Request**: Brief description

**Problem**: What problem does this solve?

**Proposed Solution**: How should this work?

**Performance Impact**: Will this make things faster/slower?

**Implementation Ideas**: Technical approach (optional)
```

### Performance Issues
```markdown
**Performance Issue**: Brief description

**Current Performance**:
- Model: Heavy (192 channels)
- FPS: X.X 
- Memory: Y.Y GB
- GPU Utilization: Z%

**Expected Performance**: 
What performance should we achieve?

**Profiling Data**:
Any profiling information (optional)
```

## 🌟 Recognition

Contributors will be recognized in:
- **README Contributors Section**
- **Release Notes** for significant improvements
- **Performance Hall of Fame** for major optimizations
- **Special Thanks** in research papers (if we publish any)

## 💬 Getting Help

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Bug reports and feature requests  
- **Discord** (coming soon): Real-time help and collaboration

## 📜 Code of Conduct

### Our Standards
- **Be respectful** and inclusive
- **Help newcomers** get started
- **Share knowledge** and learn together
- **Focus on technical merit**
- **Give constructive feedback**

### Unacceptable Behavior
- Personal attacks or harassment
- Discriminatory language
- Spamming or trolling
- Sharing others' private information

## 🎊 First-Time Contributors

Special welcome to first-time contributors! Look for:

- **`good first issue`** - Small, well-defined tasks
- **`documentation`** - Improve docs and examples
- **`cleanup`** - Code organization and style improvements

Don't be afraid to ask questions - we're here to help! 🚀

---

**Ready to contribute? Pick an issue and let's make real-time video AI accessible to everyone!**