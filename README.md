# TinyML Implementations - Edge AI & IoT Project

TinyML project for deploying machine learning models on ultra-low-power microcontrollers with comprehensive compression, quantization, and deployment pipelines.

## ⚠️ DISCLAIMER

**This is a research and educational demonstration only.**

This TinyML implementation is NOT intended for safety-critical applications, production deployment, or any use where model accuracy and reliability are critical. Use at your own risk.

## Project Overview

This project focuses on **model efficiency** through compression, pruning, quantization-aware training, and distillation for microcontroller deployment. It provides a complete pipeline from training to deployment on edge devices.

### Key Features

- **Multiple Model Architectures**: Baseline, Tiny, Pruned, and Quantized models
- **Comprehensive Compression**: Post-training quantization (PTQ), quantization-aware training (QAT), magnitude-based pruning
- **Multi-format Export**: TensorFlow Lite (float32/int8), ONNX for various edge runtimes
- **Performance Benchmarking**: Latency, throughput, and memory usage analysis
- **Interactive Demo**: Streamlit-based web interface for model comparison and live testing
- **Reproducible Research**: Deterministic seeding, comprehensive logging, and structured evaluation

## Project Structure

```
0767_TinyML_Implementations/
├── src/                          # Source code
│   ├── __init__.py
│   ├── models.py                 # Model definitions and training
│   ├── export.py                 # Export and deployment pipeline
│   └── utils.py                  # Utilities and device management
├── scripts/                      # Training and evaluation scripts
│   └── train.py                  # Main training pipeline
├── configs/                      # Configuration files
│   └── default.yaml              # Default configuration
├── demo/                         # Interactive demo application
│   └── app.py                    # Streamlit demo
├── assets/                       # Generated models and results
│   ├── models/                   # Exported models
│   └── results/                  # Training results and metrics
├── tests/                        # Unit tests
├── data/                         # Dataset storage
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── DISCLAIMER.md                 # Safety disclaimer
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/TinyML-Implementations.git
cd TinyML-Implementations

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Models

Train different model types:

```bash
# Train baseline model
python scripts/train.py --model-type baseline --epochs 10 --export --benchmark

# Train tiny model optimized for microcontrollers
python scripts/train.py --model-type tiny --epochs 10 --export --benchmark

# Train pruned model with magnitude-based pruning
python scripts/train.py --model-type pruned --epochs 10 --export --benchmark

# Train quantized model
python scripts/train.py --model-type quantized --epochs 10 --export --benchmark
```

### 3. Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- Model training interface
- Performance comparison dashboard
- Live digit classification demo
- Performance analysis and benchmarking

## Model Types

### 1. Baseline Model
- **Architecture**: 2 Conv2D layers (8, 16 filters) + 2 Dense layers (32, 10 units)
- **Use Case**: Reference model for comparison
- **Size**: ~0.5 MB

### 2. Tiny Model
- **Architecture**: 1 Conv2D layer (4 filters) + 2 Dense layers (8, 10 units)
- **Use Case**: Ultra-low-power microcontrollers
- **Size**: ~0.1 MB

### 3. Pruned Model
- **Technique**: Magnitude-based pruning with polynomial decay
- **Sparsity**: 50% final sparsity
- **Use Case**: Memory-constrained devices

### 4. Quantized Model
- **Technique**: Quantization-aware training
- **Precision**: INT8 quantization
- **Use Case**: Inference acceleration

## Export Formats

### TensorFlow Lite
- **Float32**: Full precision for accuracy-critical applications
- **Int8**: Quantized for memory and compute efficiency
- **Target**: Microcontrollers, mobile devices

### ONNX
- **Format**: Open Neural Network Exchange
- **Target**: Cross-platform edge runtimes (ONNX Runtime, OpenVINO, etc.)

## Performance Metrics

The project evaluates models across multiple dimensions:

### Accuracy Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision (weighted average)
- **Recall**: Per-class recall (weighted average)
- **F1 Score**: Harmonic mean of precision and recall

### Efficiency Metrics
- **Model Size**: Memory footprint in MB
- **Inference Latency**: P50, P95, mean latency in milliseconds
- **Throughput**: Frames per second (FPS)
- **Memory Usage**: Peak RAM consumption

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
model:
  input_shape: [28, 28, 1]
  num_classes: 10
  seed: 42

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001

export:
  formats: ["tflite_float32", "tflite_int8", "onnx"]
  quantization_types: ["float32", "int8"]
```

## Device Support

### Development Platforms
- **CPU**: Intel/AMD x86_64, ARM64
- **GPU**: NVIDIA CUDA (optional)
- **Apple Silicon**: M1/M2 with MPS support

### Edge Deployment Targets
- **Microcontrollers**: Arduino Nano 33 BLE, ESP32
- **Single Board Computers**: Raspberry Pi, Jetson Nano
- **Mobile**: Android (TFLite), iOS (CoreML)
- **Edge Servers**: Intel NUC, ARM-based edge devices

## Advanced Features

### Deterministic Training
- Reproducible results across runs
- Seeded random number generators
- Deterministic GPU operations

### Model Optimization
- Automatic mixed precision training
- Gradient clipping and normalization
- Learning rate scheduling

### Export Pipeline
- Representative dataset calibration
- Model validation and testing
- Performance benchmarking

## Evaluation and Benchmarking

### Automated Evaluation
```bash
# Run comprehensive evaluation
python scripts/train.py --model-type baseline --export --benchmark
```

### Performance Comparison
The demo provides interactive comparison of:
- Model accuracy vs. size trade-offs
- Latency vs. throughput analysis
- Compression ratio effectiveness

## Development

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- Black code formatting
- Ruff linting

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Limitations and Considerations

### Model Limitations
- **Dataset**: MNIST only (28x28 grayscale digits)
- **Task**: Classification only
- **Architecture**: Simple CNN architectures

### Deployment Considerations
- **Accuracy**: May not meet production requirements
- **Robustness**: Limited to clean, controlled environments
- **Scalability**: Not optimized for large-scale deployment

### Hardware Requirements
- **Training**: 4GB+ RAM, GPU recommended
- **Inference**: 1GB+ RAM for desktop, 256KB+ for microcontrollers
- **Storage**: 1GB+ for models and dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes only. See DISCLAIMER.md for important safety information.

## Acknowledgments

- TensorFlow Lite team for TinyML framework
- MNIST dataset creators
- Open source ML optimization libraries
- Edge AI research community

## References

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [TinyML Book](https://tinymlbook.com/)
- [MLPerf Tiny Benchmark](https://mlcommons.org/en/inference-tiny/)

---

**Remember**: This is a research demonstration. Do not use for safety-critical applications.
# TinyML-Implementations
