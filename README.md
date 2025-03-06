# ResolutionSuggester

[![Python application](https://github.com/RomanVPX/resolution-suggester/actions/workflows/python-app.yml/badge.svg)](https://github.com/RomanVPX/resolution-suggester/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

A powerful tool for analyzing texture quality at different resolutions and suggesting optimal downsizing parameters based on perceptual metrics.

## 🚀 Features

- **Comprehensive Analysis**: Analyze image quality at various resolutions using multiple metrics
- **Multiple Metrics**: PSNR, SSIM, MS-SSIM, TDPR, and LPIPS support
- **Optimal Resolution Suggestion**: Get recommendations for the best resolution based on quality thresholds
- **Interpolation Methods**: Choose between Bilinear, Bicubic, and Mitchell-Netravali interpolations
- **Per-Channel Analysis**: Examine quality metrics for individual RGB/RGBA channels
- **Visualization**: Generate charts showing quality vs. resolution
- **ML-Powered**: Fast quality estimation using machine learning models
- **Export Options**: Save results to CSV and JSON formats
- **Multilingual**: Supports English and Russian interfaces

## 📋 Requirements

- Python 3.10+
- NumPy, OpenCV, Pillow for image processing
- PyTorch (optional, for GPU acceleration and LPIPS)
- Matplotlib for visualization
- Scikit-learn for ML features

## 🔧 Installation

### From Source

```bash
git clone https://github.com/RomanVPX/ResolutionSuggester.git
cd ResolutionSuggester
pip install .
```

For development installation:

```bash
pip install -e ".[dev]"
```

## 💻 Usage

ResolutionSuggester provides two command-line interfaces:

- `resolution_suggester` - main command
- `res-suggest` - shorthand alias

### Basic Usage

```bash
res-suggest /path/to/image.png
```

### Analyze Multiple Images

```bash
res-suggest /path/to/textures/folder/
```

### Select Quality Metric

```bash
# Choose from: psnr, ssim, ms_ssim, tdpr, lpips
res-suggest image.png -m ssim
```

### Choose LPIPS Neural Network

```bash
# Choose from: alex (default/balanced), vgg (memory-hungry), squeeze (fast)
res-suggest image.png -m lpips --lpips-net vgg
```

### Choose Interpolation Method

```bash
# Choose from: bilinear, bicubic, mitchell (default)
res-suggest image.png -i bilinear
```

### Per-Channel Analysis

```bash
res-suggest image.png -c
```

### Generate Visualization Chart

```bash
res-suggest image.png --chart
```

### Switch Theme for Charts

```bash
# Choose from: dark (default), light
res-suggest image.png --chart --theme light
```

### Export Results

```bash
res-suggest image.png -o # .csv
res-suggest image.png -j # .json
```

### Use ML Model for Fast Estimation

```bash
res-suggest image.png --ml
```

### Set Minimum Analysis Size

```bash
# Default and minimum is 16
res-suggest image.png --min-size 32
```

### Control Parallel Processing

```bash
# Disable parallel processing
res-suggest image.png --no-parallel

# Set specific number of threads
res-suggest image.png -t 4
```

### Save Generated Images

```bash
# Save downscaled images
res-suggest image.png --save-im-down

# Save upscaled images after downscaling
res-suggest image.png --save-im-up

# Save all generated images
res-suggest image.png -s
```

### Switch Language

```bash
# Choose from: en, ru, auto (default)
res-suggest image.png --lang ru
```

## 📊 Example Output

```bash
╭───── Analysis (MS_SSIM) ─────╮
│ LevelLight-01_comp_light.exr │
╰──────────────────────────────╯
┏━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Resolution ┃ R    ┃ G    ┃ B    ┃ A    ┃ Min  ┃ Quality                 ┃ Quality Bar     ┃
┡━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 1024x1024  │ 1.00 │ 1.00 │ 1.00 │ 1.00 │ 1.00 │ Original                │                 │
│ 512x512    │ 1.00 │ 1.00 │ 1.00 │ 0.99 │ 0.99 │ Excellent quality       │ ██████████████░ │
│ 256x256    │ 0.97 │ 0.98 │ 0.97 │ 0.96 │ 0.96 │ Very good quality       │ ██████████████░ │
│ 128x128    │ 0.91 │ 0.93 │ 0.90 │ 0.87 │ 0.87 │ Noticeable quality loss │ █████████████░░ │
│ 64x64      │ 0.82 │ 0.86 │ 0.80 │ 0.76 │ 0.76 │ Noticeable quality loss │ ███████████░░░░ │
│ 32x32      │ 0.75 │ 0.80 │ 0.71 │ 0.64 │ 0.64 │ Noticeable quality loss │ █████████░░░░░░ │
│ 16x16      │ 0.73 │ 0.78 │ 0.67 │ 0.59 │ 0.59 │ Noticeable quality loss │ ████████░░░░░░░ │
└────────────┴──────┴──────┴──────┴──────┴──────┴─────────────────────────┴─────────────────┘
```

## 🛠️ Advanced Features

### Generate ML Training Dataset

```bash
res-suggest --generate-dataset /path/to/training/images/
```

### Train ML Model After Dataset Generation

```bash
res-suggest --generate-dataset /path/to/training/images/ --train-ml
```

### Compare Real and ML-Predicted Results

```bash
res-suggest image.png --compare-ml
```

### GPU Acceleration Control

```bash
# Disable GPU acceleration
res-suggest image.png --no-gpu
```

## 🔧 Supported File Formats

- EXR (.exr)
- TGA (.tga)
- PNG (.png)
- JPEG (.jpg, .jpeg)

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guide](docs/CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Roman Vishnyakov - *Initial work* - [RomanVPX](https://github.com/RomanVPX)
