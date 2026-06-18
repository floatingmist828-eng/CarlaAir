# 🛠️ Installation Guide

Welcome to Griffin installation! This guide will help you set up Griffin, an aerial-ground cooperative 3D perception framework built on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).

## 📋 Overview

Griffin requires specific versions of PyTorch, CUDA, and mmdetection3d for optimal performance. The installation process takes approximately 10-20 minutes depending on your internet connection and system specifications.

## 🔧 Detailed Installation

### Prerequisites

Before starting, ensure your system meets these requirements:

| Requirement          | Version/Description               |
| -------------------- | --------------------------------- |
| **Operating System** | Linux (Ubuntu 18.04+ recommended) |
| **GPU**              | CUDA-capable NVIDIA GPU           |
| **CUDA**             | Version 11.X                      |
| **Python**           | 3.8 (managed via Conda)           |

### Step 1: Environment Setup

Create a dedicated Conda environment to avoid conflicts:

```bash
conda create -n griffin python=3.8 -y
conda activate griffin
```

> 💡 **Tip**: Always activate the `griffin` environment before working with the framework

### Step 2: Install PyTorch

Install PyTorch with CUDA 11.1 support for optimal compatibility:

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Configure CUDA Environment

Set the CUDA_HOME environment variable for compilation requirements:

```bash
export CUDA_HOME=/usr/local/cuda-11.8/  # Adjust path to your CUDA installation
```

(Optional) Make it permanent by adding to your shell configuration:
```bash
echo 'export CUDA_HOME=/usr/local/cuda-11.8/' >> ~/.bashrc
source ~/.bashrc
```

> 🔍 **Find your CUDA path**: Run `which nvcc` or `ls /usr/local/` to locate your CUDA installation

### Step 4: Install Core Dependencies

Install the OpenMMLab ecosystem components:

```bash
# Install mmcv-full with CUDA support
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install mmdet and mmsegmentation
pip install mmdet==2.14.0 mmsegmentation==0.14.1
```

> ⏱️ **Expected time**: This step may take 5-10 minutes as mmcv-full is a large package

### Step 5: Install mmdetection3d

Clone and install mmdetection3d from source:

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .
cd ..
```

> 📌 **Note**: We use version v0.17.1 for stability and compatibility with Griffin.

### Step 6: Install Griffin

Clone the Griffin repository and install its dependencies:

```bash
git clone https://github.com/wang-jh18-SVM/Griffin.git griffin
cd griffin
pip install -r requirements.txt
```

### Step 7: Install AB3DMOT (Optional)

Griffin provides an optional integration with [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) for Late Fusion tracking capabilities.

```bash
# Clone the required toolbox to this specified location
cd projects/ab3dmot_plugin
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
cd Xinshuo_PyToolbox
pip install -r requirements.txt

# Install additional dependencies
conda install -c conda-forge easydict -y
pip install filterpy
cd ../../..
```

> 🎯 **When to use**: Install this only if you plan to use Late Fusion tracking methods

## ✅ Verification

Verify your installation is working correctly:

```bash
# Test PyTorch and CUDA
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"

# Test mmdetection3d
python -c "import mmdet3d; print(f'✅ mmdet3d version: {mmdet3d.__version__}')"

```

**Expected output:**
```
✅ PyTorch version: 1.9.1+cu111
✅ CUDA available: True
✅ mmdet3d version: 0.17.1
✅ Griffin installation successful!
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue                 | Solution                                                       |
| --------------------- | -------------------------------------------------------------- |
| **CUDA not found**    | Ensure CUDA is installed and `CUDA_HOME` is set correctly      |
| **Version conflicts** | Use the exact versions specified in this guide                 |
| **Import errors**     | Check that all dependencies are installed in the correct order |

### Getting Help

1. **Check example environment**: We provide `example.yaml` with tested dependency versions
2. **Verify dependencies**: Use `conda list` and `pip list` to check installed package versions
3. **Clean installation**: If issues persist, create a fresh conda environment

```bash
# Clean reinstall
conda env remove -n griffin --all
# Then repeat the installation steps
```

## 🎉 Next Steps

Once installation is complete:

1. 📊 **[Dataset Preparation](Dataset_Preparation.md)** - Download and prepare Griffin dataset
2. 🏋️ **[Training and Evaluation](Training_and_Evaluation.md)** - Train models and run evaluations
3. 👁️ **[Visualization](Visualization.md)** - Visualize results and data

Happy coding with Griffin! 🚁🚗