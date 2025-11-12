# Fashion AI Virtual Try-On System

## Overview

This is an ML-powered virtual try-on (VTON) system, mostly inspired by **"Try-On Diffusion (A Tale of Two UNets)"**. I've implemented the dual U-Net architecture with some efficiency improvements along the way, using MoveNet for pose estimation, SCHP for human parsing, and cross-attention for conditioning.

**Key Features:** Dual U-Net architecture • MoveNet pose estimation • SCHP human parsing • Multi-resolution support (64×44 to 1024×704) • DDIM & Karras sampling • Mixed precision training • Experimental alternative approaches

## Directory Structure

### `base_vton/` - **Main Implementation**
This is where the main VTON system lives, following the "Tale of Two UNets" approach with some pragmatic efficiency tweaks.
- **`model.py`**: The dual U-Nets (`Unet_Person_Masked`, `Unet_Clothing`) with cross-attention and FiLM conditioning
- **`train.py`**: Training pipeline featuring EMA, AMP, gradient accumulation, and TensorBoard logging
- **`datasets.py`**: Data loading with pose processing, masking, and augmentation
- **`evaluate.py`**: Evaluation and inference utilities

### `vton_blur/` - **Alternative Approach**
This was an experiment trying out blur-based conditioning as an alternative to cross-attention.

### `clothing_autoencoder/` - **Alternative Approach**
An autoencoder/classifier approach for learning clothing features, built with residual connections and self-attention.

### `data_preprocessing_vton/`
All the preprocessing code for fashion datasets, with computer vision tools baked right in.

**Core CV Components:**
- **`pose.py`**: MoveNet (Thunder/Lightning) for 17-keypoint pose detection
- **`schp.py`**: SCHP human parsing supporting different segmentation schemes (ATR, Pascal, and LIP)

**Dataset Processors:** There are processors for multiple data sources here.

### Other Directories
- **`diffusion_ddim.py`** & **`diffusion_karras.py`**: Two diffusion sampling implementations
- **`fmnist/`**: Some Fashion-MNIST experiments
- **`config.py`**: System-wide configuration covering paths, model settings, and hyperparameters
- **`nn_utils.py`** & **`utils.py`**: Various neural network blocks and utility functions
- **`scripts/`**: Shell scripts for running external tools like SCHP

## Architecture Comparison with Try-On Diffusion

### Core Architecture
| Aspect | Try-On Diffusion | This Repo |
|--------|------------------|-----------------|
| **Dual UNets** | Person + Clothing UNets with cross-conditioning | `Unet_Person_Masked` + auxiliary network (`Unet_Clothing` or `Clothing_Classifier`) |
| **Auxiliary Network** | Full clothing UNet | **Default:** `Clothing_Classifier` (lighter encoder)<br>**Available:** `Unet_Clothing` (full UNet) |
| **Cross-Attention** | Cross-branch conditioning at selected resolutions | Multi-scale: mid blocks (16×) and up path (32×, 64×)<br>Resolution-aware (e.g., 64× level handling for medium size) |
| **Conditioning** | Cross-UNet conditioning | FiLM time embeddings + cross-attention from clothing features<br>Optional pose/noise FiLM branches (toggleable) |

### Computer Vision Integration
- **Pose:** Using MoveNet with 17-keypoint detection, which gets rasterized into channel-wise sparse binary maps (one-hot encoded at keypoint locations)
- **Parsing:** SCHP handles person masks and body segmentation
- **Pipeline:** The preprocessing is all implemented explicitly in `data_preprocessing_vton/` with multi-scale support

### Efficiency Improvements
- **Model:** The default auxiliary encoder (`Clothing_Classifier`) is lighter on parameters and memory
- **Training:** Mixed precision training (AMP/bfloat16), gradient accumulation, EMA, and dynamic accumulation rate
- **Data:** Multi-resolution support (tiny/small/medium/large), with dataset sharding at medium scale
- **Logging:** TensorBoard integration with gradient monitoring

### Experimental Alternatives
- **`vton_blur/`**: Tried blur-based conditioning instead of cross-attention
- **`clothing_autoencoder/`**: Explored autoencoder variants for clothing understanding

### Technical Details
**Resolutions:** The system supports Tiny (64×44) • Small (128×88) • Medium (256×176) • Large (1024×704)

**Sampling:** Both DDIM and Karras diffusion methods are implemented

**Main UNet Inputs (training example):** The `Unet_Person_Masked` network receives a 19-channel tensor that combines masked person RGB, pose matrix channels, mask coordinates, and auxiliary signals (defined in `base_vton/train.py` where `num_start_channels = 19`).

**Data Flow:** The pipeline goes: Pose estimation → Human parsing → Mask generation → Clothing encoding → Diffusion generation with cross-attention conditioning

