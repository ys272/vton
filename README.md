# Fashion AI Virtual Try-On System

## Overview

AI-powered virtual try-on (VTON) system inspired by **"Try-On Diffusion (A Tale of Two UNets)"**. Implements the dual U-Net architecture with efficiency improvements, MoveNet pose estimation, SCHP human parsing, and cross-attention conditioning.

**Key Features:** Dual U-Net architecture • MoveNet pose estimation • SCHP human parsing • Multi-resolution support (64×44 to 1024×704) • DDIM & Karras sampling • Mixed precision training • Experimental alternative approaches

## Directory Structure

### `base_vton/` - **Main Implementation**
Primary VTON system following "Tale of Two UNets" with efficiency improvements.
- **`model.py`**: Dual U-Nets (`Unet_Person_Masked`, `Unet_Clothing`) with cross-attention and FiLM conditioning
- **`train.py`**: Training pipeline with EMA, AMP, gradient accumulation, TensorBoard logging
- **`datasets.py`**: Data loading with pose processing, masking, and augmentation
- **`evaluate.py`**: Evaluation and inference utilities

### `vton_blur/` - **Alternative Approach**
Experimental blur-based conditioning (alternative to cross-attention).

### `clothing_autoencoder/` - **Alternative Approach**
Experimental autoencoder/classifier for clothing feature extraction with residual connections and self-attention.

### `data_preprocessing_vton/`
Fashion dataset preprocessing with computer vision integration.

**Core CV Components:**
- **`pose.py`**: MoveNet (Thunder/Lightning) for 17-keypoint pose detection
- **`schp.py`**: SCHP human parsing (ATR, Pascal, LIP schemes)

**Dataset Processors:** Handles multiple data sources (artistic, clothing-people pairs, graphic tees, online datasets, multi-pose, high-res paired, same-person-two-poses)

### Other Directories
- **`diffusion_ddim.py`** & **`diffusion_karras.py`**: Diffusion sampling implementations
- **`fmnist/`**: Fashion-MNIST experiments
- **`config.py`**: System configuration (paths, model settings, hyperparameters)
- **`nn_utils.py`** & **`utils.py`**: Neural network blocks and utilities
- **`scripts/`**: Shell scripts for external tools (SCHP)

## Architecture Comparison with Try-On Diffusion

### Core Architecture
| Aspect | Try-On Diffusion | This Repository |
|--------|------------------|-----------------|
| **Dual UNets** | Person + Clothing UNets with cross-conditioning | `Unet_Person_Masked` + auxiliary network (`Unet_Clothing` or `Clothing_Classifier`) |
| **Auxiliary Network** | Full clothing UNet | **Default:** `Clothing_Classifier` (lighter encoder)<br>**Available:** `Unet_Clothing` (full UNet) |
| **Cross-Attention** | Cross-branch conditioning at selected resolutions | Multi-scale: mid blocks (16×) and up path (32×, 64×)<br>Resolution-aware (e.g., 64× level handling for medium size) |
| **Conditioning** | Cross-UNet conditioning | FiLM time embeddings + cross-attention from clothing features<br>Optional pose/noise FiLM branches (toggleable) |

### Computer Vision Integration
- **Pose:** MoveNet with 17-keypoint detection → rasterized into channel-wise sparse binary maps (one-hot at keypoint locations)
- **Parsing:** SCHP integration for person masks and body segmentation
- **Pipeline:** Explicit preprocessing implementation (`data_preprocessing_vton/`) with multi-scale support

### Efficiency Improvements
- **Model:** Lighter auxiliary encoder (`Clothing_Classifier`) reduces parameters/memory
- **Training:** Mixed precision (AMP/bfloat16), gradient accumulation, EMA, dynamic accumulation rate
- **Data:** Multi-resolution support (t/s/m/l), dataset sharding for medium scale
- **Logging:** TensorBoard integration with gradient monitoring

### Experimental Alternatives
- **`vton_blur/`**: Blur-based conditioning instead of cross-attention
- **`clothing_autoencoder/`**: Autoencoder variants for clothing understanding

### Technical Details
**Resolutions:** Tiny (64×44) • Small (128×88) • Medium (256×176) • Large (1024×704)

**Sampling:** DDIM and Karras diffusion methods

**Main UNet Inputs (example for training):** `Unet_Person_Masked` receives a 19-channel tensor composed of masked person RGB, pose matrix channels, mask coordinates, and auxiliary signals, as defined in `base_vton/train.py` (see `num_start_channels = 19`).

**Data Flow:** Pose estimation → Human parsing → Mask generation → Clothing encoding → Diffusion generation with cross-attention conditioning

---

*Extends "Try-On Diffusion" methodology with efficiency improvements, explicit CV integration, and experimental architectural alternatives.*
