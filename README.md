# 🗺️ Rural Route Navigator: Emergency Path Generation

> A dual-stage navigation and perception system that utilizes a custom **U-Net architecture** for road segmentation and the **A* search algorithm** to calculate optimal emergency paths through unstructured rural landscapes.

## Overview

During emergency responses in rural or disaster-stricken areas, established road networks are often undocumented, outdated, or obstructed. This project solves the challenge of navigating these terrains by combining deep learning to "see" the roads from aerial imagery and heuristic search to "plan" the safest, most efficient route.

The application is delivered as an interactive dashboard, allowing users to select two points on a map and receive an automatically calculated optimal path in real-time.

<p align="center">
  <img src="assets/Sequence 01_3.gif" width="800" alt="User Interaction Demo">
</p>

## Core Components

### Custom U-Net Architecture

The system utilizes a custom-built PyTorch U-Net model engineered for high-accuracy binary segmentation of aerial imagery. The architecture expects standard normalized RGB input tensors (N, 3, H, W) and outputs raw logits (N, 1, H, W) rather than activated probabilities, ensuring numerical stability during loss calculation (e.g., using BCEWithLogitsLoss).

Key architectural details include:

* **Double Convolution Blocks:** Both the encoder and decoder rely on DoubleConv modules. These consist of two consecutive 3x3 2D Convolutions with padding=1 (to strictly preserve spatial dimensions), each followed by 2D Batch Normalization and an in-place ReLU activation.

* **Contracting Path (Encoder):** The network extracts features through four standard downsampling levels with progressively increasing channel depths: 64, 128, 256, and 512. Dimensionality reduction is handled via 2x2 Max Pooling (stride 2). This depth provides a sufficient receptive field to identify rural terrain while explicitly avoiding the "vanishing road" problem that occurs when thin continuous structures are over-compressed in deeper networks.

* **Bottleneck:** The deepest layer expands the feature maps to 1024 channels before beginning the reconstruction phase.

* **Expansive Path (Decoder):** Upsampling is performed using 2x2 Transposed Convolutions (stride 2). To recover lost spatial precision, skip connections from the contracting path are concatenated (dim=1) with the upsampled feature maps. The forward pass also includes dynamic bilinear interpolation to gracefully handle and realign any 1-pixel shape mismatches caused by odd-dimension inputs.

* **Final Classifier:** A terminal 1x1 convolution maps the final 64-channel feature block down to a single channel representing the navigable road structure.

### Binary Masking & Perception

The primary output of the vision model is a **binary navigability mask**.

* Filters out environmental noise to isolate only viable, structurally sound paths.

* Converts raw RGB aerial imagery into a simplified, mathematically traversable grid.

* Thresholding is optimized specifically for rural terrain characteristics, ensuring stability in non-urban, unpaved environments.

### A* Pathfinding & Distance Approximation

Once the navigable grid is generated, the routing engine calculates the most efficient traversal:

* **Heuristic Search (A*):** Finds the optimal path between user-selected coordinates by balancing actual grid distance and heuristic estimates.

* **Terrain-Aware Routing:** Instead of simple Euclidean ("straight-line") math, the logic relies on the segmented grid to provide realistic, obstacle-avoiding travel paths across the rural map.

## Tech Stack

* **Front-End Interface:** Streamlit (Interactive Web UI)
* **Deep Learning Framework:** PyTorch
* **Computer Vision:** OpenCV, Matplotlib, NumPy, and PIL (Image processing & mask manipulation)
* **Algorithms:** Custom A* implementation
* **Language:** Python 3.x

## Repository Structure

```text
├── app.py                  # Streamlit frontend dashboard and user interface
├── inference_engine.py     # Wraps the PyTorch model for real-time predictions
├── pathfinding.py          # Custom A* search algorithm and grid logic
├── Model.py                # U-Net PyTorch architecture definition
├── Train.py                # Model training loop, loss functions, and optimization
├── Test.py                 # Evaluation script with batch IoU metrics and visualization
├── Data_Process.py         # Dataloaders, image transformations, and dataset splitting
└── requirements.txt        # Project dependencies