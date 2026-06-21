# 🗺️ Rural Route Navigator: Emergency Path Generation

> A dual-stage navigation and perception system that utilizes a custom **U-Net architecture** for road segmentation and the **A* search algorithm** to calculate optimal emergency paths through unstructured rural landscapes.

## 🚀 Overview

During emergency responses in rural or disaster-stricken areas, established road networks are often undocumented, outdated, or obstructed. This project solves the challenge of navigating these terrains by combining deep learning to "see" the roads from aerial imagery and heuristic search to "plan" the safest, most efficient route.

The application is delivered as an interactive dashboard, allowing users to select two points on a map and receive an automatically calculated optimal path in real-time.

<p align="center">
  <img src="assets/Sequence 01_3.gif" width="800" alt="User Interaction Demo">
</p>

## 🛠 Core Components

### 🧠 Custom U-Net Architecture

The system utilizes a lightweight, custom-built PyTorch U-Net model designed for high-accuracy spatial feature extraction without the computational bloat of deeper networks:

* **Targeted Receptive Field:** Engineered to identify and extract narrow rural road textures from complex environmental backgrounds (soil, grass, tree canopies).

* **Optimized Downsampling:** Strategically limited pooling layers to prevent the "vanishing road" problem, preserving the continuous topology of thin rural paths.

* **High-Fidelity Decoder:** Ensures the resulting segmentation map maintains high resolution for precise pixel-level navigation.

### 🛣️ Binary Masking & Perception

The primary output of the vision model is a **binary navigability mask**.

* Filters out environmental noise to isolate only viable, structurally sound paths.

* Converts raw RGB aerial imagery into a simplified, mathematically traversable grid.

* Thresholding is optimized specifically for rural terrain characteristics, ensuring stability in non-urban, unpaved environments.

### 📍 A* Pathfinding & Distance Approximation

Once the navigable grid is generated, the routing engine calculates the most efficient traversal:

* **Heuristic Search (A*):** Finds the optimal path between user-selected coordinates by balancing actual grid distance and heuristic estimates.

* **Terrain-Aware Routing:** Instead of simple Euclidean ("straight-line") math, the logic relies on the segmented grid to provide realistic, obstacle-avoiding travel paths across the rural map.

## 💻 Tech Stack

* **Front-End Interface:** Streamlit (Interactive Web UI)
* **Deep Learning Framework:** PyTorch
* **Computer Vision:** OpenCV, Matplotlib, NumPy, and PIL (Image processing & mask manipulation)
* **Algorithms:** Custom A* implementation
* **Language:** Python 3.x

## 📂 Repository Structure

```text
├── app.py                  # Streamlit frontend dashboard and user interface
├── inference_engine.py     # Wraps the PyTorch model for real-time predictions
├── pathfinding.py          # Custom A* search algorithm and grid logic
├── Model.py                # U-Net PyTorch architecture definition
├── Train.py                # Model training loop, loss functions, and optimization
├── Test.py                 # Evaluation script with batch IoU metrics and visualization
├── Data_Process.py         # Dataloaders, image transformations, and dataset splitting
└── requirements.txt        # Project dependencies