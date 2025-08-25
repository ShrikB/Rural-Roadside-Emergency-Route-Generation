# Emergency Route Generation using U-Net

This project implements a **U-Net convolutional neural network** for **road segmentation** to support **emergency route generation** in areas with limited or outdated cartography.  
By extracting road networks from aerial or satellite imagery, the model enables downstream shortest-path computation for reliable routing in emergency situations.

---

## 📖 Overview

- **Architecture:** U-Net  
  - Contracting path (downsampling with convolutions and pooling)  
  - Expansive path (upsampling with interpolation and transposed convolutions)  
  - Skip connections to preserve spatial detail  
- **Objective:** Extract binary road masks from imagery.  
- **Application:** Generate weighted road graphs → compute shortest emergency routes.  

---


## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib / PIL**
- **scikit-learn** (for data prep and train/validation splits)

---
