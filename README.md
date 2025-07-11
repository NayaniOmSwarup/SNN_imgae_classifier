# 🧠 Hybrid ANN–SNN Image Classifier using SHNO-Based Spiking Layer

This project presents a **hybrid neural network** that combines conventional Artificial Neural Network (ANN) layers with a **physics-accurate Spiking Neural Network (SNN) layer** based on **Spin Hall Nano-Oscillators (SHNOs)**. It models real-world nanomagnetic device behavior and integrates it into a deep learning pipeline for **4-class image classification**.

> 🧩 **Only one hidden layer uses SHNO-based spiking neurons** with LLGS dynamics; the rest of the network is a standard ANN, forming a hybrid architecture.

---

## ⚡ What Makes This Project Unique?

* 🔬 **Physics-accurate SHNO spiking layer** built from first principles using LLGS equations.
* 🧠 **Surrogate gradient learning** to enable backpropagation through the spiking layer.
* 🏗️ **Hybrid architecture**: combines ANN feature extractors with a biologically and physically inspired SNN core.
* ⏱️ **Multi-scale temporal integration** mimicking neuromorphic time-series behavior.
* 🌐 **Structured noise and dynamic scaling** to model real-world variability in SHNO hardware.

---

## 🧠 Hybrid Architecture Overview

```text
Input Image
   ↓
ANN Feature Extractor (Linear → ReLU → Dropout → Linear)
   ↓
⚡ SHNO Spiking Layer (LLGS-based magnetization dynamics)
   ↓
ANN Output Classifier (Linear → ReLU → Dropout → Linear)
   ↓
Prediction (4 classes)
```

This fusion enables the model to benefit from:

* ANN's expressiveness and training stability,
* SNN’s energy-efficient, time-dependent, physics-grounded representation in the SHNO layer.

---

## 📂 File Structure

```
SNN_image_classifier/
│
├── SNN_project.ipynb         # Notebook with hybrid ANN-SNN model
├── README.md                 # This file
```

---

## 📦 Requirements

```bash
pip install torch torchvision numpy matplotlib
```

CUDA is automatically used if available.

---

## 📈 Model Summary

* **Input**: 28x28 grayscale images (likely MNIST or custom)
* **ANN Layers**:

  * 784 → 256 → 128 → 60 neurons
* **SHNO SNN Layer**:

  * 60 spiking neurons with LLGS-based dynamics
* **ANN Output Head**:

  * 60 → 4 output classes (main + auxiliary prediction)

---

## 🔬 SHNO Spiking Layer

The core `SHNOLayer` simulates:

* 🧲 **Magnetization (m)** updates over time using:

  * Effective magnetic field (H\_eff)
  * Spin-Hall torque
  * LLGS dynamics with damping (α), gyromagnetic ratio (γ), and spin torque terms
* ⏳ **Spikes** generated when the y-component of magnetization crosses a threshold
* 🔄 **Soft reset mechanism** and structured dynamics over 15 time steps

Implemented using:

```python
class SHNOLayer(nn.Module):
    ...
    def forward(self, J_in):
        ...
        # Spike generation logic using exact physics
```

---

## 🧪 Training Loop

* Trains using **cross-entropy loss** on both the main and auxiliary classifier heads.
* Time series simulation across 15 steps per forward pass.
* Gradient-based learning through surrogate spikes.

---

## 📊 Example Output (Visualized in Notebook)

* Spike raster plots across time
* Membrane potential evolution
* Classification accuracy progression
<img width="1147" height="845" alt="image" src="https://github.com/user-attachments/assets/871c2e62-1583-40e5-9541-c086902c60d5" />
<img width="1367" height="802" alt="image" src="https://github.com/user-attachments/assets/fcf0d3fe-46c8-49e2-bc95-da636a7be66c" />
<img width="1160" height="857" alt="image" src="https://github.com/user-attachments/assets/69318c19-88eb-4dd3-a32f-2e585dba7447" />


