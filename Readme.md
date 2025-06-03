# 🔬 Histopathological Image Super-Resolution: SR3 Diffusion Model with Interactive GUI 🖼️

This repository presents a comprehensive solution for super-resolution of histopathological tissue images using a PyTorch-based **SR3 (Super-Resolution via Repeated Refinement)** diffusion model. It includes image patching, blending, model inference, and an interactive GUI built with Tkinter.

---

## 📝 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Installation](#-installation)
- [Usage](#-usage)
- [Files in this Repository](#-files-in-this-repository)
- [Results & Visuals](#-results--visuals)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🚀 Project Overview

<table>
  <tr>
    <td align="center"><b>40X Sample (Input)</b><br><img src="002-40x.jpg" width="300"/></td>
    <td align="center"><b>100X Zoomed (SR3 Output)</b><br><img src="002-100x.jpg" width="300"/></td>
    <td align="center"><b>400X Zoomed (SR3 Output)</b><br><img src="002-400x.jpg" width="300"/></td>
  </tr>
</table>

The primary objective of this project is to enhance low-resolution histopathological images (from standard lens microscopes) to generate high-resolution equivalents—matching electron microscopy quality—using a custom SR3 diffusion model.

Additionally, the project provides a GUI to:
- Load histopathological images
- Select image regions via bounding box
- Apply real-time super-resolution on the selected regions

---

## ✨ Features

- **SR3 Diffusion Model**: Implements a DDPM-based SR3 architecture.
- **Patch-Based Processing**: Supports large image patching and seamless reconstruction.
- **Configurable Patching**: Adjustable patch sizes and overlap.
- **Interactive GUI**: Built using Tkinter to allow visual selection and super-resolution.
- **Zoom Factors**: 2.5X and 4X zoom supported.
- **PyTorch Implementation**: Fully GPU-compatible.
- **Customizable Loss & Schedules**: Supports `l1`, `mse`, and noise schedules like linear, cosine.

---

## 🧠 Model Architecture

- **UNetModel**: A U-Net variant with:
  - `in_channels=6`, `out_channels=3`
  - Skip connections to preserve fine details
- **Diffusion Class**:
  - Forward (q): Adds noise to the image across steps
  - Reverse (p): Learns to denoise back to HR
  - Uses `make_beta_schedule` for noise control

---

## ⚙️ Training

- **Optimizer**: `torch.optim.Adam`
- **Loss**: `nn.L1Loss`
- **Dataset**: Custom folder structure:
