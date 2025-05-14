
# TSGAN: Tumor-Aware Synthesis of Contrast-Enhanced MRI Without Contrast Agent

This project implements a full MLOps-enabled pipeline for **contrast-free synthesis of CeT1-weighted MRI scans** using a novel **TSGAN (Tumor-Aware Segmentation-Guided GAN)** model. The goal is to generate high-quality CeT1 scans from PreT1 inputs, minimizing the use of contrast agents for cancer diagnosis.

---

##  Project Highlights

- ✅ 3D MRI Processing with U-Net based Generator
- ✅ Tumor-Aware Segmentation Pretraining (3D U-Net)
- ✅ GAN Training with Dice + Perceptual + Adversarial Loss
- ✅ Mixed Precision Training (AMP) with Gradient Scaling
- ✅ Monitoring with TensorBoard + Image Visualizations
- ✅ Clean PyTorch Training Pipeline
- ✅ Modular Codebase (DataLoader, Model, Utils)
- ✅ Streamlit-based Inference App for real-time testing
- ✅ Future-ready: Docker, MLflow, GitHub CI/CD, Clinical Validation

---

##  Dataset

- Source: [BraTS, TCIA, or Custom Dataset]
- Format: 3D MRI volumes in `.nii.gz`
- Modalities:
  - PreT1 (input)
  - CeT1 (target)
  - Tumor Segmentation (mask for guidance)

---

## 📁 Project Structure
MRI_Synthesis_Project/
├── data/
│ └── Resized_dataset/
├── models/
│ ├── tsgan_generator.py
│ ├── tsgan_discriminator.py
│ └── unet3d_segmentation.py
├── utils/
│ ├── dataset_loader.py
│ ├── losses.py
│ ├── logger.py
│ └── visualization.py
├── md_train.py # TSGAN Training script
├── segment_train.py # Tumor segmentation (3D U-Net)
├── generate_mri.py # Inference Script
├── app.py # Streamlit App for real-time inference
├── requirements.txt
└── README.md
