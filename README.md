# ResTL-CORAL: Resting-State Transfer Learning for Cross-Subject MI-EEG Classification

This repository is a PyTorch-based implementation of **ResTL** — a cross-subject transfer learning framework for motor imagery EEG classification, inspired by the MICCAI 2024 paper:  
> *"Subject-Adaptive Transfer Learning Using Resting State EEG Signals for Cross-Subject EEG Motor Imagery Classification."*

We support flexible backbone model choices (e.g., `EEGNet`, `Conformer`), and integrate **CORAL loss** for domain adaptation. The pipeline uses *resting-state EEG to synthesize task-like EEG signals*, reducing the cost of calibration in new subjects.

---

## 🔧 How to Run

### 1. Train the model (The first step)
```bash
# Using Conformer backbone (default)
python main.py -m Conformer

# or with CORAL variant
python main-coral.py -m Conformer
```
### 2. Test the model (The second and third step)
```bash
# Standard test pipeline
python main.py -t test -m Conformer

# or CORAL-based test
python main-coral.py -t test -m Conformer
```
> You can replace Conformer with EEGNet or other supported backbones.
## 📁 Project Structure
```graphql
ResTL-CORAL/
├─ auxiliary.py              # Auxiliary training utilities
├─ conformer.py              # Conformer backbone network
├─ Dataset.py                # Custom OpenBMI dataset (RS + MI)
├─ main.py                   # Main training/testing script (EEGNet/Conformer)
├─ main-coral.py       # CORAL-enhanced variant
├─ model.py                  # Feature decomposition and distribution modeling
├─ train.py                  # Training pipeline (multi-loss setup)
├─ utils.py                  # Logging, checkpointing, etc.
├─ Overview.png              # Model architecture overview
├─ predata.ipynb             # Data preprocessing notebook
└─ README.md                 # Project introduction and instructions
```
## 📚 References & Acknowledgements
Jeon et al. : https://github.com/eunjin93/SICR_BCI \
EEG-Conformer: https://github.com/eeyhsong/EEG-Conformer \
MICCAI2024-ResTL: https://github.com/SionAn/MICCAI2024-ResTL
## 📌 Notes
- The dataset used is OpenBMI (MI subset). You should download and place it under `openBMI_MI/` following the same naming format. Or you can modify `root_dir` to the path where you store your data.
- For training with leave-one-subject-out protocol, specify subject IDs via `-te` argument.
- Only OpenBMI dataset loading is currently implemented
