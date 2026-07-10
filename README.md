[Chest_XRay_Classifier_README.md](https://github.com/user-attachments/files/29906197/Chest_XRay_Classifier_README.md)
# Chest X-Ray Pathology Classifier

Automated multi-label thoracic disease detection from frontal chest X-ray images using deep learning.

**Built by:** [Tisha Chatterjee](https://linkedin.com/in/tishachatterjee) — Biomedical Engineering Student & Independent Researcher  
**Research Affiliations:** MIT Critical Data | Stanford S.Y.A.L.I.S Labs | CureQuest | Global Neurosurgery Fellowship (GWU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/tishact7/Chest_XRay_Pathology_Classifier/blob/main/CWH_chest_xray_ai_notebook.ipynb))

---

## What It Does

Upload a frontal chest X-ray image. Get independent probability scores for **14 thoracic pathologies** plus a "No Finding" classification.

**Use case:** AI-assisted screening for resource-limited settings where radiologist density is critically low.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | NIH Clinical Center — ChestX-ray14 |
| **Total Images** | 112,120 frontal-view chest X-rays |
| **Total Patients** | 30,805 |
| **Labels** | 14 thoracic diseases + "No Finding" |
| **Label Generation** | Extracted from radiology reports using Natural Language Processing (NLP) |
| **Image Type** | Frontal-view chest X-rays (PA/AP) |

### Classes Detected (14 + 1)

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Atelectasis | 8 | Hernia |
| 2 | Cardiomegaly | 9 | Infiltration |
| 3 | Consolidation | 10 | Mass |
| 4 | Edema | 11 | Nodule |
| 5 | Effusion | 12 | Pleural Thickening |
| 6 | Emphysema | 13 | Pneumonia |
| 7 | Fibrosis | 14 | Pneumothorax |
| — | **No Finding** | — | — |

### Training Subset

Due to Google Colab storage constraints (~12 GB GPU RAM, limited disk), the model trains on:

| Attribute | Detail |
|-----------|--------|
| **Subset** | Batch 1 of NIH ChestX-ray14 |
| **Images** | ~4,999 (~5,000) X-rays |
| **Size** | ~1 GB |
| **Reason** | Full dataset = ~45 GB — impractical for standard Colab session |

### Data Split

| Split | Percentage | Purpose |
|-------|------------|---------|
| Training | 80% | Model learning |
| Validation | 10% | Hyperparameter tuning & early stopping |
| Testing | 10% | Final performance evaluation |

---

## Model Architecture

| Attribute | Detail |
|-----------|--------|
| **Architecture** | DenseNet-121 |
| **Framework** | PyTorch |
| **Pretrained Weights** | ImageNet (14M+ natural images) |
| **Technique** | Transfer Learning → Progressive Unfreezing |

### Transfer Learning Strategy

1. **Phase 1 — Feature Extraction:**
   - Freeze all convolutional layers
   - Replace final classifier layer with 14-output linear layer
   - Train only the new classifier head
   - Epochs: 5 | LR: 0.001

2. **Phase 2 — Fine-Tuning:**
   - Unfreeze entire network
   - Train at reduced learning rate to preserve pretrained features
   - Epochs: 3 | LR: 0.0001

**Total Training:** 8 epochs

### Why DenseNet-121?

- **Parameter efficiency:** ~8M parameters vs. ResNet-50's ~25M — faster training, less overfitting on limited data
- **Feature reuse:** Dense connections improve gradient flow for medical imaging fine-grained patterns
- **Medical imaging standard:** Widely validated on ChestX-ray14 benchmark

---

## Image Preprocessing Pipeline

| Step | Transformation | Detail |
|------|---------------|--------|
| 1 | Grayscale → RGB | 1-channel to 3-channel for ImageNet compatibility |
| 2 | Resize | 256 × 256 pixels |
| 3 | Center Crop | 224 × 224 pixels (DenseNet-121 input size) |
| 4 | Tensor Conversion | PyTorch tensor format |
| 5 | Normalization | ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] |

### Training Augmentation (DataLoader)

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Random Crop | 224 × 224 from 256 × 256 | Spatial variability, reduce overfitting |
| Random Horizontal Flip | p = 0.5 | Left-right symmetry in chest anatomy |

**Note:** Aggressive augmentation avoided for medical imaging to preserve pathological features.

---

## Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **Loss Function** | `BCEWithLogitsLoss` | Multi-label classification — one X-ray may contain multiple diseases simultaneously |
| **Optimizer** | Adam | Adaptive learning rates, robust for transfer learning |
| **Initial LR** | 0.001 | Standard for classifier head training |
| **Fine-Tune LR** | 0.0001 | 10× reduction to prevent catastrophic forgetting of pretrained features |
| **LR Scheduler** | `ReduceLROnPlateau` | Automatically reduces LR when validation loss plateaus — prevents manual tuning |
| **Batch Size** | 32 | Balances GPU memory utilization and gradient stability |
| **Epochs (Total)** | 8 | 5 (feature extraction) + 3 (fine-tuning) |

---

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **ROC-AUC** | Primary — threshold-independent discrimination ability per class |
| **Precision** | Of predicted positives, how many are true positives? |
| **Recall** | Of actual positives, how many did we catch? |
| **F1-Score** | Harmonic mean of precision & recall — balanced performance |
| **Validation Loss** | Model generalization monitoring |
| **Training Loss** | Convergence monitoring |

**Multi-label approach:** Each disease predicted independently via sigmoid activation — no mutual exclusivity assumption (clinically realistic: a patient can have pneumonia + effusion).

---

## Prediction Output

For each uploaded chest X-ray, the model outputs:

```python
{
  "Atelectasis": 0.82,
  "Cardiomegaly": 0.15,
  "Consolidation": 0.91,
  "Edema": 0.07,
  "Effusion": 0.88,
  "Emphysema": 0.03,
  "Fibrosis": 0.12,
  "Hernia": 0.01,
  "Infiltration": 0.76,
  "Mass": 0.34,
  "Nodule": 0.22,
  "Pleural Thickening": 0.09,
  "Pneumonia": 0.94,
  "Pneumothorax": 0.05,
  "No Finding": 0.02
}
```

**Interpretation:** Probabilities are independent per class. A threshold (typically 0.5) converts to binary predictions. Clinical decision support, not diagnosis.

---

## Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **Environment** | Google Colab (GPU T4) |
| **Deep Learning** | PyTorch, Torchvision |
| **Data Processing** | NumPy, Pandas, Pillow (PIL) |
| **Visualization** | Matplotlib |
| **Metrics** | Scikit-learn |
| **Data Source** | Kaggle API (NIH ChestX-ray14) |

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Click the **"Open in Colab"** badge above
2. Runtime → Change runtime type → Select **GPU (T4)**
3. Run all cells sequentially
4. Upload a chest X-ray image (PNG/JPG)
5. View predicted pathology probabilities

### Option 2: Local Machine

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook chest_xray_classifier.ipynb
```

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~2 GB RAM minimum

---

## Files

| File | Description |
|------|-------------|
| `chest_xray_classifier.ipynb` | Main training & inference notebook |
| `requirements.txt` | Python dependencies |
| `sample_predictions/` | Example model outputs |
| `README.md` | This file |

---

## Limitations & Future Work

| Current Limitation | Planned Improvement |
|--------------------|---------------------|
| Trained on ~5,000 images (Batch 1 only) | Scale to full 112,120-image dataset via cloud storage |
| Single frontal view | Add lateral view support for comprehensive assessment |
| No localization (where is the pathology?) | Integrate Grad-CAM for explainable heatmap visualization |
| No clinical metadata (age, sex, symptoms) | Add multimodal fusion for personalized risk scoring |
| No web interface | Deploy as Gradio/Streamlit app for clinician access |
| No regulatory validation | Pursue IRB-approved retrospective validation on hospital data |

---

## Clinical Disclaimer

> **This model is for research and educational purposes only.** It is **not FDA-approved, not CE-marked, and not validated for clinical diagnosis.** Predictions should never replace qualified radiologist interpretation. Always consult a medical professional for diagnostic decisions.

---

## About the Author

**Tisha Chatterjee** is a Biomedical Engineering student and independent researcher converging on brain-computer interfaces, neuroplasticity, and medical imaging. She is a researcher at MIT Critical Data, a Global Neurosurgery Fellow at George Washington University, a cohort member at Stanford-affiliated S.Y.A.L.I.S Labs (top 8% of 2,000+ applicants), and a bioinformatics researcher at CureQuest. She is a TEDx Scholar (Top 10 globally) and an IIT Bombay Techfest College Ambassador (Rank 36 / 10,000+).

- 🔗 [LinkedIn](https://linkedin.com/in/tishachatterjee)
- 📧 tishachatterjee77@gmail.com
- 🧠 Research Interests: BCI · Neuroplasticity · Medical Imaging · Computational Biology · AI in Healthcare

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{chatterjee2026chestxray,
  author = {Chatterjee, Tisha},
  title = {Chest X-Ray Pathology Classifier: Multi-Label Thoracic Disease Detection with DenseNet-121},
  year = {2026},
  howpublished = {\url{https://github.com/YOUR-USERNAME/YOUR-REPO}},
  note = {Independent research project}
}
```

---

## License

**MIT License** — Open for research, educational, and non-commercial use.

**Not for clinical deployment** without regulatory approval (FDA, CE, CDSCO, etc.).

---

## Acknowledgments

- **NIH Clinical Center** for the ChestX-ray14 dataset
- **PyTorch Team** for the DenseNet-121 implementation
- **Google Colab** for free GPU access enabling independent research
- Mentors at MIT Critical Data, Stanford S.Y.A.L.I.S Labs, and CureQuest for research methodology guidance

---

*Last updated: July 2026 | Model version: 1.0 | Next release: v1.1 (Grad-CAM + Gradio UI)*
