# Rice Leaf Disease Classification (Deep Learning)

This project implements an end-to-end deep learning pipeline for rice leaf disease classification using CNN and transfer learning models.

## 🔍 Overview
- Task: Image classification (5 disease categories)
- Dataset: ~1,100 rice leaf images
- Models:
  - Custom CNN (baseline)
  - ResNet18 (transfer learning)
  - ResNet50 (deep architecture)

## ⚙️ Key Features
- Data preprocessing and cleaning
- Stratified train/validation/test split
- Data augmentation for improved generalisation
- Model training and validation pipeline
- Multi-scenario evaluation (white / field / mixed backgrounds)
- Performance metrics: accuracy, precision, recall, F1-score
- Confusion matrix and result visualisation

## 📊 Results
- ResNet50 achieved the best performance across most scenarios
- Transfer learning significantly improved model robustness compared to baseline CNN
- Performance varied across different background conditions, highlighting real-world challenges

## Results (ResNet50)

### White Background
![ResNet50 White](figs/resnet50_confusion_matrix_white.png)

### Field Scenario
![ResNet50 Field](figs/resnet50_confusion_matrix_field.png)

### Mixed Scenario
![ResNet50 Mixed](figs/resnet50_confusion_matrix_mixed.png)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py --model all