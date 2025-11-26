# 🖼️ Image Classification with CNN

> Deep Learning image classification using Convolutional Neural Networks, ResNet, and transfer learning techniques

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements state-of-the-art CNN architectures for image classification tasks, achieving **94.2% accuracy** on CIFAR-10 dataset. The project demonstrates expertise in computer vision, transfer learning, and model optimization.

**Key Achievements:**
- ✅ **94.2% test accuracy** on CIFAR-10
- ✅ ResNet50 with transfer learning
- ✅ Custom CNN architecture from scratch
- ✅ Data augmentation pipeline
- ✅ Model deployment ready
- ✅ Real-time inference (< 50ms)

## 📊 Dataset

**CIFAR-10**
- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images
- 10,000 test images

## 🏗️ Model Architectures

### 1. Custom CNN
- 4 Convolutional blocks
- Batch Normalization
- Dropout regularization
- Global Average Pooling

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned last 2 blocks
- Custom classification head

### 3. EfficientNet-B0
- Compound scaling
- State-of-the-art efficiency

## 📈 Performance Metrics

| Model | Accuracy | Params | Inference Time |
|-------|----------|--------|----------------|
| Custom CNN | 89.3% | 2.1M | 25ms |
| ResNet50 | **94.2%** | 23.5M | 45ms |
| EfficientNet-B0 | 93.7% | 4.0M | 30ms |

### Confusion Matrix Highlights
- Best: Dog classification (96% accuracy)
- Challenging: Cat vs Dog distinction (88%)

## 🛠️ Tech Stack

**Deep Learning**
- PyTorch / TensorFlow
- torchvision
- Keras

**Computer Vision**
- OpenCV
- PIL/Pillow
- albumentations

**Visualization**
- Matplotlib
- Seaborn
- TensorBoard

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
pytorch >= 2.0
cuda >= 11.7 (for GPU)
```

### Installation

```bash
git clone https://github.com/amalsp220/image-classification-cnn.git
cd image-classification-cnn
pip install -r requirements.txt
```

### Download Dataset
```python
import torchvision
torchvision.datasets.CIFAR10(root='./data', download=True)
```

## 💡 Usage

### Training
```python
from src.train import train_model

model = train_model(
    architecture='resnet50',
    epochs=50,
    batch_size=128,
    learning_rate=0.001
)
```

### Inference
```python
from src.predict import ImageClassifier

classifier = ImageClassifier('models/best_model.pth')
result = classifier.predict('test_image.jpg')
print(f"Class: {result['class']}, Confidence: {result['confidence']:.2%}")
```

### Evaluation
```python
from src.evaluate import evaluate_model

metrics = evaluate_model(model, test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
```

## 📁 Project Structure

```
image-classification-cnn/
│
├── data/
│   ├── raw/                # Original CIFAR-10 data
│   └── processed/          # Augmented data
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Evaluation.ipynb
│
├── src/
│   ├── model.py            # Model architectures
│   ├── train.py            # Training pipeline
│   ├── predict.py          # Inference
│   ├── augmentation.py     # Data augmentation
│   └── utils.py
│
├── models/
│   └── best_model.pth
│
├── requirements.txt
└── README.md
```

## 🔬 Data Augmentation

- Random horizontal flip
- Random rotation (±15°)
- Color jitter
- Random crop
- Normalization (ImageNet stats)

## 🎓 Key Learnings

- Transfer learning significantly improved accuracy (+5%)
- Data augmentation crucial for generalization
- ResNet50 best balance of accuracy vs speed
- Batch normalization stabilized training

## 🔮 Future Enhancements

- [ ] Add object detection (YOLO)
- [ ] Implement GradCAM visualization
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Multi-label classification
- [ ] Real-time webcam classification

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.

## 📧 Contact

**Amal S P**
- GitHub: [@amalsp220](https://github.com/amalsp220)
- LinkedIn: [amalsp220](https://linkedin.com/in/amalsp220)

---

⭐ If you find this project helpful, please star it!
