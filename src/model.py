import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomCNN(nn.Module):
    """Custom CNN architecture for image classification"""
    
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResNetClassifier(nn.Module):
    """ResNet50 transfer learning model"""
    
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 transfer learning model"""
    
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        from efficientnet_pytorch import EfficientNet
        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
