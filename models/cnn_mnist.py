"""
Простая CNN модель для MNIST датасета
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """Простая CNN с двумя свёрточными слоями для MNIST"""
    
    def __init__(self, num_classes=10):
        super(MnistCNN, self).__init__()
        
        # Свёрточные слои
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Первый свёрточный блок
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Второй свёрточный блок
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(num_classes=10):
    """Фабричная функция для создания модели"""
    return MnistCNN(num_classes=num_classes)
