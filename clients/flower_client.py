"""
Flower клиент для федеративного обучения
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
from collections import OrderedDict


class FlowerClient(fl.client.NumPyClient):
    """Клиент для федеративного обучения с Flower"""
    
    def __init__(self, cid: int, model: nn.Module, trainloader: DataLoader, 
                 testloader: DataLoader, device: torch.device):
        """
        Args:
            cid: ID клиента
            model: PyTorch модель
            trainloader: DataLoader для обучения
            testloader: DataLoader для тестирования
            device: устройство (cpu/cuda)
        """
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        
        # Для хранения глобальных весов (нужно для FedProx)
        self.global_params = None
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Получить параметры модели как numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Установить параметры модели из numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # Сохраняем глобальные параметры для FedProx (копируем numpy arrays)
        self.global_params = [p.copy() for p in parameters]
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Обучение модели на локальных данных
        
        Args:
            parameters: глобальные параметры модели
            config: конфигурация (epochs, batch_size, lr, mu для FedProx)
            
        Returns:
            Обновлённые параметры, количество примеров, метрики
        """
        self.set_parameters(parameters)
        
        # Извлекаем параметры из конфига
        epochs = config.get('local_epochs', 1)
        learning_rate = config.get('learning_rate', 0.01)
        mu = config.get('mu', 0.0)
        proximal = config.get('proximal', False)
        
        # Обучаем модель
        num_examples, metrics = self.train(epochs, learning_rate, mu, proximal)
        
        # Возвращаем обновлённые параметры
        return self.get_parameters(config={}), num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Оценка модели на локальных тестовых данных
        
        Args:
            parameters: параметры модели для оценки
            config: конфигурация
            
        Returns:
            loss, количество примеров, метрики (включая accuracy)
        """
        self.set_parameters(parameters)
        loss, accuracy, num_examples = self.test()
        
        return float(loss), num_examples, {"accuracy": float(accuracy)}
    
    def train(self, epochs: int, lr: float, mu: float = 0.0, 
              proximal: bool = False) -> Tuple[int, Dict]:
        """
        Локальное обучение модели
        
        Args:
            epochs: количество эпох
            lr: learning rate
            mu: проксимальный параметр для FedProx
            proximal: использовать ли проксимальный терм
            
        Returns:
            Количество примеров, метрики
        """
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Сохраняем глобальные параметры для FedProx
        if proximal and self.global_params is not None:
            global_params = [torch.tensor(p).to(self.device) for p in self.global_params]
        
        num_examples = 0
        total_loss = 0.0
        
        for epoch in range(epochs):
            for batch in self.trainloader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Добавляем проксимальный терм для FedProx
                if proximal and self.global_params is not None and mu > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss += (mu / 2) * proximal_term
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                num_examples += len(images)
                total_loss += loss.item() * len(images)
        
        avg_loss = total_loss / num_examples if num_examples > 0 else 0.0
        
        # Вычисляем градиентное расхождение для метрик
        gradient_divergence = 0.0
        if proximal and self.global_params is not None:
            for local_param, global_param in zip(self.model.parameters(), global_params):
                gradient_divergence += torch.sum((local_param.cpu() - global_param.cpu()) ** 2).item()
            gradient_divergence = np.sqrt(gradient_divergence)
        
        metrics = {
            'loss': avg_loss,
            'gradient_divergence': gradient_divergence
        }
        
        return num_examples, metrics
    
    def test(self) -> Tuple[float, float, int]:
        """
        Тестирование модели
        
        Returns:
            loss, accuracy, количество примеров
        """
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.testloader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * len(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy, total


def create_client_fn(model_fn, train_loaders: List[DataLoader], 
                     test_loaders: List[DataLoader], device: torch.device):
    """
    Фабрика для создания клиентов
    
    Args:
        model_fn: функция создания модели
        train_loaders: список DataLoader для обучения
        test_loaders: список DataLoader для тестирования
        device: устройство
        
    Returns:
        Функция создания клиента по cid
    """
    def client_fn(cid: str):
        """Создать клиента по ID"""
        client_id = int(cid)
        
        # Создаём новую модель для клиента
        model = model_fn()
        
        flower_client = FlowerClient(
            cid=client_id,
            model=model,
            trainloader=train_loaders[client_id],
            testloader=test_loaders[client_id],
            device=device
        )
        
        # Конвертируем в Client для совместимости с новым API Flower
        return flower_client.to_client()
    
    return client_fn
