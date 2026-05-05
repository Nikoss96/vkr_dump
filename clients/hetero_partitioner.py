"""
Гетерогенное партиционирование данных с использованием распределения Дирихле
"""
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, Subset
import torch


class DirichletPartitioner:
    """
    Класс для создания non-IID партиций данных с использованием распределения Дирихле
    
    Параметр alpha контролирует степень гетерогенности:
    - alpha = 10.0: почти IID (равномерное распределение)
    - alpha = 1.0: умеренная гетерогенность
    - alpha = 0.5: сильная гетерогенность
    - alpha = 0.1: экстремальная гетерогенность (почти один класс на клиента)
    """
    
    def __init__(self, num_clients: int, alpha: float, seed: int = 42):
        """
        Args:
            num_clients: количество клиентов
            alpha: параметр концентрации Дирихле
            seed: seed для воспроизводимости
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        np.random.seed(seed)
    
    def partition(self, dataset: Dataset, num_classes: int) -> List[List[int]]:
        """
        Партиционирование датасета по клиентам с использованием Dirichlet
        
        Args:
            dataset: датасет для партиционирования
            num_classes: количество классов в датасете
            
        Returns:
            Список индексов для каждого клиента
        """
        # Получаем метки всех данных
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            # Fallback: извлекаем метки вручную
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        # Индексы для каждого класса
        class_indices = {k: np.where(labels == k)[0] for k in range(num_classes)}
        
        # Инициализация списков индексов для клиентов
        client_indices = [[] for _ in range(self.num_clients)]
        
        # Для каждого класса распределяем данные между клиентами
        for k in range(num_classes):
            # Генерируем пропорции из распределения Дирихле
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Перемешиваем индексы класса
            class_idx = class_indices[k]
            np.random.shuffle(class_idx)
            
            # Разбиваем индексы класса согласно пропорциям
            proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            splits = np.split(class_idx, proportions)
            
            # Распределяем по клиентам
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())
        
        # Перемешиваем индексы внутри каждого клиента
        for client_id in range(self.num_clients):
            np.random.shuffle(client_indices[client_id])
        
        return client_indices
    
    def get_statistics(self, client_indices: List[List[int]], dataset: Dataset, 
                      num_classes: int) -> Dict:
        """
        Получить статистику распределения данных по клиентам
        
        Args:
            client_indices: индексы данных для каждого клиента
            dataset: датасет
            num_classes: количество классов
            
        Returns:
            Словарь со статистикой
        """
        # Получаем метки
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        statistics = {
            'num_clients': self.num_clients,
            'alpha': self.alpha,
            'total_samples': len(dataset),
            'clients': []
        }
        
        for client_id, indices in enumerate(client_indices):
            client_labels = labels[indices]
            class_distribution = {
                int(k): int(np.sum(client_labels == k)) 
                for k in range(num_classes)
            }
            
            statistics['clients'].append({
                'client_id': client_id,
                'num_samples': len(indices),
                'class_distribution': class_distribution
            })
        
        return statistics
    
    def create_client_datasets(self, dataset: Dataset, 
                              client_indices: List[List[int]]) -> List[Subset]:
        """
        Создать отдельные датасеты для каждого клиента
        
        Args:
            dataset: исходный датасет
            client_indices: индексы для каждого клиента
            
        Returns:
            Список Subset датасетов для каждого клиента
        """
        return [Subset(dataset, indices) for indices in client_indices]


def partition_data(dataset: Dataset, num_clients: int, alpha: float, 
                   num_classes: int, seed: int = 42) -> Tuple[List[Subset], Dict]:
    """
    Вспомогательная функция для быстрого партиционирования
    
    Args:
        dataset: датасет для партиционирования
        num_clients: количество клиентов
        alpha: параметр Дирихле
        num_classes: количество классов
        seed: random seed
        
    Returns:
        Кортеж (список клиентских датасетов, статистика)
    """
    partitioner = DirichletPartitioner(num_clients, alpha, seed)
    client_indices = partitioner.partition(dataset, num_classes)
    statistics = partitioner.get_statistics(client_indices, dataset, num_classes)
    client_datasets = partitioner.create_client_datasets(dataset, client_indices)
    
    return client_datasets, statistics


def print_partition_statistics(statistics: Dict):
    """Печать статистики партиционирования"""
    print(f"\n{'='*60}")
    print(f"Статистика партиционирования данных")
    print(f"{'='*60}")
    print(f"Количество клиентов: {statistics['num_clients']}")
    print(f"Alpha (Dirichlet): {statistics['alpha']}")
    print(f"Всего примеров: {statistics['total_samples']}")
    print(f"\nРаспределение по клиентам:")
    
    for client in statistics['clients']:
        cid = client['client_id']
        n_samples = client['num_samples']
        dist = client['class_distribution']
        
        # Форматируем распределение классов
        dist_str = ", ".join([f"{k}:{v}" for k, v in dist.items() if v > 0])
        
        print(f"  Клиент {cid:2d}: {n_samples:5d} примеров | {dist_str}")
    
    print(f"{'='*60}\n")
