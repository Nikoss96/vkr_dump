"""
Пример использования модуля федеративного обучения
Демонстрирует основные возможности и API
"""

import torch
import yaml
from pathlib import Path

# Импорты модулей проекта
from models.cnn_mnist import get_model
from clients.hetero_partitioner import partition_data, print_partition_statistics
from strategies.adaptive_fedprox import create_adaptive_fedprox_strategy
from utils.metrics import MetricsLogger
from utils.plots import ExperimentVisualizer


def example_1_partitioning():
    """Пример 1: Гетерогенное партиционирование данных"""
    print("\n" + "="*60)
    print("ПРИМЕР 1: Партиционирование данных с Dirichlet")
    print("="*60 + "\n")
    
    import torchvision
    import torchvision.transforms as transforms
    
    # Загружаем MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Партиционируем данные
    client_datasets, statistics = partition_data(
        dataset=train_dataset,
        num_clients=5,
        alpha=0.5,
        num_classes=10,
        seed=42
    )
    
    # Выводим статистику
    print_partition_statistics(statistics)
    
    print(f"Создано {len(client_datasets)} клиентских датасетов")
    print(f"Размеры: {[len(ds) for ds in client_datasets]}")


def example_2_model():
    """Пример 2: Создание и проверка модели"""
    print("\n" + "="*60)
    print("ПРИМЕР 2: Создание модели")
    print("="*60 + "\n")
    
    # Создаём модель
    model = get_model(num_classes=10)
    print(f"Модель: {model.__class__.__name__}")
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # Тестовый forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Размер выхода: {output.shape}")


def example_3_metrics():
    """Пример 3: Логирование метрик"""
    print("\n" + "="*60)
    print("ПРИМЕР 3: Работа с метриками")
    print("="*60 + "\n")
    
    # Создаём логгер
    logger = MetricsLogger("example_experiment")
    
    # Симулируем несколько раундов
    import random
    for round_num in range(1, 11):
        accuracy = 0.5 + (round_num / 10) * 0.4 + random.uniform(-0.05, 0.05)
        loss = 2.0 - (round_num / 10) * 1.5 + random.uniform(-0.1, 0.1)
        
        logger.log_round(
            round_num=round_num,
            accuracy=accuracy,
            loss=loss,
            num_clients=10
        )
    
    # Выводим сводку
    logger.print_summary()
    
    # Получаем финальные метрики
    final = logger.get_final_metrics()
    print("\nФинальные метрики:")
    for key, value in final.items():
        print(f"  {key}: {value}")


def example_4_visualization():
    """Пример 4: Визуализация результатов"""
    print("\n" + "="*60)
    print("ПРИМЕР 4: Визуализация")
    print("="*60 + "\n")
    
    # Создаём визуализатор
    visualizer = ExperimentVisualizer(plots_dir="./example_plots")
    
    # Симулируем данные экспериментов
    import numpy as np
    
    rounds = list(range(1, 21))
    
    experiments = {
        'FedAvg': {
            'rounds': rounds,
            'accuracy': [0.5 + 0.3 * (1 - np.exp(-r/10)) + np.random.uniform(-0.02, 0.02) 
                        for r in rounds],
            'loss': [2.0 * np.exp(-r/15) + 0.5 for r in rounds]
        },
        'FedProx': {
            'rounds': rounds,
            'accuracy': [0.5 + 0.35 * (1 - np.exp(-r/10)) + np.random.uniform(-0.02, 0.02) 
                        for r in rounds],
            'loss': [2.0 * np.exp(-r/13) + 0.4 for r in rounds]
        },
        'Adaptive FedProx': {
            'rounds': rounds,
            'accuracy': [0.5 + 0.4 * (1 - np.exp(-r/10)) + np.random.uniform(-0.02, 0.02) 
                        for r in rounds],
            'loss': [2.0 * np.exp(-r/11) + 0.3 for r in rounds]
        }
    }
    
    # Создаём графики
    visualizer.plot_accuracy_vs_rounds(
        experiments,
        title="Example: Accuracy Comparison",
        filename="example_accuracy.png"
    )
    
    visualizer.plot_loss_vs_rounds(
        experiments,
        title="Example: Loss Comparison",
        filename="example_loss.png"
    )
    
    print("Графики созданы в ./example_plots/")


def example_5_config():
    """Пример 5: Работа с конфигурацией"""
    print("\n" + "="*60)
    print("ПРИМЕР 5: Работа с конфигурацией")
    print("="*60 + "\n")
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("Основные параметры:")
    print(f"  Seed: {config['general']['seed']}")
    print(f"  Rounds: {config['general']['rounds']}")
    print(f"  Clients: {config['general']['num_clients']}")
    print(f"  Local epochs: {config['general']['local_epochs']}")
    print(f"  Batch size: {config['general']['batch_size']}")
    
    print("\nПараметры гетерогенности:")
    print(f"  Alpha: {config['heterogeneity']['alpha']}")
    
    print("\nПараметры Adaptive FedProx:")
    print(f"  μ₀: {config['adaptive_fedprox']['mu0']}")
    print(f"  min_μ: {config['adaptive_fedprox']['min_mu']}")
    print(f"  Стратегия: {config['adaptive_fedprox']['decay_strategy']}")


def main():
    """Запуск всех примеров"""
    print("\n" + "="*60)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ МОДУЛЯ FL")
    print("="*60)
    
    # Создаём необходимые директории
    Path("./example_plots").mkdir(exist_ok=True)
    Path("./results").mkdir(exist_ok=True)
    
    try:
        # Запускаем примеры
        example_1_partitioning()
        example_2_model()
        example_3_metrics()
        example_4_visualization()
        example_5_config()
        
        print("\n" + "="*60)
        print("ВСЕ ПРИМЕРЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
