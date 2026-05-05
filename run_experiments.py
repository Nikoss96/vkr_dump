"""
Главный скрипт для запуска экспериментов федеративного обучения
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import flwr as fl
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Импорты модулей проекта
from models import cnn_mnist, resnet_cifar
from clients.hetero_partitioner import partition_data, print_partition_statistics
from clients.flower_client import create_client_fn
from strategies.fedavg import FedAvgStrategy
from strategies.fedprox import create_fedprox_strategy
from strategies.adaptive_fedprox import create_adaptive_fedprox_strategy
from utils.metrics import MetricsLogger, compare_experiments, save_comparison, print_comparison_table
from utils.plots import ExperimentVisualizer
from flwr.common import ndarrays_to_parameters


def load_config(config_path: str = "config.yaml") -> Dict:
    """Загрузка конфигурации из YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Установка seed для воспроизводимости"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name: str, data_dir: str = "./data"):
    """
    Загрузка датасета
    
    Args:
        dataset_name: 'mnist' или 'cifar10'
        data_dir: директория для данных
        
    Returns:
        train_dataset, test_dataset, num_classes
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_path, train=False, download=True, transform=transform
        )
        num_classes = 10
        
    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test
        )
        num_classes = 10
    else:
        raise ValueError(f"Неподдерживаемый датасет: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes


def create_dataloaders(client_datasets: List, test_dataset, batch_size: int):
    """
    Создание DataLoader для клиентов
    
    Args:
        client_datasets: список датасетов клиентов
        test_dataset: тестовый датасет
        batch_size: размер батча
        
    Returns:
        train_loaders, test_loaders (на test_dataset для каждого клиента)
    """
    train_loaders = []
    test_loaders = []
    
    for client_dataset in client_datasets:
        train_loader = DataLoader(
            client_dataset, batch_size=batch_size, shuffle=True
        )
        train_loaders.append(train_loader)
        
        # Для тестирования используем глобальный тестовый датасет
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


def get_model_fn(dataset_name: str, num_classes: int):
    """Получить функцию создания модели"""
    if dataset_name.lower() == 'mnist':
        return lambda: cnn_mnist.get_model(num_classes)
    elif dataset_name.lower() == 'cifar10':
        return lambda: resnet_cifar.get_model(num_classes)
    else:
        raise ValueError(f"Неподдерживаемый датасет: {dataset_name}")


def get_initial_parameters(model_fn):
    """Получить начальные параметры модели"""
    model = model_fn()
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])


def create_evaluate_fn(model_fn, test_dataset, device, batch_size=32):
    """Создать функцию оценки для сервера"""
    
    def evaluate(server_round: int, parameters, config):
        """Функция оценки глобальной модели"""
        model = model_fn()
        model.to(device)
        
        # Устанавливаем параметры
        from collections import OrderedDict
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Оцениваем на тестовом датасете
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * len(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate


def create_fit_config_fn(local_epochs: int, learning_rate: float = 0.01):
    """Создать функцию конфигурации для обучения"""
    
    def fit_config(server_round: int):
        """Конфигурация для клиентов на каждом раунде"""
        config = {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "server_round": server_round,
        }
        return config
    
    return fit_config


def run_federated_experiment(
    strategy_name: str,
    config: Dict,
    dataset_name: str,
    alpha: float,
    num_clients: int,
    device: torch.device,
    **strategy_kwargs
) -> MetricsLogger:
    """
    Запуск федеративного эксперимента
    
    Args:
        strategy_name: имя стратегии ('fedavg', 'fedprox', 'adaptive_fedprox')
        config: конфигурация
        dataset_name: имя датасета
        alpha: параметр Dirichlet
        num_clients: количество клиентов
        device: устройство
        **strategy_kwargs: дополнительные параметры для стратегии
        
    Returns:
        MetricsLogger с результатами
    """
    print(f"\n{'='*80}")
    print(f"Запуск эксперимента: {strategy_name}")
    print(f"Датасет: {dataset_name}, Alpha: {alpha}, Клиенты: {num_clients}")
    print(f"{'='*80}\n")
    
    # Загружаем датасет
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name)
    
    # Партиционируем данные
    client_datasets, statistics = partition_data(
        train_dataset, num_clients, alpha, num_classes, seed=config['general']['seed']
    )
    print_partition_statistics(statistics)
    
    # Создаём DataLoaders
    batch_size = config['general']['batch_size']
    train_loaders, test_loaders = create_dataloaders(
        client_datasets, test_dataset, batch_size
    )
    
    # Получаем функцию создания модели
    model_fn = get_model_fn(dataset_name, num_classes)
    
    # Создаём клиентскую функцию
    client_fn = create_client_fn(model_fn, train_loaders, test_loaders, device)
    
    # Получаем начальные параметры
    initial_parameters = get_initial_parameters(model_fn)
    
    # Создаём функцию оценки
    evaluate_fn = create_evaluate_fn(model_fn, test_dataset, device, batch_size)
    
    # Создаём функцию конфигурации
    fit_config_fn = create_fit_config_fn(config['general']['local_epochs'])
    
    # Создаём стратегию
    fraction_fit = config['general']['fraction_fit']
    min_fit_clients = max(2, int(num_clients * fraction_fit))
    rounds = config['general']['rounds']
    
    if strategy_name == 'fedavg':
        strategy = FedAvgStrategy(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config_fn,
            initial_parameters=initial_parameters,
        )
    elif strategy_name == 'fedprox':
        mu = strategy_kwargs.get('mu', 0.1)
        strategy = create_fedprox_strategy(
            mu=mu,
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config_fn,
            initial_parameters=initial_parameters,
        )
    elif strategy_name == 'adaptive_fedprox':
        mu0 = strategy_kwargs.get('mu0', 0.5)
        min_mu = strategy_kwargs.get('min_mu', 0.01)
        decay_strategy = strategy_kwargs.get('decay_strategy', 'linear')
        
        strategy = create_adaptive_fedprox_strategy(
            mu0=mu0,
            min_mu=min_mu,
            decay_strategy=decay_strategy,
            total_rounds=rounds,
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config_fn,
            initial_parameters=initial_parameters,
        )
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy_name}")
    
    # Запуск симуляции
    print(f"Начало федеративного обучения ({rounds} раундов)...")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Собираем метрики
    exp_name = f"{strategy_name}_{dataset_name}_alpha{alpha}_clients{num_clients}"
    logger = MetricsLogger(exp_name)
    
    # Заполняем логгер данными из истории
    # Flower возвращает словарь losses_centralized и metrics_centralized
    if hasattr(history, 'losses_centralized') and history.losses_centralized:
        # Формат: [(round_num, loss), ...]
        for round_num, loss in history.losses_centralized:
            # Находим соответствующую accuracy
            accuracy = 0.0
            if hasattr(history, 'metrics_centralized') and 'accuracy' in history.metrics_centralized:
                # Ищем accuracy для этого раунда
                acc_list = history.metrics_centralized['accuracy']
                for acc_round, acc_val in acc_list:
                    if acc_round == round_num:
                        accuracy = acc_val
                        break
            
            logger.log_round(
                round_num=round_num,
                accuracy=accuracy,
                loss=loss,
                num_clients=min_fit_clients
            )
    
    # Если history пустой, используем metrics из стратегии
    if not logger.metrics['rounds'] and hasattr(strategy, 'metrics_history'):
        hist = strategy.metrics_history
        for i, round_num in enumerate(hist.get('round', [])):
            logger.log_round(
                round_num=round_num,
                accuracy=hist.get('accuracy', [0])[i] if i < len(hist.get('accuracy', [])) else 0.0,
                loss=hist.get('loss', [0])[i] if i < len(hist.get('loss', [])) else 0.0,
                num_clients=hist.get('num_clients', [0])[i] if i < len(hist.get('num_clients', [])) else 0
            )
    
    # Сохраняем метрики
    logger.save_to_csv()
    logger.save_to_json()
    logger.print_summary()
    
    return logger


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Эксперименты федеративного обучения')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--all', action='store_true', 
                       help='Запустить все эксперименты')
    parser.add_argument('--compare', action='store_true',
                       help='Сравнение методов')
    parser.add_argument('--dataset', type=str, default='mnist',
                       help='Датасет (mnist/cifar10)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Alpha для Dirichlet')
    parser.add_argument('--heterogeneity', action='store_true',
                       help='Анализ гетерогенности')
    parser.add_argument('--scale', action='store_true',
                       help='Анализ масштабирования')
    parser.add_argument('--clients', type=str, default='10',
                       help='Количество клиентов (через запятую для scale)')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Устанавливаем seed
    set_seed(config['general']['seed'])
    
    # Определяем устройство
    device_name = config['general'].get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Создаём директории
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    visualizer = ExperimentVisualizer()
    
    # Определяем датасет
    dataset = args.dataset if not args.all else 'mnist'
    
    if args.all or args.compare:
        print("\n" + "="*80)
        print("СЦЕНАРИЙ 1: Сравнение FedAvg vs FedProx vs Adaptive FedProx")
        print("="*80 + "\n")
        
        alpha = args.alpha if not args.all else 0.5
        num_clients = int(args.clients.split(',')[0]) if not args.all else config['general']['num_clients']
        
        loggers = []
        
        # FedAvg
        logger_fedavg = run_federated_experiment(
            'fedavg', config, dataset, alpha, num_clients, device
        )
        loggers.append(logger_fedavg)
        
        # FedProx
        logger_fedprox = run_federated_experiment(
            'fedprox', config, dataset, alpha, num_clients, device,
            mu=config['fedprox']['mu']
        )
        loggers.append(logger_fedprox)
        
        # Adaptive FedProx
        logger_adaptive = run_federated_experiment(
            'adaptive_fedprox', config, dataset, alpha, num_clients, device,
            mu0=config['adaptive_fedprox']['mu0'],
            min_mu=config['adaptive_fedprox']['min_mu'],
            decay_strategy=config['adaptive_fedprox']['decay_strategy']
        )
        loggers.append(logger_adaptive)
        
        # Сравнение
        comparison_df = compare_experiments(loggers)
        save_comparison(comparison_df, f"comparison_{dataset}_alpha{alpha}.csv")
        print_comparison_table(comparison_df)
        
        # Визуализация
        experiments_data = {
            'FedAvg': {
                'rounds': logger_fedavg.metrics['rounds'],
                'accuracy': logger_fedavg.metrics['global_accuracy'],
                'loss': logger_fedavg.metrics['global_loss']
            },
            'FedProx': {
                'rounds': logger_fedprox.metrics['rounds'],
                'accuracy': logger_fedprox.metrics['global_accuracy'],
                'loss': logger_fedprox.metrics['global_loss']
            },
            'Adaptive FedProx': {
                'rounds': logger_adaptive.metrics['rounds'],
                'accuracy': logger_adaptive.metrics['global_accuracy'],
                'loss': logger_adaptive.metrics['global_loss']
            }
        }
        
        visualizer.plot_accuracy_vs_rounds(
            experiments_data,
            title=f"Accuracy vs Rounds ({dataset.upper()}, alpha={alpha})",
            filename=f"accuracy_comparison_{dataset}_alpha{alpha}.png"
        )
        
        visualizer.plot_loss_vs_rounds(
            experiments_data,
            title=f"Loss vs Rounds ({dataset.upper()}, alpha={alpha})",
            filename=f"loss_comparison_{dataset}_alpha{alpha}.png"
        )
        
        visualizer.plot_convergence_comparison(
            experiments_data,
            target_accuracy=0.80,
            filename=f"convergence_comparison_{dataset}_alpha{alpha}.png"
        )
    
    if args.all or args.heterogeneity:
        print("\n" + "="*80)
        print("СЦЕНАРИЙ 2: Влияние гетерогенности данных")
        print("="*80 + "\n")
        
        alpha_values = config['experiments']['alpha_values']
        num_clients = config['general']['num_clients']
        
        results = {
            'FedAvg': [],
            'FedProx': [],
            'Adaptive FedProx': []
        }
        
        for alpha in alpha_values:
            print(f"\n--- Тестирование с alpha = {alpha} ---\n")
            
            # FedAvg
            logger = run_federated_experiment(
                'fedavg', config, dataset, alpha, num_clients, device
            )
            results['FedAvg'].append(logger.get_final_metrics()['final_accuracy'])
            
            # FedProx
            logger = run_federated_experiment(
                'fedprox', config, dataset, alpha, num_clients, device,
                mu=config['fedprox']['mu']
            )
            results['FedProx'].append(logger.get_final_metrics()['final_accuracy'])
            
            # Adaptive FedProx
            logger = run_federated_experiment(
                'adaptive_fedprox', config, dataset, alpha, num_clients, device,
                mu0=config['adaptive_fedprox']['mu0'],
                min_mu=config['adaptive_fedprox']['min_mu'],
                decay_strategy=config['adaptive_fedprox']['decay_strategy']
            )
            results['Adaptive FedProx'].append(logger.get_final_metrics()['final_accuracy'])
        
        # Визуализация
        visualizer.plot_heterogeneity_impact(
            alpha_values, results,
            title=f"Impact of Data Heterogeneity ({dataset.upper()})",
            filename=f"heterogeneity_impact_{dataset}.png"
        )
    
    if args.all or args.scale:
        print("\n" + "="*80)
        print("СЦЕНАРИЙ 3: Влияние количества клиентов")
        print("="*80 + "\n")
        
        client_counts = config['experiments']['client_counts']
        alpha = 0.5
        
        results = {
            'FedAvg': [],
            'FedProx': [],
            'Adaptive FedProx': []
        }
        
        for num_clients in client_counts:
            print(f"\n--- Тестирование с {num_clients} клиентами ---\n")
            
            # FedAvg
            logger = run_federated_experiment(
                'fedavg', config, dataset, alpha, num_clients, device
            )
            results['FedAvg'].append(logger.get_final_metrics()['final_accuracy'])
            
            # FedProx
            logger = run_federated_experiment(
                'fedprox', config, dataset, alpha, num_clients, device,
                mu=config['fedprox']['mu']
            )
            results['FedProx'].append(logger.get_final_metrics()['final_accuracy'])
            
            # Adaptive FedProx
            logger = run_federated_experiment(
                'adaptive_fedprox', config, dataset, alpha, num_clients, device,
                mu0=config['adaptive_fedprox']['mu0'],
                min_mu=config['adaptive_fedprox']['min_mu'],
                decay_strategy=config['adaptive_fedprox']['decay_strategy']
            )
            results['Adaptive FedProx'].append(logger.get_final_metrics()['final_accuracy'])
        
        # Визуализация
        visualizer.plot_client_scaling(
            client_counts, results,
            title=f"Impact of Number of Clients ({dataset.upper()})",
            filename=f"client_scaling_{dataset}.png"
        )
    
    print("\n" + "="*80)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("="*80)
    print(f"Результаты сохранены в ./results/")
    print(f"Графики сохранены в ./plots/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
