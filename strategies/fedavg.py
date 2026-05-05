"""
Базовый FedAvg алгоритм
Усреднение весов моделей с весами по размеру датасета клиента
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Стандартная стратегия FedAvg с логированием метрик
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        
        # Метрики для логирования
        self.metrics_history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'num_clients': []
        }
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Агрегация результатов обучения от клиентов"""
        
        # Вызываем стандартную агрегацию FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Логируем количество клиентов
        self.metrics_history['round'].append(server_round)
        self.metrics_history['num_clients'].append(len(results))
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Агрегация результатов оценки от клиентов"""
        
        if not results:
            return None, {}
        
        # Взвешенное усреднение метрик
        total_samples = sum([r.num_examples for _, r in results])
        
        if total_samples == 0:
            return None, {}
        
        weighted_loss = sum([r.loss * r.num_examples for _, r in results]) / total_samples
        
        # Если есть accuracy в метриках
        accuracies = [r.metrics.get('accuracy', 0) for _, r in results if 'accuracy' in r.metrics]
        if accuracies:
            weighted_accuracy = sum([
                r.metrics['accuracy'] * r.num_examples 
                for _, r in results if 'accuracy' in r.metrics
            ]) / total_samples
        else:
            weighted_accuracy = 0.0
        
        # Сохраняем в историю
        if server_round not in self.metrics_history['round']:
            self.metrics_history['round'].append(server_round)
            self.metrics_history['loss'].append(weighted_loss)
            self.metrics_history['accuracy'].append(weighted_accuracy)
            self.metrics_history['num_clients'].append(len(results))
        
        metrics = {
            'accuracy': weighted_accuracy,
            'num_examples': total_samples
        }
        
        return weighted_loss, metrics
    
    def get_metrics_history(self) -> Dict:
        """Получить историю метрик"""
        return self.metrics_history
