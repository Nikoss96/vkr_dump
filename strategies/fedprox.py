"""
FedProx алгоритм с проксимальным термом
Добавляет штраф за отклонение от глобальной модели
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from strategies.fedavg import FedAvgStrategy


class FedProxStrategy(FedAvgStrategy):
    """
    FedProx стратегия с проксимальным термом
    
    Добавляет к функции потерь:
    L_local = L_original + (mu / 2) * ||w_local - w_global||²
    
    где mu - проксимальный параметр (штраф за отклонение)
    """
    
    def __init__(
        self,
        mu: float = 0.01,
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
        """
        Args:
            mu: проксимальный параметр (чем больше, тем сильнее штраф)
        """
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
        
        self.mu = mu
        
        # Дополнительные метрики для FedProx
        self.metrics_history['mu'] = []
        self.metrics_history['gradient_divergence'] = []
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Конфигурация обучения с добавлением mu параметра"""
        
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Добавляем mu в конфигурацию для клиентов
        config['mu'] = self.mu
        config['proximal'] = True
        
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Выбираем клиентов
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Логируем mu
        self.metrics_history['mu'].append(self.mu)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Агрегация с расчётом gradient divergence"""
        
        # Стандартная агрегация
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Вычисляем среднюю дивергенцию градиентов (если она есть в metrics)
        divergences = [
            res.metrics.get('gradient_divergence', 0.0) 
            for _, res in results
        ]
        
        if divergences:
            avg_divergence = np.mean(divergences)
            self.metrics_history['gradient_divergence'].append(avg_divergence)
        
        return aggregated_parameters, aggregated_metrics


def create_fedprox_strategy(
    mu: float,
    fraction_fit: float,
    min_fit_clients: int,
    evaluate_fn: Optional[callable] = None,
    on_fit_config_fn: Optional[callable] = None,
    initial_parameters: Optional[Parameters] = None,
) -> FedProxStrategy:
    """Фабричная функция для создания FedProx стратегии"""
    
    return FedProxStrategy(
        mu=mu,
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_fit_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
    )
