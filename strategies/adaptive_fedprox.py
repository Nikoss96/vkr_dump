"""
Adaptive FedProx - предлагаемый метод с динамической адаптацией mu
ЭЛЕМЕНТ НАУЧНОЙ НОВИЗНЫ
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from strategies.fedprox import FedProxStrategy


class AdaptiveFedProxStrategy(FedProxStrategy):
    """
    Адаптивная версия FedProx с динамической настройкой параметра mu
    
    НАУЧНАЯ НОВИЗНА:
    Вместо фиксированного mu используется адаптивная стратегия:
    1. На ранних раундах - сильный штраф (стабильность)
    2. На поздних раундах - слабый штраф (точная настройка)
    3. Учитывается градиентное расхождение между моделями
    
    Доступные стратегии адаптации:
    - 'linear': линейное уменьшение mu со временем
    - 'exp': экспоненциальное уменьшение
    - 'gradient_based': адаптация на основе величины расхождения градиентов
    """
    
    def __init__(
        self,
        mu0: float = 0.5,
        min_mu: float = 0.01,
        decay_strategy: str = 'linear',
        total_rounds: int = 50,
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
            mu0: начальное значение mu (должно быть достаточно большим для стабильности)
            min_mu: минимальное значение mu (не опускаться ниже)
            decay_strategy: стратегия уменьшения ('linear', 'exp', 'gradient_based')
            total_rounds: общее количество раундов для планирования decay
        """
        # Инициализируем с начальным mu
        super().__init__(
            mu=mu0,
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
        
        self.mu0 = mu0
        self.min_mu = min_mu
        self.decay_strategy = decay_strategy
        self.total_rounds = total_rounds
        
        # История градиентных расхождений для gradient_based стратегии
        self.divergence_history = []
        
        print(f"\n{'='*60}")
        print(f"Инициализация Adaptive FedProx")
        print(f"{'='*60}")
        print(f"  Начальный mu (mu0): {self.mu0}")
        print(f"  Минимальный mu: {self.min_mu}")
        print(f"  Стратегия адаптации: {self.decay_strategy}")
        print(f"  Всего раундов: {self.total_rounds}")
        print(f"{'='*60}\n")
    
    def _compute_adaptive_mu(self, server_round: int, 
                            avg_divergence: Optional[float] = None) -> float:
        """
        Вычисление адаптивного значения mu
        
        Args:
            server_round: текущий раунд
            avg_divergence: среднее градиентное расхождение (для gradient_based)
            
        Returns:
            Новое значение mu
        """
        if self.decay_strategy == 'linear':
            # Линейное уменьшение: mu = mu0 * (1 - round/T_max)
            decay_factor = max(0.0, 1.0 - server_round / self.total_rounds)
            mu = self.mu0 * decay_factor
            
        elif self.decay_strategy == 'exp':
            # Экспоненциальное уменьшение: mu = mu0 * exp(-k * round/T_max)
            k = 3.0  # коэффициент скорости убывания
            decay_factor = np.exp(-k * server_round / self.total_rounds)
            mu = self.mu0 * decay_factor
            
        elif self.decay_strategy == 'gradient_based':
            # Адаптация на основе градиентного расхождения
            if avg_divergence is not None and len(self.divergence_history) > 0:
                # Нормализуем текущую дивергенцию относительно истории
                max_divergence = max(self.divergence_history)
                if max_divergence > 0:
                    normalized_div = avg_divergence / max_divergence
                    # Если дивергенция большая - увеличиваем mu, если малая - уменьшаем
                    mu = self.mu0 * np.clip(normalized_div, 0.1, 1.0)
                else:
                    mu = self.mu0 * 0.5
            else:
                # На первых раундах используем линейную стратегию
                decay_factor = max(0.0, 1.0 - server_round / self.total_rounds)
                mu = self.mu0 * decay_factor
        else:
            # По умолчанию - линейная
            decay_factor = max(0.0, 1.0 - server_round / self.total_rounds)
            mu = self.mu0 * decay_factor
        
        # Применяем ограничение min_mu
        mu = max(self.min_mu, mu)
        
        return mu
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Конфигурация обучения с адаптивным mu"""
        
        # Вычисляем текущее значение mu
        avg_divergence = None
        if self.divergence_history:
            avg_divergence = self.divergence_history[-1]
        
        self.mu = self._compute_adaptive_mu(server_round, avg_divergence)
        
        # Логируем изменение mu
        if server_round % 5 == 0 or server_round == 1:
            print(f"Раунд {server_round}: адаптивный mu = {self.mu:.6f}")
        
        # Вызываем базовую конфигурацию FedProx
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Агрегация с обновлением истории дивергенций"""
        
        # Стандартная агрегация FedProx
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Обновляем историю дивергенций для gradient_based стратегии
        if self.metrics_history['gradient_divergence']:
            self.divergence_history.append(
                self.metrics_history['gradient_divergence'][-1]
            )
        
        return aggregated_parameters, aggregated_metrics


def create_adaptive_fedprox_strategy(
    mu0: float,
    min_mu: float,
    decay_strategy: str,
    total_rounds: int,
    fraction_fit: float,
    min_fit_clients: int,
    evaluate_fn: Optional[callable] = None,
    on_fit_config_fn: Optional[callable] = None,
    initial_parameters: Optional[Parameters] = None,
) -> AdaptiveFedProxStrategy:
    """Фабричная функция для создания Adaptive FedProx стратегии"""
    
    return AdaptiveFedProxStrategy(
        mu0=mu0,
        min_mu=min_mu,
        decay_strategy=decay_strategy,
        total_rounds=total_rounds,
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_fit_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
    )
