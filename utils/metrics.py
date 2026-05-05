"""
Утилиты для сбора и обработки метрик
"""
import json
import csv
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd


class MetricsLogger:
    """Класс для логирования метрик экспериментов"""
    
    def __init__(self, experiment_name: str, results_dir: str = "./results"):
        """
        Args:
            experiment_name: название эксперимента
            results_dir: директория для сохранения результатов
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'experiment_name': experiment_name,
            'rounds': [],
            'global_accuracy': [],
            'global_loss': [],
            'comm_cost': [],
            'num_clients': [],
        }
    
    def log_round(self, round_num: int, accuracy: float, loss: float, 
                  num_clients: int, model_size: int = 0):
        """
        Логирование метрик раунда
        
        Args:
            round_num: номер раунда
            accuracy: точность
            loss: функция потерь
            num_clients: количество клиентов в раунде
            model_size: размер модели в байтах (для оценки comm_cost)
        """
        self.metrics['rounds'].append(round_num)
        self.metrics['global_accuracy'].append(accuracy)
        self.metrics['global_loss'].append(loss)
        self.metrics['num_clients'].append(num_clients)
        
        # Оценка коммуникационных затрат: размер модели * 2 (отправка + получение)
        comm_cost = model_size * 2 if model_size > 0 else 0
        self.metrics['comm_cost'].append(comm_cost)
    
    def add_custom_metric(self, metric_name: str, value: Any):
        """Добавить кастомную метрику"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def save_to_csv(self, filename: str = None):
        """Сохранить метрики в CSV"""
        if filename is None:
            filename = f"{self.experiment_name}_metrics.csv"
        
        filepath = self.results_dir / filename
        
        # Создаём DataFrame
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)
        
        print(f"Метрики сохранены в {filepath}")
        return filepath
    
    def save_to_json(self, filename: str = None):
        """Сохранить метрики в JSON"""
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Метрики сохранены в {filepath}")
        return filepath
    
    def get_final_metrics(self) -> Dict:
        """Получить финальные метрики эксперимента"""
        if not self.metrics['rounds']:
            return {}
        
        final_metrics = {
            'experiment_name': self.experiment_name,
            'total_rounds': len(self.metrics['rounds']),
            'final_accuracy': self.metrics['global_accuracy'][-1] if self.metrics['global_accuracy'] else 0,
            'final_loss': self.metrics['global_loss'][-1] if self.metrics['global_loss'] else 0,
            'best_accuracy': max(self.metrics['global_accuracy']) if self.metrics['global_accuracy'] else 0,
            'best_accuracy_round': self.metrics['rounds'][
                self.metrics['global_accuracy'].index(max(self.metrics['global_accuracy']))
            ] if self.metrics['global_accuracy'] else 0,
        }
        
        # Раунды до достижения целевой точности
        for target in [0.70, 0.75, 0.80, 0.85, 0.90]:
            rounds_to_target = self._rounds_to_accuracy(target)
            final_metrics[f'rounds_to_{int(target*100)}'] = rounds_to_target
        
        return final_metrics
    
    def _rounds_to_accuracy(self, target_accuracy: float) -> int:
        """Вычислить количество раундов до достижения целевой точности"""
        for i, acc in enumerate(self.metrics['global_accuracy']):
            if acc >= target_accuracy:
                return self.metrics['rounds'][i]
        return -1  # Не достигнуто
    
    def print_summary(self):
        """Вывести краткую сводку метрик"""
        final = self.get_final_metrics()
        
        print(f"\n{'='*60}")
        print(f"Сводка эксперимента: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Всего раундов: {final['total_rounds']}")
        print(f"Финальная точность: {final['final_accuracy']:.4f}")
        print(f"Лучшая точность: {final['best_accuracy']:.4f} (раунд {final['best_accuracy_round']})")
        
        print(f"\nРаунды до достижения целевой точности:")
        for target in [70, 75, 80, 85, 90]:
            key = f'rounds_to_{target}'
            if key in final:
                rounds = final[key]
                if rounds > 0:
                    print(f"  {target}%: {rounds} раундов")
                else:
                    print(f"  {target}%: не достигнуто")
        
        print(f"{'='*60}\n")


def compare_experiments(loggers: List[MetricsLogger]) -> pd.DataFrame:
    """
    Сравнить несколько экспериментов
    
    Args:
        loggers: список MetricsLogger для сравнения
        
    Returns:
        DataFrame со сравнением
    """
    comparison_data = []
    
    for logger in loggers:
        final = logger.get_final_metrics()
        comparison_data.append(final)
    
    df = pd.DataFrame(comparison_data)
    
    return df


def save_comparison(comparison_df: pd.DataFrame, filename: str, results_dir: str = "./results"):
    """Сохранить сравнение экспериментов"""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    filepath = results_path / filename
    comparison_df.to_csv(filepath, index=False)
    
    print(f"Сравнение экспериментов сохранено в {filepath}")
    return filepath


def print_comparison_table(comparison_df: pd.DataFrame):
    """Вывести красивую таблицу сравнения"""
    print(f"\n{'='*80}")
    print("Сравнение экспериментов")
    print(f"{'='*80}")
    
    # Выбираем ключевые колонки для отображения
    display_cols = [
        'experiment_name', 
        'final_accuracy', 
        'best_accuracy',
        'rounds_to_80',
        'rounds_to_85'
    ]
    
    # Фильтруем доступные колонки
    available_cols = [col for col in display_cols if col in comparison_df.columns]
    
    display_df = comparison_df[available_cols].copy()
    
    # Форматируем числа
    if 'final_accuracy' in display_df.columns:
        display_df['final_accuracy'] = display_df['final_accuracy'].apply(lambda x: f"{x:.4f}")
    if 'best_accuracy' in display_df.columns:
        display_df['best_accuracy'] = display_df['best_accuracy'].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    print(f"{'='*80}\n")
