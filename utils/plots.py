"""
Утилиты для визуализации результатов экспериментов
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


class ExperimentVisualizer:
    """Класс для визуализации результатов федеративного обучения"""
    
    def __init__(self, plots_dir: str = "./plots"):
        """
        Args:
            plots_dir: директория для сохранения графиков
        """
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_accuracy_vs_rounds(self, experiments: Dict[str, Dict], 
                               title: str = "Accuracy vs Rounds",
                               filename: str = "accuracy_vs_rounds.png"):
        """
        График точности от раундов для нескольких экспериментов
        
        Args:
            experiments: словарь {название: {rounds: [], accuracy: []}}
            title: заголовок графика
            filename: имя файла для сохранения
        """
        plt.figure(figsize=(12, 6))
        
        for exp_name, data in experiments.items():
            plt.plot(data['rounds'], data['accuracy'], 
                    marker='o', markersize=4, linewidth=2, label=exp_name)
        
        plt.xlabel('Rounds', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {filepath}")
        return filepath
    
    def plot_loss_vs_rounds(self, experiments: Dict[str, Dict],
                           title: str = "Loss vs Rounds",
                           filename: str = "loss_vs_rounds.png"):
        """
        График функции потерь от раундов
        
        Args:
            experiments: словарь {название: {rounds: [], loss: []}}
            title: заголовок графика
            filename: имя файла для сохранения
        """
        plt.figure(figsize=(12, 6))
        
        for exp_name, data in experiments.items():
            if 'loss' in data and data['loss']:
                plt.plot(data['rounds'], data['loss'], 
                        marker='o', markersize=4, linewidth=2, label=exp_name)
        
        plt.xlabel('Rounds', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {filepath}")
        return filepath
    
    def plot_heterogeneity_impact(self, alpha_values: List[float], 
                                  accuracies: Dict[str, List[float]],
                                  title: str = "Impact of Data Heterogeneity (Alpha)",
                                  filename: str = "heterogeneity_impact.png"):
        """
        Столбчатая диаграмма влияния параметра alpha (гетерогенности)
        
        Args:
            alpha_values: список значений alpha
            accuracies: словарь {метод: [точности для каждого alpha]}
            title: заголовок
            filename: имя файла
        """
        x = np.arange(len(alpha_values))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = list(accuracies.keys())
        for i, method in enumerate(methods):
            offset = width * (i - len(methods)/2 + 0.5)
            ax.bar(x + offset, accuracies[method], width, label=method)
        
        ax.set_xlabel('Alpha (Dirichlet concentration)', fontsize=14)
        ax.set_ylabel('Final Accuracy', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a}' for a in alpha_values])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {filepath}")
        return filepath
    
    def plot_client_scaling(self, client_counts: List[int],
                           accuracies: Dict[str, List[float]],
                           title: str = "Impact of Number of Clients",
                           filename: str = "client_scaling.png"):
        """
        График влияния количества клиентов
        
        Args:
            client_counts: список количеств клиентов
            accuracies: словарь {метод: [точности]}
            title: заголовок
            filename: имя файла
        """
        plt.figure(figsize=(12, 6))
        
        for method, accs in accuracies.items():
            plt.plot(client_counts, accs, marker='o', markersize=8, 
                    linewidth=2, label=method)
        
        plt.xlabel('Number of Clients', fontsize=14)
        plt.ylabel('Final Accuracy', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {filepath}")
        return filepath
    
    def plot_heatmap(self, data: pd.DataFrame, x_col: str, y_col: str, 
                    value_col: str, title: str = "Heatmap",
                    filename: str = "heatmap.png"):
        """
        Тепловая карта (например, точность как функция mu и alpha)
        
        Args:
            data: DataFrame с данными
            x_col: колонка для оси X
            y_col: колонка для оси Y
            value_col: колонка со значениями
            title: заголовок
            filename: имя файла
        """
        # Создаём pivot таблицу
        pivot = data.pivot(index=y_col, columns=x_col, values=value_col)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', 
                   cbar_kws={'label': value_col})
        
        plt.title(title, fontsize=16)
        plt.xlabel(x_col, fontsize=14)
        plt.ylabel(y_col, fontsize=14)
        
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {filepath}")
        return filepath
    
    def plot_convergence_comparison(self, experiments: Dict[str, Dict],
                                   target_accuracy: float = 0.80,
                                   title: str = "Communication Efficiency",
                                   filename: str = "convergence_comparison.png"):
        """
        График коммуникационной эффективности (раунды до целевой точности)
        
        Args:
            experiments: словарь экспериментов
            target_accuracy: целевая точность
            title: заголовок
            filename: имя файла
        """
        methods = []
        rounds_to_target = []
        
        for exp_name, data in experiments.items():
            methods.append(exp_name)
            
            # Находим раунд достижения целевой точности
            rounds = -1
            for i, acc in enumerate(data['accuracy']):
                if acc >= target_accuracy:
                    rounds = data['rounds'][i]
                    break
            
            rounds_to_target.append(rounds if rounds > 0 else data['rounds'][-1])
        
        # Столбчатая диаграмма
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, rounds_to_target, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_ylabel('Rounds to Target Accuracy', fontsize=14)
        ax.set_title(f'{title} (Target: {target_accuracy*100:.0f}%)', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12)
        
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {filepath}")
        return filepath
    
    def plot_adaptive_mu(self, rounds: List[int], mu_values: List[float],
                        title: str = "Adaptive Mu Over Rounds",
                        filename: str = "adaptive_mu.png"):
        """
        График изменения mu в Adaptive FedProx
        
        Args:
            rounds: список раундов
            mu_values: значения mu
            title: заголовок
            filename: имя файла
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(rounds, mu_values, marker='o', markersize=4, 
                linewidth=2, color='purple')
        
        plt.xlabel('Rounds', fontsize=14)
        plt.ylabel('Mu (Proximal Parameter)', fontsize=14)
        plt.title(title, fontsize=16)
        plt.grid(True, alpha=0.3)
        
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранён: {filepath}")
        return filepath
