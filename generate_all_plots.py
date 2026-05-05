"""
Генерация всех графиков из результатов экспериментов
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Настройка стиля
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

results_dir = Path("results")
base_exp_01_dir = Path("base_exp_01_alpha_getero")
base_exp_05_dir = Path("base_exp_05_alpha_getero")
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

print("="*80)
print("ГЕНЕРАЦИЯ ГРАФИКОВ ИЗ ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ")
print("="*80 + "\n")

# ============================================================================
# 1. ЗАГРУЗКА ВСЕХ ДАННЫХ
# ============================================================================

def load_json_metrics(filepath):
    """Загрузка метрик из JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

# Собрать все данные
all_data = {}

# Из results/ (heterogeneity experiment с rounds=20)
for alpha in [10.0, 1.0, 0.5]:
    for method in ['fedavg', 'fedprox', 'adaptive_fedprox']:
        filename = f"{method}_mnist_alpha{alpha}_clients10_metrics.json"
        filepath = results_dir / filename
        if filepath.exists():
            data = load_json_metrics(filepath)
            key = f"{method}_alpha{alpha}"
            all_data[key] = data
            print(f"✓ Загружено: {filename}")

# Из base_exp_01/ (α=0.1, rounds=50)
for method in ['fedavg', 'fedprox', 'adaptive_fedprox']:
    filename = f"{method}_mnist_alpha0.1_clients10_metrics.json"
    filepath = base_exp_01_dir / filename
    if filepath.exists():
        data = load_json_metrics(filepath)
        key = f"{method}_alpha0.1"
        all_data[key] = data
        print(f"✓ Загружено: {filename} (50 раундов)")

# Из base_exp_05/ (α=0.5, rounds=50)
for method in ['fedavg', 'fedprox', 'adaptive_fedprox']:
    filename = f"{method}_mnist_alpha0.5_clients10_metrics.json"
    filepath = base_exp_05_dir / filename
    if filepath.exists():
        data = load_json_metrics(filepath)
        key = f"{method}_alpha0.5_50rounds"
        all_data[key] = data
        print(f"✓ Загружено: {filename} (50 раундов, старый)")

print(f"\nВсего загружено: {len(all_data)} экспериментов\n")

# ============================================================================
# 2. ГРАФИК 1: Сравнение методов при α=0.1 (высокая гетерогенность, 50 раундов)
# ============================================================================

print("Создание графика 1: Сравнение методов (α=0.1, 50 раундов)...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

methods_alpha01 = [
    ('fedavg_alpha0.1', 'FedAvg', '#3498db'),
    ('fedprox_alpha0.1', 'FedProx (μ=0.01)', '#2ecc71'),
    ('adaptive_fedprox_alpha0.1', 'Adaptive FedProx', '#e74c3c')
]

# Accuracy
for key, label, color in methods_alpha01:
    if key in all_data:
        data = all_data[key]
        ax1.plot(data['rounds'], data['global_accuracy'], 
                label=label, linewidth=2.5, marker='o', 
                markevery=5, markersize=6, color=color)

ax1.set_xlabel('Раунд обучения', fontsize=12, fontweight='bold')
ax1.set_ylabel('Точность (Accuracy)', fontsize=12, fontweight='bold')
ax1.set_title('Сравнение точности при высокой гетерогенности (α=0.1)', 
             fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# Loss
for key, label, color in methods_alpha01:
    if key in all_data:
        data = all_data[key]
        ax2.plot(data['rounds'], data['global_loss'], 
                label=label, linewidth=2.5, marker='s', 
                markevery=5, markersize=6, color=color)

ax2.set_xlabel('Раунд обучения', fontsize=12, fontweight='bold')
ax2.set_ylabel('Потери (Loss)', fontsize=12, fontweight='bold')
ax2.set_title('Сравнение потерь при высокой гетерогенности (α=0.1)', 
             fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig(plots_dir / 'comparison_alpha0.1_50rounds.png', dpi=300, bbox_inches='tight')
print(f"✓ Сохранено: plots/comparison_alpha0.1_50rounds.png\n")
plt.close()

# ============================================================================
# 3. ГРАФИК 2: Влияние гетерогенности на каждый метод
# ============================================================================

print("Создание графика 2: Влияние параметра α на точность...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods_for_heterogeneity = [
    ('fedavg', 'FedAvg', axes[0]),
    ('fedprox', 'FedProx (μ=0.01)', axes[1]),
    ('adaptive_fedprox', 'Adaptive FedProx', axes[2])
]

alpha_values = [0.1, 0.5, 1.0, 10.0]
colors_alpha = {0.1: '#e74c3c', 0.5: '#f39c12', 1.0: '#3498db', 10.0: '#2ecc71'}

for method, method_label, ax in methods_for_heterogeneity:
    for alpha in alpha_values:
        key = f"{method}_alpha{alpha}"
        if key in all_data:
            data = all_data[key]
            ax.plot(data['rounds'], data['global_accuracy'], 
                   label=f'α={alpha}', linewidth=2, 
                   color=colors_alpha[alpha], alpha=0.8)
    
    ax.set_xlabel('Раунд обучения', fontsize=11, fontweight='bold')
    ax.set_ylabel('Точность', fontsize=11, fontweight='bold')
    ax.set_title(f'{method_label}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

plt.suptitle('Влияние гетерогенности (α) на точность обучения', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(plots_dir / 'heterogeneity_impact_all_methods.png', dpi=300, bbox_inches='tight')
print(f"✓ Сохранено: plots/heterogeneity_impact_all_methods.png\n")
plt.close()

# ============================================================================
# 4. ГРАФИК 3: Финальная точность vs α для всех методов
# ============================================================================

print("Создание графика 3: Финальная точность vs α...")

fig, ax = plt.subplots(figsize=(12, 7))

methods_comparison = [
    ('fedavg', 'FedAvg', '#3498db', 'o'),
    ('fedprox', 'FedProx (μ=0.01)', '#2ecc71', 's'),
    ('adaptive_fedprox', 'Adaptive FedProx', '#e74c3c', '^')
]

for method, label, color, marker in methods_comparison:
    final_accs = []
    alphas_used = []
    
    for alpha in [0.1, 0.5, 1.0, 10.0]:
        key = f"{method}_alpha{alpha}"
        if key in all_data:
            data = all_data[key]
            final_acc = data['global_accuracy'][-1]
            final_accs.append(final_acc * 100)  # В процентах
            alphas_used.append(alpha)
    
    if final_accs:
        ax.plot(alphas_used, final_accs, label=label, 
               linewidth=3, marker=marker, markersize=12, 
               color=color, alpha=0.8)

ax.set_xlabel('Параметр гетерогенности α (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('Финальная точность (%)', fontsize=13, fontweight='bold')
ax.set_title('Влияние гетерогенности данных на качество методов FL', 
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xscale('log')
ax.set_ylim([70, 100])

# Аннотации
ax.axvspan(0.05, 0.3, alpha=0.1, color='red', label='Высокая гетерогенность')
ax.axvspan(5, 15, alpha=0.1, color='green', label='Низкая гетерогенность (почти IID)')

plt.tight_layout()
plt.savefig(plots_dir / 'final_accuracy_vs_alpha.png', dpi=300, bbox_inches='tight')
print(f"✓ Сохранено: plots/final_accuracy_vs_alpha.png\n")
plt.close()

# ============================================================================
# 5. ГРАФИК 4: Скорость сходимости (Rounds to Target Accuracy)
# ============================================================================

print("Создание графика 4: Скорость сходимости...")

def find_rounds_to_target(accuracy_list, target=0.90):
    """Найти раунд, когда точность достигла target"""
    for i, acc in enumerate(accuracy_list):
        if acc >= target:
            return i
    return len(accuracy_list)  # Не достигла

fig, ax = plt.subplots(figsize=(14, 7))

targets = [0.80, 0.85, 0.90, 0.95]
bar_width = 0.25
x_pos = np.arange(len(targets))

for i, (method, label, color, _) in enumerate(methods_comparison):
    rounds_alpha01 = []
    
    for target in targets:
        key = f"{method}_alpha0.1"
        if key in all_data:
            data = all_data[key]
            rounds = find_rounds_to_target(data['global_accuracy'], target)
            rounds_alpha01.append(rounds)
        else:
            rounds_alpha01.append(50)  # Max
    
    ax.bar(x_pos + i*bar_width, rounds_alpha01, bar_width, 
          label=label, color=color, alpha=0.8)

ax.set_xlabel('Целевая точность', fontsize=13, fontweight='bold')
ax.set_ylabel('Раундов до достижения', fontsize=13, fontweight='bold')
ax.set_title('Скорость сходимости методов (α=0.1)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels([f'{int(t*100)}%' for t in targets])
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(plots_dir / 'convergence_speed_alpha0.1.png', dpi=300, bbox_inches='tight')
print(f"✓ Сохранено: plots/convergence_speed_alpha0.1.png\n")
plt.close()

# ============================================================================
# 6. ГРАФИК 5: Стабильность обучения (volatility)
# ============================================================================

print("Создание графика 5: Стабильность обучения...")

fig, ax = plt.subplots(figsize=(12, 7))

for method, label, color, marker in methods_comparison:
    volatilities = []
    alphas_used = []
    
    for alpha in [0.1, 0.5, 1.0, 10.0]:
        key = f"{method}_alpha{alpha}"
        if key in all_data:
            data = all_data[key]
            # Вычислить волатильность как std отклонение последних 20 раундов
            if len(data['global_accuracy']) >= 20:
                last_20 = data['global_accuracy'][-20:]
                volatility = np.std(last_20) * 100  # В процентах
                volatilities.append(volatility)
                alphas_used.append(alpha)
    
    if volatilities:
        ax.plot(alphas_used, volatilities, label=label, 
               linewidth=3, marker=marker, markersize=12, 
               color=color, alpha=0.8)

ax.set_xlabel('Параметр гетерогенности α (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('Волатильность точности (std, %)', fontsize=13, fontweight='bold')
ax.set_title('Стабильность методов: чем ниже, тем стабильнее', 
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3, which='both')
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(plots_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Сохранено: plots/stability_analysis.png\n")
plt.close()

# ============================================================================
# 7. ТАБЛИЦА СРАВНЕНИЯ
# ============================================================================

print("Создание сводной таблицы результатов...\n")

comparison_data = []

for alpha in [0.1, 0.5, 1.0, 10.0]:
    for method, label, _, _ in methods_comparison:
        key = f"{method}_alpha{alpha}"
        if key in all_data:
            data = all_data[key]
            final_acc = data['global_accuracy'][-1]
            final_loss = data['global_loss'][-1]
            best_acc = max(data['global_accuracy'])
            
            # Найти раунды до 80% и 90%
            rounds_80 = find_rounds_to_target(data['global_accuracy'], 0.80)
            rounds_90 = find_rounds_to_target(data['global_accuracy'], 0.90)
            
            comparison_data.append({
                'α': alpha,
                'Метод': label,
                'Финальная точность': f"{final_acc:.4f}",
                'Лучшая точность': f"{best_acc:.4f}",
                'Финальный Loss': f"{final_loss:.4f}",
                'Раундов до 80%': rounds_80,
                'Раундов до 90%': rounds_90,
                'Всего раундов': len(data['rounds']) - 1
            })

df_comparison = pd.DataFrame(comparison_data)

# Сохранить в CSV
csv_path = plots_dir / 'full_comparison_table.csv'
df_comparison.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✓ Сохранена таблица: {csv_path}")

# Вывести на экран
print("\n" + "="*80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*80)
print(df_comparison.to_string(index=False))
print("="*80 + "\n")

# ============================================================================
# 8. ГРАФИК 6: Тепловая карта финальных точностей
# ============================================================================

print("Создание графика 6: Тепловая карта результатов...")

fig, ax = plt.subplots(figsize=(10, 6))

# Подготовить данные для heatmap
heatmap_data = []
method_labels = []

for method, label, _, _ in methods_comparison:
    row = []
    method_labels.append(label)
    
    for alpha in [0.1, 0.5, 1.0, 10.0]:
        key = f"{method}_alpha{alpha}"
        if key in all_data:
            data = all_data[key]
            final_acc = data['global_accuracy'][-1] * 100
            row.append(final_acc)
        else:
            row.append(np.nan)
    
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, 
                          columns=['α=0.1', 'α=0.5', 'α=1.0', 'α=10.0'],
                          index=method_labels)

sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn', 
           cbar_kws={'label': 'Финальная точность (%)'}, 
           vmin=70, vmax=100, ax=ax, linewidths=1, linecolor='gray')

ax.set_title('Тепловая карта финальных точностей: Метод × Гетерогенность', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Параметр гетерогенности α', fontsize=12, fontweight='bold')
ax.set_ylabel('Метод федеративного обучения', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / 'heatmap_final_accuracy.png', dpi=300, bbox_inches='tight')
print(f"✓ Сохранено: plots/heatmap_final_accuracy.png\n")
plt.close()

# ============================================================================
# ИТОГИ
# ============================================================================

print("\n" + "="*80)
print("✅ ВСЕ ГРАФИКИ СОЗДАНЫ УСПЕШНО!")
print("="*80)
print(f"\nСоздано графиков: 6")
print(f"Создано таблиц: 1")
print(f"Папка: plots/")
print("\nСписок файлов:")
print("  1. comparison_alpha0.1_50rounds.png - Сравнение методов при высокой гетерогенности")
print("  2. heterogeneity_impact_all_methods.png - Влияние α на каждый метод")
print("  3. final_accuracy_vs_alpha.png - Финальная точность vs α")
print("  4. convergence_speed_alpha0.1.png - Скорость сходимости")
print("  5. stability_analysis.png - Стабильность методов")
print("  6. heatmap_final_accuracy.png - Тепловая карта")
print("  7. full_comparison_table.csv - Сводная таблица")
print("\n" + "="*80)
