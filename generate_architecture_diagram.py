"""
Генерация диаграммы архитектуры федеративной системы для диссертации
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

# Настройка стиля
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Цветовая палитра
COLOR_SERVER = '#3498db'      # Синий
COLOR_STRATEGY = '#2ecc71'    # Зелёный
COLOR_CLIENT = '#e74c3c'      # Красный
COLOR_DATA = '#f39c12'        # Оранжевый
COLOR_MODEL = '#9b59b6'       # Фиолетовый
COLOR_SIMULATION = '#95a5a6'  # Серый

# ============================================================================
# СЕРВЕР (верхняя часть)
# ============================================================================

# Главный блок сервера
server_box = FancyBboxPatch((3.5, 7), 7, 2.5, 
                            boxstyle="round,pad=0.1",
                            edgecolor=COLOR_SERVER, 
                            facecolor=COLOR_SERVER, 
                            alpha=0.3, 
                            linewidth=3)
ax.add_patch(server_box)

ax.text(7, 8.8, 'FLOWER SERVER', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Стратегии внутри сервера
strategies = [
    ('FedAvg\n(Baseline)', 4.2, 7.5),
    ('FedProx\n(μ=0.01)', 7, 7.5),
    ('Adaptive\nFedProx', 9.8, 7.5)
]

for name, x, y in strategies:
    strategy_box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.7,
                                  boxstyle="round,pad=0.05",
                                  edgecolor=COLOR_STRATEGY,
                                  facecolor='white',
                                  linewidth=2)
    ax.add_patch(strategy_box)
    ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold')

# Глобальная модель
model_box = FancyBboxPatch((6, 8.3), 2, 0.4,
                          boxstyle="round,pad=0.05",
                          edgecolor=COLOR_MODEL,
                          facecolor=COLOR_MODEL,
                          alpha=0.4,
                          linewidth=2)
ax.add_patch(model_box)
ax.text(7, 8.5, 'w_global (Global Model)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Flower Simulation API
sim_box = FancyBboxPatch((0.5, 7), 2.5, 2.5,
                        boxstyle="round,pad=0.1",
                        edgecolor=COLOR_SIMULATION,
                        facecolor=COLOR_SIMULATION,
                        alpha=0.2,
                        linewidth=2,
                        linestyle='--')
ax.add_patch(sim_box)
ax.text(1.75, 8.8, 'Flower\nSimulation', ha='center', va='center', 
        fontsize=10, fontweight='bold', style='italic')
ax.text(1.75, 8.2, 'Ray Backend', ha='center', va='center', 
        fontsize=8, style='italic', color='gray')
ax.text(1.75, 7.8, 'Virtual Clients', ha='center', va='center', 
        fontsize=8, style='italic', color='gray')
ax.text(1.75, 7.4, 'CPU/GPU Pool', ha='center', va='center', 
        fontsize=8, style='italic', color='gray')

# ============================================================================
# ФЕДЕРАТИВНЫЙ ЦИКЛ (средняя часть)
# ============================================================================

# Раунд федеративного обучения
round_y = 5.5
ax.text(7, round_y + 0.7, 'ФЕДЕРАТИВНЫЙ ЦИКЛ ОБУЧЕНИЯ', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

# Шаги цикла
steps = [
    ('1. Select Clients\n(fraction_fit=0.3)', 2, round_y),
    ('2. Distribute w_global\n→ Clients', 4.5, round_y),
    ('3. Local Training\n(E=3 epochs)', 7, round_y),
    ('4. Collect Updates\nw_k ← Clients', 9.5, round_y),
    ('5. Aggregate\nFedAvg/FedProx', 12, round_y)
]

for i, (step, x, y) in enumerate(steps):
    step_box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.7,
                             boxstyle="round,pad=0.05",
                             edgecolor='black',
                             facecolor='lightblue' if i % 2 == 0 else 'lightgreen',
                             linewidth=1.5)
    ax.add_patch(step_box)
    ax.text(x, y, step, ha='center', va='center', fontsize=7.5)

# Стрелки между шагами
for i in range(len(steps)-1):
    x1, y1 = steps[i][1], steps[i][2]
    x2, y2 = steps[i+1][1], steps[i+1][2]
    arrow = FancyArrowPatch((x1+0.7, y1), (x2-0.7, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# ============================================================================
# КЛИЕНТЫ (нижняя часть)
# ============================================================================

# Клиенты
client_y = 2.5
clients_info = [
    ('Client 1', 2, client_y, 'D₁: 5843 samples\nα=0.1 (non-IID)'),
    ('Client 2', 4.5, client_y, 'D₂: 6012 samples\nα=0.1 (non-IID)'),
    ('Client 3', 7, client_y, 'D₃: 5721 samples\nα=0.1 (non-IID)'),
    ('Client ...', 9.5, client_y, 'D_k: ~6000 samples\nα=0.1 (non-IID)'),
    ('Client K', 12, client_y, 'D_K: 5998 samples\nα=0.1 (non-IID)')
]

for name, x, y, data_info in clients_info:
    # Клиент
    client_box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 1.2,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLOR_CLIENT,
                               facecolor=COLOR_CLIENT,
                               alpha=0.3,
                               linewidth=2)
    ax.add_patch(client_box)
    
    # Название клиента
    ax.text(x, y + 0.4, name, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Локальная модель
    local_model = Rectangle((x-0.5, y), 1, 0.2,
                           edgecolor=COLOR_MODEL,
                           facecolor=COLOR_MODEL,
                           alpha=0.6)
    ax.add_patch(local_model)
    ax.text(x, y + 0.1, 'w_k', ha='center', va='center', 
            fontsize=7, color='white', fontweight='bold')
    
    # Данные
    data_box = FancyBboxPatch((x-0.7, y-1.3), 1.4, 0.7,
                             boxstyle="round,pad=0.05",
                             edgecolor=COLOR_DATA,
                             facecolor=COLOR_DATA,
                             alpha=0.3,
                             linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(x, y - 0.95, data_info, ha='center', va='center', 
            fontsize=6.5, style='italic')

# ============================================================================
# СТРЕЛКИ КОММУНИКАЦИИ
# ============================================================================

# Стрелка вниз: сервер → клиенты (распределение модели)
for x in [2, 4.5, 7, 9.5, 12]:
    arrow_down = FancyArrowPatch((7, 7), (x, client_y + 0.8),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color=COLOR_MODEL,
                                linestyle='--', alpha=0.6)
    ax.add_patch(arrow_down)

ax.text(5, 4.5, 'Broadcast w_global', ha='center', va='center',
        fontsize=9, color=COLOR_MODEL, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Стрелка вверх: клиенты → сервер (отправка обновлений)
for x in [2, 4.5, 7, 9.5, 12]:
    arrow_up = FancyArrowPatch((x, client_y + 0.8), (7, 7),
                              arrowstyle='->', mutation_scale=15,
                              linewidth=2, color=COLOR_CLIENT,
                              linestyle=':', alpha=0.6)
    ax.add_patch(arrow_up)

ax.text(9, 4.5, 'Upload Δw_k', ha='center', va='center',
        fontsize=9, color=COLOR_CLIENT, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ============================================================================
# DATA PARTITIONING (левый блок)
# ============================================================================

partition_box = FancyBboxPatch((0.2, 1.5), 1.3, 2.5,
                              boxstyle="round,pad=0.1",
                              edgecolor=COLOR_DATA,
                              facecolor='lightyellow',
                              alpha=0.4,
                              linewidth=2,
                              linestyle='--')
ax.add_patch(partition_box)

ax.text(0.85, 3.6, 'Dirichlet', ha='center', va='center',
        fontsize=9, fontweight='bold')
ax.text(0.85, 3.3, 'Partitioner', ha='center', va='center',
        fontsize=9, fontweight='bold')
ax.text(0.85, 2.9, '━━━━━━━', ha='center', va='center', fontsize=8)
ax.text(0.85, 2.6, 'α = 0.1', ha='center', va='center', fontsize=8, style='italic')
ax.text(0.85, 2.3, 'Non-IID', ha='center', va='center', fontsize=8, style='italic')
ax.text(0.85, 2.0, 'MNIST', ha='center', va='center', fontsize=8, style='italic')
ax.text(0.85, 1.7, '60k samples', ha='center', va='center', fontsize=7, style='italic')

# Стрелки от партиционера к клиентам
for x in [2, 4.5, 7]:
    arrow_part = FancyArrowPatch((1.5, 2.5), (x - 0.8, client_y - 0.5),
                                arrowstyle='->', mutation_scale=10,
                                linewidth=1.5, color=COLOR_DATA,
                                linestyle=':', alpha=0.5)
    ax.add_patch(arrow_part)

# ============================================================================
# ЛЕГЕНДА И АННОТАЦИИ
# ============================================================================

# Заголовок
ax.text(7, 9.7, 'АРХИТЕКТУРА ФЕДЕРАТИВНОЙ СИСТЕМЫ',
        ha='center', va='center', fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgray', alpha=0.5))

# Параметры системы (правый блок)
params_box = FancyBboxPatch((11.7, 0.3), 2, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor='gray',
                           facecolor='lightgray',
                           alpha=0.3,
                           linewidth=1.5)
ax.add_patch(params_box)

params_text = """Параметры:
━━━━━━━━━
K = 10 клиентов
T = 50 раундов
E = 3 эпохи
C = 0.3 (30%)
B = 32 (batch)"""

ax.text(12.7, 1.05, params_text, ha='center', va='center',
        fontsize=7, family='monospace')

# Формулы (нижний блок)
formula_y = 0.5

# FedAvg
ax.text(2.5, formula_y, 'FedAvg:', ha='left', va='center',
        fontsize=9, fontweight='bold')
ax.text(2.5, formula_y - 0.3, r'$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^t$',
        ha='left', va='center', fontsize=10, style='italic')

# FedProx
ax.text(6, formula_y, 'FedProx:', ha='left', va='center',
        fontsize=9, fontweight='bold')
ax.text(6, formula_y - 0.3, r'$L_k = L_{CE} + \frac{\mu}{2}||w - w_{global}||^2$',
        ha='left', va='center', fontsize=10, style='italic')

# Adaptive
ax.text(9.7, formula_y, 'Adaptive:', ha='left', va='center',
        fontsize=9, fontweight='bold')
ax.text(9.7, formula_y - 0.3, r'$\mu(t) = \mu_0 \cdot (1 - t/T)$',
        ha='left', va='center', fontsize=10, style='italic')

# Легенда (левый нижний угол)
legend_elements = [
    mlines.Line2D([0], [0], color=COLOR_MODEL, linewidth=2, linestyle='--', 
                  label='Распределение модели'),
    mlines.Line2D([0], [0], color=COLOR_CLIENT, linewidth=2, linestyle=':', 
                  label='Обновления клиентов'),
    mlines.Line2D([0], [0], color=COLOR_DATA, linewidth=2, linestyle=':', 
                  label='Разбиение данных'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9)

# Сохранение
plt.tight_layout()
plt.savefig('plots/architecture_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Диаграмма сохранена: plots/architecture_diagram.png")

# Также сохраним в высоком разрешении для диссертации
plt.savefig('plots/architecture_diagram_hires.pdf', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ PDF версия: plots/architecture_diagram_hires.pdf")

plt.show()
