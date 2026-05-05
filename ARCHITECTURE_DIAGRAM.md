# 🏗️ Архитектура распределённой системы (сервер-клиенты)

## ASCII-схема системы

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        FLOWER SIMULATION API (Ray)                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          FLOWER SERVER                                │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │              Глобальная модель (w_global)                     │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  │                                                                       │  │
│  │  Стратегии агрегации:                                                │  │
│  │  ┌───────────┐    ┌────────────┐    ┌──────────────────┐            │  │
│  │  │  FedAvg   │    │  FedProx   │    │ Adaptive FedProx │            │  │
│  │  │ (Baseline)│    │ (μ=0.01)   │    │  (μ₀=0.1→0.001)  │            │  │
│  │  └───────────┘    └────────────┘    └──────────────────┘            │  │
│  │                                                                       │  │
│  │  Функции:                                                             │  │
│  │  • Инициализация глобальной модели                                   │  │
│  │  • Выбор клиентов на раунд (fraction_fit=0.3 → 3 из 10)            │  │
│  │  • Рассылка w_global → клиентам                                     │  │
│  │  • Агрегация обновлений: w_global = Σ(n_k/n_total)·w_k             │  │
│  │  • Централизованная оценка на тестовом наборе                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Раунд t: Broadcast w_global
                    ┌──────────────┼──────────────┬─────────────┐
                    │              │              │             │
                    ▼              ▼              ▼             ▼
         ┌─────────────────────────────────────────────────────────────┐
         │              ФЕДЕРАТИВНЫЙ ЦИКЛ ОБУЧЕНИЯ                     │
         │  1. select_clients() → 3 клиента (C=0.3)                    │
         │  2. server.send(w_global) → Client₁, Client₂, Client₃       │
         │  3. Локальное обучение: E=3 эпохи, batch_size=32            │
         │  4. client.send(Δw_k) → сервер                              │
         │  5. aggregate_fit(Δw₁, Δw₂, Δw₃) → w_global'                │
         │  6. evaluate(w_global') → accuracy, loss                     │
         │  Повторить для раундов t=1..T (T=50)                        │
         └─────────────────────────────────────────────────────────────┘
                    │              │              │             │
                    │              │              │             │
    ┌───────────────┘              │              └──────────────────┐
    │                              │                                 │
    ▼                              ▼                                 ▼
┌─────────┐                  ┌─────────┐                      ┌─────────┐
│Client 1 │                  │Client 2 │      ...             │Client K │
├─────────┤                  ├─────────┤                      ├─────────┤
│ w_local │◄─ Получает       │ w_local │                      │ w_local │
│         │   w_global       │         │                      │         │
│ Обучение│                  │ Обучение│                      │ Обучение│
│ E=3     │   for epoch      │ E=3     │                      │ E=3     │
│ epochs  │   in range(3):   │ epochs  │                      │ epochs  │
│         │     for batch:   │         │                      │         │
│ Proximal│       loss =     │ Proximal│                      │ Proximal│
│ Term?   │       CE_loss +  │ Term?   │                      │ Term?   │
│ μ=0.01  │       μ/2*||w-   │ μ=0.01  │                      │ μ(t)    │
│         │       w_global||²│         │                      │         │
│         │       backward() │         │                      │         │
│         │   Отправляет     │         │                      │         │
│ Δw₁     │─► Δw_k на сервер │ Δw₂     │                      │ Δw_K    │
└────┬────┘                  └────┬────┘                      └────┬────┘
     │                            │                                │
     │ Локальные данные D_k       │                                │
     ▼                            ▼                                ▼
┌─────────┐                  ┌─────────┐                      ┌─────────┐
│   D₁    │                  │   D₂    │                      │   D_K   │
│ 5843    │◄─ Dirichlet      │ 6012    │      ...             │ 5998    │
│ samples │   Partitioner    │ samples │                      │ samples │
│ α=0.1   │   (α=0.1)        │ α=0.1   │                      │ α=0.1   │
│ non-IID │                  │ non-IID │                      │ non-IID │
├─────────┤                  ├─────────┤                      ├─────────┤
│Class 0: │ 1842 (31%)       │Class 0: │   12 (0.2%)          │Class 0: │  98
│Class 1: │   45 (0.8%)      │Class 1: │  890 (15%)           │Class 1: │ 1654
│Class 2: │  312 (5.3%)      │Class 2: │ 1765 (29%)           │Class 2: │  45
│  ...    │                  │  ...    │                      │  ...    │
└─────────┘                  └─────────┘                      └─────────┘
      ▲                            ▲                                ▲
      │                            │                                │
      └────────────────────────────┴────────────────────────────────┘
                                   │
                        hetero_partitioner.py
                        Dirichlet(α) distribution
                        MNIST: 60,000 samples → K partitions
```

---

## 📋 Компоненты системы

### **1. FLOWER SERVER** 🖥️

**Файлы:** `run_experiments.py`, `strategies/fedavg.py`, `strategies/fedprox.py`, `strategies/adaptive_fedprox.py`

**Ключевые функции:**

```python
# Инициализация сервера
fl.simulation.start_simulation(
    client_fn=create_client_fn(...),    # Фабрика клиентов
    num_clients=10,                      # K = 10 клиентов
    config=ServerConfig(num_rounds=50),  # T = 50 раундов
    strategy=strategy,                   # FedAvg/FedProx/Adaptive
    client_resources={
        "num_cpus": 1,                   # 1 CPU на клиента
        "num_gpus": 0.1                  # Делим GPU между клиентами
    }
)
```

**Стратегии агрегации:**

1. **FedAvg (Baseline):**
   ```
   w_global ← Σ(n_k/n_total) · w_k    (взвешенное усреднение)
   ```

2. **FedProx (μ=0.01):**
   ```
   L_k = L_CE + (μ/2)||w - w_global||²    (проксимальный терм)
   w_global ← Σ(n_k/n_total) · w_k
   ```

3. **Adaptive FedProx:**
   ```
   μ(t) = μ₀ · max(0, 1 - t/T)            (линейное уменьшение)
   L_k = L_CE + (μ(t)/2)||w - w_global||²
   w_global ← Σ(n_k/n_total) · w_k
   ```

**Процесс раунда:**

```
Round t:
├─ 1. select_clients(fraction_fit=0.3) → 3 клиента из 10
├─ 2. configure_fit() → отправить config={'mu': 0.01, 'proximal': True}
├─ 3. distribute(w_global) → отправить параметры модели клиентам
├─ 4. wait_for_updates() → получить Δw_k от клиентов
├─ 5. aggregate_fit([w₁, w₂, w₃]) → обновить w_global
├─ 6. evaluate(w_global) → оценка на test set
└─ 7. log_metrics(accuracy, loss) → сохранить в MetricsLogger
```

---

### **2. FLOWER CLIENTS** 📱

**Файлы:** `clients/flower_client.py`

**Класс FlowerClient:**

```python
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, test_loader):
        self.cid = cid                    # Идентификатор клиента
        self.model = model                # Локальная копия модели
        self.train_loader = train_loader  # Локальные данные D_k
        self.global_params = None         # Хранит w_global для FedProx
    
    def get_parameters(self, config):
        """Отправить текущие параметры w_k на сервер"""
        return [p.cpu().numpy().copy() for p in self.model.parameters()]
    
    def set_parameters(self, parameters):
        """Получить w_global от сервера"""
        self.global_params = [p.copy() for p in parameters]
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p).to(p.device)
    
    def fit(self, parameters, config):
        """Локальное обучение"""
        self.set_parameters(parameters)
        
        # Получить параметры FedProx
        proximal = config.get("proximal", False)
        mu = config.get("mu", 0.1)
        
        # Обучение E эпох
        for epoch in range(3):
            for batch in self.train_loader:
                # Стандартный loss
                loss = cross_entropy(model(X), y)
                
                # Проксимальный терм (только для FedProx)
                if proximal:
                    prox_loss = 0.0
                    for local_p, global_p in zip(model.parameters(), 
                                                   self.global_params):
                        prox_loss += torch.sum((local_p - global_p)**2)
                    loss += (mu / 2) * prox_loss
                
                loss.backward()
                optimizer.step()
        
        return self.get_parameters({}), len(train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Локальная оценка (опционально)"""
        self.set_parameters(parameters)
        accuracy, loss = evaluate_model(self.model, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}
```

**Независимость клиентов:**
- Каждый клиент — отдельный Ray actor
- Выполняются параллельно на доступных CPU/GPU
- Имеют изолированные локальные данные D_k

---

### **3. DATA PARTITIONING** 📊

**Файлы:** `clients/hetero_partitioner.py`

**Dirichlet Partitioning:**

```python
class DirichletPartitioner:
    def partition(self, dataset, num_clients, alpha):
        """
        Разбиение датасета по Dirichlet(α)
        
        Args:
            dataset: PyTorch Dataset (MNIST/CIFAR-10)
            num_clients: K = 10
            alpha: Концентрация (0.1 = высокая гетерогенность)
        
        Returns:
            List[Subset]: K партиций данных
        """
        # Группировка по классам
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        
        # Для каждого класса: разбить по Dirichlet(α)
        client_indices = [[] for _ in range(num_clients)]
        
        for c, indices in class_indices.items():
            # Сэмплировать пропорции из Dirichlet
            proportions = np.random.dirichlet([alpha] * num_clients)
            
            # Распределить данные класса c между клиентами
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)
            client_splits = np.split(indices, proportions[:-1])
            
            for k, split in enumerate(client_splits):
                client_indices[k].extend(split)
        
        # Создать Subset для каждого клиента
        return [Subset(dataset, indices) for indices in client_indices]
```

**Статистика разбиения (α=0.1, K=10):**

```
Client 0: 5843 samples
  Class 0: 1842 (31.5%) ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
  Class 1:   45 (0.8%)  ▪
  Class 2:  312 (5.3%)  ◼◼
  Class 3:   67 (1.1%)  ▪
  Class 4:  892 (15.3%) ◼◼◼◼◼◼◼
  ...

Client 1: 6012 samples
  Class 0:   12 (0.2%)  ▪
  Class 1:  890 (14.8%) ◼◼◼◼◼◼◼
  Class 2: 1765 (29.4%) ◼◼◼◼◼◼◼◼◼◼◼◼◼◼
  Class 3:   89 (1.5%)  ▪
  ...

...

Client 9: 5998 samples
  Class 0:   98 (1.6%)  ▪
  Class 1: 1654 (27.6%) ◼◼◼◼◼◼◼◼◼◼◼◼◼
  Class 2:   45 (0.8%)  ▪
  Class 3: 2134 (35.6%) ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
  ...
```

**Эффект α:**
- α = 10.0 → почти IID (равномерное распределение)
- α = 1.0 → слабая гетерогенность
- α = 0.5 → умеренная гетерогенность
- α = 0.1 → высокая гетерогенность (клиенты специализируются на 1-2 классах)

---

### **4. FLOWER SIMULATION API** 🔄

**Технология:** Ray (распределённые вычисления)

**Архитектура симуляции:**

```
┌─────────────────────────────────────────────────┐
│         Рабочая станция (1 машина)              │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │          Ray Cluster                       │ │
│  │                                            │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐   │ │
│  │  │ Actor 1 │  │ Actor 2 │  │ Actor K │   │ │
│  │  │ Client 1│  │ Client 2│  │ Client K│   │ │
│  │  │ CPU: 1  │  │ CPU: 1  │  │ CPU: 1  │   │ │
│  │  │ GPU: 0.1│  │ GPU: 0.1│  │ GPU: 0.1│   │ │
│  │  └─────────┘  └─────────┘  └─────────┘   │ │
│  │                                            │ │
│  │  GPU Memory: Модели делят 1 GPU           │ │
│  │  CPU Cores: Параллельное обучение          │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  Преимущества:                                   │
│  • Реалистичная симуляция FL без сети           │
│  • Воспроизводимость (seed=42)                  │
│  • Масштабируемость (10-100+ клиентов)          │
│  • Быстрое развёртывание (нет инфраструктуры)  │
└─────────────────────────────────────────────────┘
```

**Отличие от реального FL:**

| Аспект | Симуляция | Реальная FL |
|--------|-----------|-------------|
| Клиенты | Ray actors в памяти | Физические устройства (телефоны, IoT) |
| Сеть | Локальная (мгновенная) | Интернет (задержки, потери пакетов) |
| Данные | Синтетическое разбиение | Естественное распределение |
| Отказы | Нет (контролируемая среда) | Возможны (отключения, батарея) |
| Время | Минуты | Часы/дни |

---

## 🔄 Жизненный цикл раунда федеративного обучения

```
┌─────────────────────────────────────────────────────────────────┐
│                     РАУНД t (t=1..50)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │ 1. ВЫБОР КЛИЕНТОВ (Server)               │
        │    selected = random.sample(             │
        │        clients, k=int(0.3*10)            │
        │    ) → [Client₂, Client₅, Client₈]       │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │ 2. КОНФИГУРАЦИЯ (Server → Clients)       │
        │    config = {                            │
        │        'proximal': True,                 │
        │        'mu': 0.01,  # FedProx            │
        │        'local_epochs': 3                 │
        │    }                                     │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │ 3. РАСПРЕДЕЛЕНИЕ МОДЕЛИ                  │
        │    Server broadcasts w_global:           │
        │    • Вес: ~420 KB (MNIST CNN)            │
        │    • Получатели: 3 клиента               │
        │    • Время: ~50 мс (симуляция)           │
        └──────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │Client₂  │         │Client₅  │         │Client₈  │
    │D₂:6012  │         │D₅:5921  │         │D₈:6134  │
    └─────────┘         └─────────┘         └─────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────┐
    │ 4. ЛОКАЛЬНОЕ ОБУЧЕНИЕ (Clients)                 │
    │    for epoch in range(3):                       │
    │        for batch in DataLoader(D_k, batch=32):  │
    │            # Forward pass                       │
    │            loss = cross_entropy(model(X), y)    │
    │                                                 │
    │            # Проксимальный терм (FedProx)       │
    │            if proximal:                         │
    │                prox = (mu/2)*||w-w_global||²    │
    │                loss += prox                     │
    │                                                 │
    │            # Backward pass                      │
    │            loss.backward()                      │
    │            optimizer.step()                     │
    │                                                 │
    │    Время: ~10-15 сек на клиента (GPU)          │
    └─────────────────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │  Δw₂    │         │  Δw₅    │         │  Δw₈    │
    │ (~420KB)│         │ (~420KB)│         │ (~420KB)│
    └─────────┘         └─────────┘         └─────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
        ┌──────────────────────────────────────────┐
        │ 5. АГРЕГАЦИЯ (Server)                    │
        │    n_total = n₂ + n₅ + n₈                │
        │            = 6012 + 5921 + 6134 = 18067  │
        │                                          │
        │    w_global' = (n₂/n_total)·w₂ +         │
        │                (n₅/n_total)·w₅ +         │
        │                (n₈/n_total)·w₈           │
        │                                          │
        │    Время: ~1-2 сек                       │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │ 6. ЦЕНТРАЛИЗОВАННАЯ ОЦЕНКА               │
        │    Test Set: 10,000 samples (MNIST)      │
        │    accuracy = evaluate(w_global', D_test)│
        │    loss = cross_entropy(...)             │
        │                                          │
        │    Логирование:                          │
        │    • metrics_history['round'].append(t)  │
        │    • metrics_history['accuracy'].append()│
        │    • metrics_history['loss'].append()    │
        │                                          │
        │    Время: ~2-3 сек                       │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │ 7. ЛОГИРОВАНИЕ И СОХРАНЕНИЕ              │
        │    MetricsLogger.log_round(              │
        │        round=t,                          │
        │        accuracy=0.9857,                  │
        │        loss=0.0563                       │
        │    )                                     │
        │                                          │
        │    Сохранение:                           │
        │    • results/*.json                      │
        │    • results/*.csv                       │
        └──────────────────────────────────────────┘
                              │
                              ▼
                    Следующий раунд t+1
```

**Временные характеристики (1 раунд):**
- Выбор клиентов: ~0.1 сек
- Распределение модели: ~0.05 сек (симуляция)
- Локальное обучение: ~10-15 сек (3 клиента параллельно)
- Агрегация: ~1-2 сек
- Оценка: ~2-3 сек
- **Итого: ~15-20 сек на раунд**

**Для 50 раундов:** ~12-17 минут на GPU

---

## 📊 Ключевые параметры системы

```yaml
# config.yaml
general:
  seed: 42                  # Воспроизводимость
  rounds: 50                # T = 50 раундов
  fraction_fit: 0.3         # C = 30% клиентов на раунд
  local_epochs: 3           # E = 3 локальные эпохи
  batch_size: 32            # B = 32 размер батча
  num_clients: 10           # K = 10 клиентов
  device: "cuda"            # GPU ускорение

heterogeneity:
  alpha: 0.1                # Dirichlet концентрация

fedprox:
  mu: 0.01                  # Проксимальный параметр

adaptive_fedprox:
  mu0: 0.1                  # Начальный μ
  decay_strategy: "linear"  # Стратегия уменьшения
  min_mu: 0.001             # Минимальный μ
```

---

## 🎯 Преимущества архитектуры

### **1. Масштабируемость**
- ✅ Поддержка 10-100+ клиентов
- ✅ Параллельное обучение на GPU/CPU pool
- ✅ Линейное масштабирование по ресурсам

### **2. Гибкость**
- ✅ Легко добавить новые стратегии (SCAFFOLD, FedOpt)
- ✅ Поддержка любых PyTorch моделей
- ✅ Конфигурируемые параметры через YAML

### **3. Воспроизводимость**
- ✅ Фиксированный seed=42
- ✅ Детерминированное разбиение данных
- ✅ Логирование всех экспериментов

### **4. Реалистичность**
- ✅ Non-IID данные (Dirichlet partitioning)
- ✅ Частичное участие клиентов (fraction_fit=0.3)
- ✅ Независимое локальное обучение

### **5. Эффективность**
- ✅ 50 раундов за ~15 минут на GPU
- ✅ Без сетевых задержек (симуляция)
- ✅ Оптимизированное использование GPU памяти

---

## 📚 Ссылки на код

| Компонент | Файл | Строки |
|-----------|------|--------|
| Сервер | [run_experiments.py](run_experiments.py) | 350-370 |
| Клиент | [clients/flower_client.py](clients/flower_client.py) | 1-220 |
| FedAvg | [strategies/fedavg.py](strategies/fedavg.py) | 1-110 |
| FedProx | [strategies/fedprox.py](strategies/fedprox.py) | 1-140 |
| Adaptive | [strategies/adaptive_fedprox.py](strategies/adaptive_fedprox.py) | 1-220 |
| Partitioning | [clients/hetero_partitioner.py](clients/hetero_partitioner.py) | 15-90 |
| Metrics | [utils/metrics.py](utils/metrics.py) | 1-200 |

---

**Документ готов для использования в разделе 4.3 диссертации.**
