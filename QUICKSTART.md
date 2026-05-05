# 🚀 Quickstart - Быстрый старт за 3 минуты

## Windows

### Шаг 1: Установка (1 мин)
```cmd
setup.bat
```

### Шаг 2: Быстрый тест (2 мин)
```cmd
python run_experiments.py --compare --dataset mnist --alpha 0.5
```

### Шаг 3: Просмотр результатов
- Графики в папке `plots/`
- Метрики в папке `results/`

**Готово!** У вас есть первые результаты для диссертации.

---

## Linux / Mac

### Шаг 1: Установка (1 мин)
```bash
chmod +x setup.sh
./setup.sh
```

### Шаг 2: Быстрый тест (2 мин)
```bash
python3 run_experiments.py --compare --dataset mnist --alpha 0.5
```

### Шаг 3: Просмотр результатов
- Графики в папке `plots/`
- Метрики в папке `results/`

**Готово!** У вас есть первые результаты для диссертации.

---

## Что дальше?

### Для полного набора экспериментов:
```bash
python run_experiments.py --all
```
⏱️ Время выполнения: ~40-60 мин на GPU, ~2-3 часа на CPU

### Для интерактивного меню:
**Windows:**
```cmd
quick_run.bat
```

**Linux/Mac:**
```bash
./quick_run.sh
```

### Для настройки параметров:
Отредактируйте `config.yaml`

---

## Возникла проблема?

1. **Нет модуля torch:**
   ```bash
   pip install torch torchvision
   ```

2. **Медленно выполняется:**
   - Уменьшите `rounds` в `config.yaml` до 10-20
   - Или используйте GPU: `device: "cuda"` в config

3. **Другие проблемы:**
   - См. `INSTALL.md` — подробное руководство
   - См. `README.md` — полная документация

---

## 📚 Документация

| Файл | Описание |
|------|----------|
| `README.md` | Полная документация проекта |
| `INSTALL.md` | Детальная инструкция по установке |
| `DISSERTATION_GUIDE.md` | Как использовать результаты в диссертации |
| `PROJECT_SUMMARY.md` | Что реализовано и как это работает |

---

## ✅ Минимальный пример для диссертации

```bash
# 1. Установка
pip install -r requirements.txt

# 2. Быстрый эксперимент (10 раундов)
python run_experiments.py --compare --dataset mnist --alpha 0.5

# 3. Найти результаты:
# - plots/accuracy_comparison_mnist_alpha0.5.png
# - plots/loss_comparison_mnist_alpha0.5.png
# - results/comparison_mnist_alpha0.5.csv
```

Эти файлы можно сразу вставить в диссертацию!

---

**Время до первых результатов: 3-5 минут** ⚡
