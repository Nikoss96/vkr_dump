# 📖 Документация проекта - Навигация

Добро пожаловать в проект исследования федеративного обучения!

## 🚀 С чего начать?

### Новичок в проекте?
Начните с **[QUICKSTART.md](QUICKSTART.md)** — быстрый старт за 3 минуты

### Нужна подробная инструкция?
Читайте **[INSTALL.md](INSTALL.md)** — детальное руководство по установке и запуску

### Хотите понять что реализовано?
См. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** — полная сводка проекта

---

## 📚 Документация

### Основные файлы

| Документ | Описание | Для кого |
|----------|----------|----------|
| **[QUICKSTART.md](QUICKSTART.md)** | Быстрый старт за 3 минуты | Все |
| **[README.md](README.md)** | Полная документация проекта | Разработчики |
| **[INSTALL.md](INSTALL.md)** | Подробная инструкция по установке | Новые пользователи |
| **[DISSERTATION_GUIDE.md](DISSERTATION_GUIDE.md)** | Как использовать результаты в диссертации | Магистранты |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | Что реализовано в проекте | Проверяющие |

### Файлы конфигурации

| Файл | Описание |
|------|----------|
| `config.yaml` | Основная конфигурация экспериментов |
| `requirements.txt` | Зависимости Python |
| `.gitignore` | Git ignore правила |

### Исходный код

| Директория | Содержание |
|------------|------------|
| `clients/` | Flower клиенты и партиционирование данных |
| `models/` | Модели нейронных сетей (CNN, ResNet) |
| `strategies/` | Алгоритмы FL (FedAvg, FedProx, Adaptive FedProx) |
| `utils/` | Утилиты (метрики, визуализация) |

### Скрипты

| Файл | Описание | Платформа |
|------|----------|-----------|
| `run_experiments.py` | Главный скрипт запуска | Все |
| `example.py` | Примеры использования API | Все |
| `setup.bat` | Автоматическая установка | Windows |
| `setup.sh` | Автоматическая установка | Linux/Mac |
| `quick_run.bat` | Интерактивное меню | Windows |
| `quick_run.sh` | Интерактивное меню | Linux/Mac |

---

## 🎯 Быстрая навигация по задачам

### "Мне нужно быстро получить результаты для диссертации"
→ **[QUICKSTART.md](QUICKSTART.md)** + **[DISSERTATION_GUIDE.md](DISSERTATION_GUIDE.md)**

### "Я хочу понять как это работает"
→ **[README.md](README.md)** + **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**

### "У меня проблемы с установкой"
→ **[INSTALL.md](INSTALL.md)** раздел "Устранение проблем"

### "Как настроить эксперименты?"
→ **[README.md](README.md)** раздел "Конфигурация" + `config.yaml`

### "Как интерпретировать результаты?"
→ **[DISSERTATION_GUIDE.md](DISSERTATION_GUIDE.md)** раздел "Как использовать результаты"

### "Хочу изменить код"
→ **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** раздел "Технические детали" + исходный код

---

## 📊 Структура проекта

```
VKR_base/
│
├── 📄 Документация
│   ├── QUICKSTART.md           ⚡ Быстрый старт
│   ├── README.md               📖 Полная документация
│   ├── INSTALL.md              🔧 Инструкция по установке
│   ├── DISSERTATION_GUIDE.md   🎓 Руководство для диссертации
│   ├── PROJECT_SUMMARY.md      📝 Сводка проекта
│   └── INDEX.md                🗂️ Этот файл
│
├── 🚀 Запуск
│   ├── run_experiments.py      Главный скрипт
│   ├── example.py              Примеры
│   ├── quick_run.bat           Меню (Windows)
│   ├── quick_run.sh            Меню (Linux/Mac)
│   ├── setup.bat               Установка (Windows)
│   └── setup.sh                Установка (Linux/Mac)
│
├── ⚙️ Конфигурация
│   ├── config.yaml             Настройки
│   ├── requirements.txt        Зависимости
│   └── .gitignore              Git ignore
│
├── 💻 Исходный код
│   ├── clients/                Клиенты и партиционирование
│   ├── models/                 Нейронные сети
│   ├── strategies/             Алгоритмы FL
│   └── utils/                  Утилиты
│
└── 📁 Данные (создаются автоматически)
    ├── data/                   Датасеты
    ├── results/                CSV/JSON метрики
    └── plots/                  Графики PNG
```

---

## 🎓 Для диссертации

Чтобы получить все необходимые материалы для диссертации:

### Шаг 1: Установка и запуск
```bash
# Windows
setup.bat
python run_experiments.py --all

# Linux/Mac
./setup.sh
python3 run_experiments.py --all
```

### Шаг 2: Использование результатов
Читайте **[DISSERTATION_GUIDE.md](DISSERTATION_GUIDE.md)**

Там подробно описано:
- Какие графики включить в какие разделы
- Как интерпретировать результаты
- Примеры текста и таблиц
- Как оформить научную новизну

---

## 💡 Советы

### Для быстрых тестов
Уменьшите количество раундов в `config.yaml`:
```yaml
general:
  rounds: 10  # вместо 50
```

### Для использования GPU
Убедитесь что в `config.yaml`:
```yaml
general:
  device: "cuda"
```

### Для воспроизводимости
Seed уже установлен в config:
```yaml
general:
  seed: 42
```

---

## 📞 Нужна помощь?

1. **Проблемы с установкой?** → [INSTALL.md](INSTALL.md) раздел "Устранение проблем"
2. **Не понимаете результаты?** → [DISSERTATION_GUIDE.md](DISSERTATION_GUIDE.md)
3. **Хотите изменить код?** → [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) + исходники
4. **Просто хотите быстро начать?** → [QUICKSTART.md](QUICKSTART.md)

---

## ✅ Чеклист готовности

- [ ] Прочитал [QUICKSTART.md](QUICKSTART.md)
- [ ] Установил зависимости (`setup.bat` или `setup.sh`)
- [ ] Запустил первый эксперимент
- [ ] Изучил результаты в `plots/` и `results/`
- [ ] Прочитал [DISSERTATION_GUIDE.md](DISSERTATION_GUIDE.md)
- [ ] Готов к полному запуску экспериментов

---

**Удачи в исследованиях! 🚀**

Проект: Федеративное обучение  
ВКР НИУ ВШЭ 2026
