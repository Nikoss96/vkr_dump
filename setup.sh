#!/bin/bash
# Скрипт для автоматической установки зависимостей
# Проект: Федеративное обучение (ВКР НИУ ВШЭ 2026)

echo "========================================"
echo "Установка зависимостей"
echo "========================================"
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 не найден! Установите Python 3.9+"
    exit 1
fi

echo "[OK] Python найден"
python3 --version
echo ""

# Проверка pip
if ! python3 -m pip --version &> /dev/null; then
    echo "[ERROR] pip не найден!"
    exit 1
fi

echo "[OK] pip найден"
echo ""

# Обновление pip
echo "Обновление pip..."
python3 -m pip install --upgrade pip
echo ""

# Установка зависимостей
echo "Установка зависимостей из requirements.txt..."
echo ""
python3 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Ошибка при установке зависимостей!"
    echo "Попробуйте установить вручную:"
    echo "  pip3 install -r requirements.txt"
    exit 1
fi

echo ""
echo "========================================"
echo "Установка завершена успешно!"
echo "========================================"
echo ""

# Проверка установки ключевых пакетов
echo "Проверка установленных пакетов:"
python3 -c "import torch; print(f'  [OK] PyTorch {torch.__version__}')" 2>/dev/null || echo "  [?] PyTorch - проверьте установку"
python3 -c "import flwr; print(f'  [OK] Flower {flwr.__version__}')" 2>/dev/null || echo "  [?] Flower - проверьте установку"
python3 -c "import numpy; print(f'  [OK] NumPy {numpy.__version__}')" 2>/dev/null || echo "  [?] NumPy - проверьте установку"
python3 -c "import matplotlib; print(f'  [OK] Matplotlib {matplotlib.__version__}')" 2>/dev/null || echo "  [?] Matplotlib - проверьте установку"
python3 -c "import pandas; print(f'  [OK] Pandas {pandas.__version__}')" 2>/dev/null || echo "  [?] Pandas - проверьте установку"
echo ""

# Создание необходимых директорий
echo "Создание рабочих директорий..."
mkdir -p data results plots
echo "  [OK] Директории созданы"
echo ""

# Установка прав на выполнение для скриптов
echo "Установка прав на выполнение..."
chmod +x quick_run.sh
echo "  [OK] Права установлены"
echo ""

echo "========================================"
echo "Готово к запуску!"
echo "========================================"
echo ""
echo "Для запуска экспериментов используйте:"
echo "  python3 run_experiments.py --all"
echo ""
echo "Или интерактивное меню:"
echo "  ./quick_run.sh"
echo ""
echo "Документация:"
echo "  README.md - Основное описание"
echo "  INSTALL.md - Инструкция по установке"
echo "  DISSERTATION_GUIDE.md - Руководство для диссертации"
echo ""
