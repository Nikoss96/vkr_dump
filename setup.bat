@echo off
REM Скрипт для автоматической установки зависимостей
REM Проект: Федеративное обучение (ВКР НИУ ВШЭ 2026)

echo ========================================
echo Установка зависимостей
echo ========================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден! Установите Python 3.9+ с python.org
    pause
    exit /b 1
)

echo [OK] Python найден
python --version
echo.

REM Проверка pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip не найден!
    pause
    exit /b 1
)

echo [OK] pip найден
echo.

REM Обновление pip
echo Обновление pip...
python -m pip install --upgrade pip
echo.

REM Установка зависимостей
echo Установка зависимостей из requirements.txt...
echo.
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Ошибка при установке зависимостей!
    echo Попробуйте установить вручную:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ========================================
echo Установка завершена успешно!
echo ========================================
echo.

REM Проверка установки ключевых пакетов
echo Проверка установленных пакетов:
python -c "import torch; print(f'  [OK] PyTorch {torch.__version__}')" 2>nul || echo   [?] PyTorch - проверьте установку
python -c "import flwr; print(f'  [OK] Flower {flwr.__version__}')" 2>nul || echo   [?] Flower - проверьте установку
python -c "import numpy; print(f'  [OK] NumPy {numpy.__version__}')" 2>nul || echo   [?] NumPy - проверьте установку
python -c "import matplotlib; print(f'  [OK] Matplotlib {matplotlib.__version__}')" 2>nul || echo   [?] Matplotlib - проверьте установку
python -c "import pandas; print(f'  [OK] Pandas {pandas.__version__}')" 2>nul || echo   [?] Pandas - проверьте установку
echo.

REM Создание необходимых директорий
echo Создание рабочих директорий...
if not exist "data" mkdir data
if not exist "results" mkdir results
if not exist "plots" mkdir plots
echo   [OK] Директории созданы
echo.

echo ========================================
echo Готово к запуску!
echo ========================================
echo.
echo Для запуска экспериментов используйте:
echo   python run_experiments.py --all
echo.
echo Или интерактивное меню:
echo   quick_run.bat
echo.
echo Документация:
echo   README.md - Основное описание
echo   INSTALL.md - Инструкция по установке
echo   DISSERTATION_GUIDE.md - Руководство для диссертации
echo.

pause
