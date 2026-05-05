@echo off
REM Скрипт для быстрого запуска экспериментов федеративного обучения
REM Автор: ВКР НИУ ВШЭ 2026

echo ========================================
echo Эксперименты федеративного обучения
echo ========================================
echo.

:menu
echo Выберите эксперимент:
echo.
echo 1. Быстрый тест (FedAvg vs FedProx vs Adaptive)
echo 2. Полный набор всех экспериментов
echo 3. Анализ гетерогенности (разные alpha)
echo 4. Анализ масштабирования (разное количество клиентов)
echo 5. Кастомный запуск
echo 0. Выход
echo.

set /p choice="Введите номер: "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto full_experiments
if "%choice%"=="3" goto heterogeneity
if "%choice%"=="4" goto scaling
if "%choice%"=="5" goto custom
if "%choice%"=="0" goto end
goto menu

:quick_test
echo.
echo Запуск быстрого теста (10 раундов)...
echo.
python run_experiments.py --compare --dataset mnist --alpha 0.5
pause
goto menu

:full_experiments
echo.
echo Запуск всех экспериментов...
echo ВНИМАНИЕ: Это займёт много времени!
echo.
set /p confirm="Продолжить? (y/n): "
if /i "%confirm%"=="y" (
    python run_experiments.py --all
)
pause
goto menu

:heterogeneity
echo.
echo Анализ влияния гетерогенности данных...
echo.
python run_experiments.py --heterogeneity
pause
goto menu

:scaling
echo.
echo Анализ влияния количества клиентов...
echo.
python run_experiments.py --scale
pause
goto menu

:custom
echo.
echo Кастомный запуск
echo.
set /p dataset="Датасет (mnist/cifar10): "
set /p alpha="Alpha для Dirichlet (0.1-10.0): "
echo.
python run_experiments.py --compare --dataset %dataset% --alpha %alpha%
pause
goto menu

:end
echo.
echo Завершение работы...
exit /b 0
