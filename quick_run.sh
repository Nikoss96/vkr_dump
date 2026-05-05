#!/bin/bash
# Скрипт для быстрого запуска экспериментов федеративного обучения
# Автор: ВКР НИУ ВШЭ 2026

echo "========================================"
echo "Эксперименты федеративного обучения"
echo "========================================"
echo ""

function show_menu() {
    echo "Выберите эксперимент:"
    echo ""
    echo "1. Быстрый тест (FedAvg vs FedProx vs Adaptive)"
    echo "2. Полный набор всех экспериментов"
    echo "3. Анализ гетерогенности (разные alpha)"
    echo "4. Анализ масштабирования (разное количество клиентов)"
    echo "5. Кастомный запуск"
    echo "0. Выход"
    echo ""
}

function quick_test() {
    echo ""
    echo "Запуск быстрого теста (10 раундов)..."
    echo ""
    python run_experiments.py --compare --dataset mnist --alpha 0.5
    read -p "Нажмите Enter для продолжения..."
}

function full_experiments() {
    echo ""
    echo "Запуск всех экспериментов..."
    echo "ВНИМАНИЕ: Это займёт много времени!"
    echo ""
    read -p "Продолжить? (y/n): " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        python run_experiments.py --all
    fi
    read -p "Нажмите Enter для продолжения..."
}

function heterogeneity() {
    echo ""
    echo "Анализ влияния гетерогенности данных..."
    echo ""
    python run_experiments.py --heterogeneity
    read -p "Нажмите Enter для продолжения..."
}

function scaling() {
    echo ""
    echo "Анализ влияния количества клиентов..."
    echo ""
    python run_experiments.py --scale
    read -p "Нажмите Enter для продолжения..."
}

function custom() {
    echo ""
    echo "Кастомный запуск"
    echo ""
    read -p "Датасет (mnist/cifar10): " dataset
    read -p "Alpha для Dirichlet (0.1-10.0): " alpha
    echo ""
    python run_experiments.py --compare --dataset "$dataset" --alpha "$alpha"
    read -p "Нажмите Enter для продолжения..."
}

while true; do
    show_menu
    read -p "Введите номер: " choice
    
    case $choice in
        1) quick_test ;;
        2) full_experiments ;;
        3) heterogeneity ;;
        4) scaling ;;
        5) custom ;;
        0) echo ""; echo "Завершение работы..."; exit 0 ;;
        *) echo "Неверный выбор. Попробуйте снова." ;;
    esac
done
