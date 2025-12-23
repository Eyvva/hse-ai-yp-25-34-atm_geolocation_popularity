# ATM Popularity Prediction API

FastAPI сервис для прогнозирования популярности банкоматов с использованием машинного обучения.

## 🚀 Быстрый старт

1. Установка зависимостей
```bash
# Активация виртуального окружения
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

2. Обучение модели (если нет файла модели)
```bash
python pipelines/train_pipeline.py
```

3. Запуск сервера
```bash
python app.py
```

Сервер запустится на `http://localhost:8000`