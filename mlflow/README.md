# MLflow chekpoint: Gradient Boosting Regression

## 📌 Описание проекта

Проект по версионированию экспериментов машинного обучения с использованием:
- **MLflow** для трекинга экспериментов
- **MinIO S3** для хранения артефактов
- **Docker** для изоляции окружения
- **Optuna** для оптимизации гиперпараметров

### Целевая задача
Регрессия для предсказания целевой переменной на основе географических и инфраструктурных признаков.

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone <https://github.com/Eyvva/hse-ai-yp-25-34-atm_geolocation_popularity.git>
cd mlflow
```

### 2. Натсройка окружения

### Python
```bash
python -m venv mlflow_env
source mlflow_env/bin/activate  # Linux/Mac
# или
mlflow_env\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Docker
```bash
docker-compose up -d
```
### 3. Настройка .env

Скопируйте файл .env.example в корне проекта:

```bash
cp .env.example
```
Отредактируй при необходимости


### 4. Запуск Jupyter
jupyter notebook notebooks/mlflow.ipynb

#### MLFlow UI
После запуска docker-compose up -d:

| Сервис | URL | Логин/Пароль |
|--------|-----|--------------|
| MLflow UI | http://localhost:5050 | — |
| MinIO Console | http://localhost:9001 | `admin` / `password` |
