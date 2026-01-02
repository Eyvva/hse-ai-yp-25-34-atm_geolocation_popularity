"""Конфигурация приложения"""

import os
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "project_model.pkl"

# Настройки API
API_TITLE = "ATM Popularity Prediction API"
API_DESCRIPTION = "API для прогнозирования популярности банкоматов с использованием пайплайна"
API_VERSION = "1.0.0"

# Настройки сервера
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True

# Настройки логирования
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"