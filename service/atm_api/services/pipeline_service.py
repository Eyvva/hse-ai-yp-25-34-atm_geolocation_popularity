"""Сервис для работы с пайплайном"""

import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

from config import MODEL_PATH

logger = logging.getLogger(__name__)


class PipelineService:
    """Сервис для работы с обученным пайплайном"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.feature_names = []
        self.load_pipeline()
    
    def load_pipeline(self) -> bool:
        """Загрузка пайплайна из файла"""
        try:
            saved_data = joblib.load(self.model_path)
            self.pipeline = saved_data['pipeline']
            self.feature_names = saved_data['feature_names']
            
            logger.info(f"Пайплайн загружен из {self.model_path}")
            logger.info(f"Признаков: {len(self.feature_names)}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Файл пайплайна не найден: {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Ошибка загрузки пайплайна: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Проверка готовности сервиса"""
        return self.pipeline is not None
    
    def predict(self, lat: float, lon: float, atm_group: str, address_rus: str) -> float:
        """Выполнение предсказания"""
        if not self.is_ready():
            raise ValueError("Пайплайн не загружен")
        
        try:
            # Подготавливаем данные
            input_data = pd.DataFrame([{
                'lat': lat,
                'long': lon,  # используем long как в данных
                'atm_group': atm_group,
                'address_rus': address_rus
            }])
            
            # Делаем предсказание
            prediction = self.pipeline.predict(input_data)
            
            return float(prediction[0])
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Получение информации о пайплайне"""
        if not self.is_ready():
            return {"error": "Пайплайн не загружен"}
        
        steps_info = []
        for step_name, step in self.pipeline.steps:
            steps_info.append({
                "name": step_name,
                "type": type(step).__name__
            })
        
        return {
            "steps": steps_info,
            "features_count": len(self.feature_names),
            "features_sample": self.feature_names[:10] if self.feature_names else []
        }


# Глобальный экземпляр сервиса
pipeline_service = PipelineService()