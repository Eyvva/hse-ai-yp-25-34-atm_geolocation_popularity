"""Pydantic модели для запросов и ответов"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class AtmData(BaseModel):
    """Данные для предсказания популярности банкомата"""
    lat: float = Field(..., ge=-90, le=90, description="Широта", example=55.7558)
    lon: float = Field(..., ge=-180, le=180, description="Долгота", example=37.6173)
    atm_group: str = Field(..., description="Группа банкомата", example="premium")
    address_rus: str = Field(..., description="Адрес банкомата", 
                             example="Москва, Тверская улица, 1")


class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    predicted_index: float = Field(..., description="Предсказанный индекс популярности")
    request_id: int = Field(..., description="ID запроса")
    coordinates: Dict[str, float] = Field(..., description="Координаты банкомата")
    atm_group: str = Field(..., description="Группа банкомата")
    address: str = Field(..., description="Адрес банкомата (урезанный)")


class HealthResponse(BaseModel):
    """Ответ для проверки здоровья"""
    status: str = Field(..., description="Статус сервиса")
    pipeline_loaded: bool = Field(..., description="Загружен ли пайплайн")
    features_count: int = Field(..., description="Количество признаков")


class PipelineInfoResponse(BaseModel):
    """Информация о пайплайне"""
    steps: List[Dict[str, str]] = Field(..., description="Шаги пайплайна")
    features_count: int = Field(..., description="Количество признаков")
    features_sample: List[str] = Field(..., description="Пример признаков")