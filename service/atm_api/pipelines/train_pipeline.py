"""Обучение и сохранение пайплайна"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from models.preprocessors import (
    ColumnSelector,
    AtmGroupPreprocessor,
    AddressRegionExtractor,
    NumericCleaner
)


def create_pipeline():
    """Создание полного пайплайна обработки данных и модели"""
    
    print("Создание пайплайна...")
    
    # 1. Определяем этапы пайплайна
    pipeline = Pipeline([
        # Шаг 1: Выбор нужных колонок (БЕЗ target)
        ('column_selector', ColumnSelector(['lat', 'long', 'atm_group', 'address_rus'])),
        
        # Шаг 2: Очистка числовых данных (ТОЛЬКО lat и long)
        ('numeric_cleaner', NumericCleaner(['lat', 'long'])),
        
        # Шаг 3: Обработка atm_group
        ('atm_group_processor', AtmGroupPreprocessor()),
        
        # Шаг 4: Извлечение и обработка региона из адреса
        ('address_processor', AddressRegionExtractor()),
        
        # Шаг 5: Масштабирование числовых признаков
        ('scaler', StandardScaler()),
        
        # Шаг 6: Модель
        ('model', Ridge(alpha=0.5))
    ])
    
    return pipeline


def train_and_save_pipeline(data_path='data/train_final.csv', 
                           output_path='pipeline_model.pkl'):
    """Обучение и сохранение пайплайна"""
    
    print("Загрузка данных...")
    train_df = pd.read_csv(data_path)
    
    print(f"Размер данных: {train_df.shape}")
    print(f"Доступные колонки: {train_df.columns.tolist()}")
    
    # Проверяем наличие нужных колонок
    required_cols = ['lat', 'long', 'atm_group', 'address_rus', 'target']
    available_cols = [col for col in required_cols if col in train_df.columns]
    print(f"Найдены колонки: {available_cols}")
    
    if len(available_cols) < len(required_cols):
        print(f"Внимание: не все нужные колонки найдены!")
    
    # Используем только доступные колонки
    train_df = train_df[available_cols]
    
    # Проверяем пропуски в target
    print(f"\nПроверяем пропуски в данных:")
    print(f"   Пропуски в target: {train_df['target'].isna().sum()} "
          f"({train_df['target'].isna().sum()/len(train_df)*100:.1f}%)")
    
    # Заполняем пропуски в target медианой
    if train_df['target'].isna().any():
        target_median = train_df['target'].median()
        train_df['target'] = train_df['target'].fillna(target_median)
        print(f"Заполнил пропуски в target медианой: {target_median:.4f}")
    
    # Разделяем данные
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    
    print(f"\nX размер: {X.shape}")
    print(f"y размер: {y.shape}")
    
    # Создаем пайплайн
    pipeline = create_pipeline()
    
    print("\nОбучение пайплайна...")
    try:
        pipeline.fit(X, y)
        print("Пайплайн успешно обучен")
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        return None
    
    # Получаем имена признаков после обработки
    feature_names = get_feature_names(pipeline, X.head(3))
    
    print("\nСохранение пайплайна...")
    # Сохраняем пайплайн и дополнительную информацию
    saved_data = {
        'pipeline': pipeline,
        'feature_names': feature_names,
        'data_shape_before': train_df.shape,
        'required_columns': ['lat', 'long', 'atm_group', 'address_rus']
    }
    
    joblib.dump(saved_data, output_path)
    
    print("\n" + "="*60)
    print("ПАЙПЛАЙН СОХРАНЕН!")
    print("="*60)
    print(f"Признаков после обработки: {len(feature_names)}")
    
    if feature_names and len(feature_names) > 0:
        print(f"Пример признаков (первые 10):")
        for i, name in enumerate(feature_names[:10]):
            print(f"   {i+1:2}. {name}")
    
    # Тестовое предсказание
    print("\nТестовое предсказание на первых 3 строках:")
    try:
        test_predictions = pipeline.predict(X.head(3))
        for i, pred in enumerate(test_predictions):
            actual = y.iloc[i]
            print(f"  Пример {i+1}: Предсказано = {pred:.4f}, Фактическое = {actual:.4f}")
    except Exception as e:
        print(f"Не удалось сделать предсказание: {e}")
    
    return pipeline


def get_feature_names(pipeline, X_sample):
    """Получение читаемых имен признаков после обработки пайплайном"""
    try:
        # Получаем трансформеры из ColumnTransformer
        preprocessor = pipeline.named_steps['scaler']  # наш StandardScaler
        # Нам нужно получить признаки ДО масштабирования
        
        # Проходим через все шаги до scaler
        transformed_data = X_sample.copy()
        
        for step_name, step in pipeline.steps:
            if step_name == 'scaler':  # останавливаемся перед scaler
                break
            if hasattr(step, 'transform'):
                transformed_data = step.transform(transformed_data)
        
        # Теперь у нас должен быть DataFrame с правильными именами
        if isinstance(transformed_data, pd.DataFrame):
            return transformed_data.columns.tolist()
        else:
            # Если всё же массив, пробуем получить имена из трансформеров
            feature_names = []
            
            # Получаем encoder из AtmGroupPreprocessor
            atm_processor = pipeline.named_steps['atm_group_processor']
            if hasattr(atm_processor, 'encoder'):
                atm_features = atm_processor.encoder.get_feature_names_out(['atm_group'])
                feature_names.extend(atm_features)
            
            # Получаем encoder из AddressRegionExtractor
            address_processor = pipeline.named_steps['address_processor']
            if hasattr(address_processor, 'encoder'):
                region_features = address_processor.encoder.get_feature_names_out(['region'])
                feature_names.extend(region_features)
            
            # Добавляем числовые признаки
            feature_names = ['lat', 'long'] + feature_names
            
            return feature_names
            
    except Exception as e:
        print(f"Не удалось получить имена признаков: {e}")
        # Создаем простые имена
        return [f'feature_{i}' for i in range(2361)]  # 2361 из вывода


if __name__ == "__main__":
    # Запуск обучения из командной строки
    train_and_save_pipeline(data_path='/data/train_final.csv')