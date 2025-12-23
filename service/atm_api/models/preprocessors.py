"""Кастомные трансформеры для обработки данных"""

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Выбор нужных колонок"""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns].copy()


class AtmGroupPreprocessor(BaseEstimator, TransformerMixin):
    """Обработка atm_group"""
    def __init__(self):
        self.encoder = None
        
    def fit(self, X, y=None):
        X = X.copy()
        X['atm_group'] = X['atm_group'].astype(str).fillna('unknown').str.strip()
        
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(X[['atm_group']])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        X['atm_group'] = X['atm_group'].astype(str).fillna('unknown').str.strip()
        
        encoded = self.encoder.transform(X[['atm_group']])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(['atm_group'])
        )
        
        result = pd.concat([X.drop('atm_group', axis=1), encoded_df], axis=1)
        return result


class AddressRegionExtractor(BaseEstimator, TransformerMixin):
    """Извлечение региона из адреса"""
    
    def __init__(self):
        self.encoder = None
        self.rare_regions = None
    
    @staticmethod
    def _get_region(text):
        """Вспомогательная функция для извлечения региона"""
        if pd.isna(text):
            return 'unknown'
        
        try:
            text = str(text).strip()
            
            # Удаляем лишние пробелы
            text = re.sub(r'\s+', ' ', text)
            
            parts = text.split(', ')
            if len(parts) < 2:
                parts = text.split(' ')
            
            # Логика извлечения региона
            for i in range(len(parts)-1, -1, -1):
                part = parts[i].lower()
                if part != 'россия' and part != 'russia':
                    return parts[i]
            
            # Если не нашли регион, возвращаем предпоследнюю часть
            if len(parts) >= 2:
                return parts[-2]
            
            return 'unknown'
            
        except:
            return 'unknown'
        
    def fit(self, X, y=None):
        X = X.copy()
        
        # Извлекаем регионы
        X['region'] = X['address_rus'].apply(self._get_region)
        
        # Находим редкие регионы
        region_counts = X['region'].value_counts()
        self.rare_regions = region_counts[region_counts < 30].index.tolist()
        
        # Кодируем регионы
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(X[['region']])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Извлекаем регионы
        X['region'] = X['address_rus'].apply(self._get_region)
        
        # Заменяем редкие регионы на 'Other'
        X['region'] = X['region'].apply(
            lambda x: 'Other' if x in self.rare_regions else x
        )
        
        # Кодируем
        encoded = self.encoder.transform(X[['region']])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(['region'])
        )
        
        result = pd.concat([X.drop(['address_rus', 'region'], axis=1), encoded_df], axis=1)
        return result


class NumericCleaner(BaseEstimator, TransformerMixin):
    """Очистка числовых колонок"""
    def __init__(self, numeric_columns=None):
        self.numeric_columns = numeric_columns
        self.medians_ = {}
        
    def fit(self, X, y=None):
        if self.numeric_columns is None:
            # Ищем только числовые колонки, которые есть в X
            self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.numeric_columns:
            if col in X.columns:
                # Преобразуем в числа
                numeric_series = pd.to_numeric(X[col], errors='coerce')
                # Если все значения NaN, используем 0
                if numeric_series.isna().all():
                    self.medians_[col] = 0.0
                else:
                    self.medians_[col] = numeric_series.median()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.numeric_columns:
            if col in X.columns:
                # Преобразуем в числа
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Заполняем пропуски медианой
                if col in self.medians_:
                    X[col] = X[col].fillna(self.medians_[col])
                else:
                    X[col] = X[col].fillna(0.0)
        
        # Проверяем что нет NaN
        if X.select_dtypes(include=[np.number]).isna().any().any():
            # Заполняем оставшиеся NaN нулями
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(0)
        
        return X