# ml_predictor.py
import os
import logging
import joblib
import numpy as np
import pandas as pd

from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

#############################################################################
# Пример минимальной модели, которая:
# 1) Читает признаки из DataFrame (или dict), где есть:
#   - 'contrast'
#   - 'variance'
#   - 'method'
# 2) Предсказывает PSNR и SSIM одним RandomForest'ом с 2 выходами
#############################################################################

class QuickPredictor:
    def __init__(self, model_path: str = 'quick_model.joblib'):
        self.model_path = model_path
        self.pipeline = None

    def _build_pipeline(self):
        # Допустим, numeric_features = ['contrast', 'variance']
        # а categorical_features = ['method']
        numeric_features = ['contrast', 'variance']
        categorical_features = ['method']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ],
            remainder='drop'
        )

        # RandomForest для 2 целевых выходов (PSNR, SSIM)
        # будем трактовать это как задачу многомерной регрессии,
        # куда мы подаём y = [[psnr, ssim], ...].
        rf = RandomForestRegressor(
            n_estimators=30,
            max_depth=8,
            random_state=42
        )

        pipeline = make_pipeline(
            preprocessor,
            rf
        )
        return pipeline

    def train(self, features_csv: str, targets_csv: str) -> None:
        """
        Обучает модель на данных.
        :param features_csv: CSV (или parquet) с признаками
        :param targets_csv: CSV (или parquet) с колонками 'psnr' и 'ssim'
        """
        if not os.path.exists(features_csv) or not os.path.exists(targets_csv):
            logging.error("Не найдены файлы с фичами/таргетами: %s, %s", features_csv, targets_csv)
            return

        # Загружаем датафреймы
        df_features = pd.read_csv(features_csv) if features_csv.endswith('.csv') \
            else pd.read_parquet(features_csv)
        df_targets = pd.read_csv(targets_csv)   if targets_csv.endswith('.csv') \
            else pd.read_parquet(targets_csv)

        # Условимся, что df_targets = [psnr, ssim]
        y = df_targets[['psnr', 'ssim']].values  # shape: (N, 2)

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df_features, y)

        joblib.dump(self.pipeline, self.model_path)
        logging.info("Модель сохранена в %s", self.model_path)

    def load(self) -> bool:
        """
        Загружает модель из self.model_path.
        Возвращает True, если модель успешно загружена, иначе False.
        """
        if not os.path.exists(self.model_path):
            logging.warning("Файл модели не найден: %s", self.model_path)
            return False
        self.pipeline = joblib.load(self.model_path)
        return True

    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Предсказывает PSNR и SSIM на одном примере (словарь).
        Возвращает {'psnr': float, 'ssim': float}.
        """
        if not self.pipeline:
            raise ValueError("Модель не загружена. Сначала вызовите load().")

        df = pd.DataFrame([features])  # DataFrame из одного примера
        pred = self.pipeline.predict(df)  # shape: (1,2)
        return {
            'psnr': float(pred[0, 0]),
            'ssim': float(pred[0, 1])
        }

#############################################################################
# Простейшая функция-обёртка для извлечения признаков из np.ndarray
# Здесь — только пара фич (контраст и дисперсия), плюс категориальный признак 'method'.
#############################################################################

def extract_texture_features(img: np.ndarray, method: str) -> Dict[str, float]:
    """
    Извлекаем простейшие признаки из изображения.
    Вы можете расширять этот список, добавлять wavelet'ы, entopy и т.п.
    """
    contrast = float(np.std(img))
    variance = float(np.var(img))
    return {
        'contrast': contrast,
        'variance': variance,
        'method': method
    }
