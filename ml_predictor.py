# ml_predictor.py
import os
import logging
import joblib
import pywt
import numpy as np
import pandas as pd

from typing import Dict, Any

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
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
        numeric_features = [
            'contrast', 'variance', 'entropy',
            'wavelet_energy', 'glcm_contrast', 'glcm_energy',
            'scale_factor'
        ]
        categorical_features = ['method']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ],
            remainder='drop'
        )

        base_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model = MultiOutputRegressor(base_model)

        return make_pipeline(preprocessor, model)

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

        # Удаляем строки с пропусками
        df_features = df_features.dropna()
        df_targets = df_targets.dropna()

        # Проверяем наличие обеих колонок
        assert {'psnr', 'ssim'}.issubset(df_targets.columns)

        # Условимся, что df_targets = [psnr, ssim]
        # y = df_targets[['psnr', 'ssim']].values  # shape: (N, 2)
        y = df_targets[['psnr', 'ssim']].to_numpy()

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


#############################################################################
# Извлечение набора признаков из оригинального изображения
#############################################################################
def extract_features_original(img: np.ndarray) -> dict:
    """Извлекает признаки только из оригинального изображения."""
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)  # конвертируем в grayscale
    else:
        img_gray = img

    # Базовые статистики
    contrast = float(np.std(img_gray))
    variance = float(np.var(img_gray))
    entropy = shannon_entropy(img_gray)

    # Вейвлет-признаки (Haar, уровень 1)
    coefficients = pywt.dwt2(img_gray, 'haar')
    c_a, (c_h, c_v, c_d) = coefficients
    wavelet_energy = np.sum(c_a**2 + c_h**2 + c_v**2 + c_d**2) / img_gray.size

    # GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(
        (img_gray * 255).astype(np.uint8),
        distances=[1],
        angles=[0],
        symmetric=True,
        normed=True
    )
    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    glcm_energy = graycoprops(glcm, 'energy')[0, 0]

    return {
        'contrast': contrast,
        'variance': variance,
        'entropy': entropy,
        'wavelet_energy': wavelet_energy,
        'glcm_contrast': glcm_contrast,
        'glcm_energy': glcm_energy,
    }