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

from config import ML_TARGET_COLUMNS, ML_DATA_DIR


class QuickPredictor:
    def __init__(self, model_path: str = os.path.join(ML_DATA_DIR, 'model.joblib')):
        """Инициализация предиктора с путём к модели"""
        self.model_path = model_path
        self.pipeline = None

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

    def get_pipeline(self):
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
        :param targets_csv: CSV (или parquet) с метриками в колонками ('psnr', 'ssim', 'ms_ssim'...)
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

        # Проверяем наличие всех трёх колонок
        assert set(ML_TARGET_COLUMNS).issubset(df_targets.columns)

        # y = df_targets[['psnr', 'ssim', 'ms_ssim']].values  # shape: (N, 2)
        y = df_targets[ML_TARGET_COLUMNS].to_numpy()

        self.pipeline = self.get_pipeline()
        self.pipeline.fit(df_features, y)

        joblib.dump(self.pipeline, self.model_path)
        logging.info("Модель сохранена в %s", self.model_path)

    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Предсказывает PSNR, SSIM и MS_SSIM на одном примере (словарь).
        Возвращает {'psnr': float, 'ssim': float, 'ms_ssim': float }.
        """
        if not self.pipeline:
            raise ValueError("Модель не загружена. Сначала вызовите load().")

        df = pd.DataFrame([features])  # DataFrame из одного примера
        pred = self.pipeline.predict(df)  # shape: (1, 3)
        return {metric: pred[0, i] for i, metric in enumerate(ML_TARGET_COLUMNS)}


#############################################################################
# Простейшая функция-обёртка для извлечения признаков из np.ndarray
# Здесь — только пара фич (контраст и дисперсия), плюс категориальный признак 'method'.
#############################################################################

def extract_texture_features(img: np.ndarray, method: str) -> Dict[str, float]:
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
def extract_features_of_original_img(img: np.ndarray) -> dict:
    """Извлекает признаки только из оригинального изображения."""
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)  # конвертируем в grayscale
    else:
        img_gray = img

    # Базовые статистики
    contrast = float(np.std(img_gray))
    variance = float(np.var(img_gray))

    entropy = shannon_entropy(img_gray)

    # TODO: проверить корректность этого метода вычисления энтропии:
    # img_uint8 = (np.clip(img_gray, 0, 1) * 255).astype(np.uint8)
    # hist = np.histogram(img_uint8, bins=256, range=(0, 255))[0]
    # hist_norm = hist / hist.sum()
    # entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

    # Вейвлет-признаки (Haar, уровень 1)
    coefficients = pywt.dwt2(img_gray, 'haar')
    c_a, (c_h, c_v, c_d) = coefficients
    # wavelet_energy = np.sum(c_a**2 + c_h**2 + c_v**2 + c_d**2) / (img_gray.size * 1e6)
    wavelet_energy = np.sum(c_a**2 + c_h**2 + c_v**2 + c_d**2) / img_gray.size
    logging.debug("Wavelet energy: %f", wavelet_energy)

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