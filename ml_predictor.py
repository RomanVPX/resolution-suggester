# ml_predictor.py
import os
import logging

import joblib
import pywt
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Any

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from config import ML_TARGET_COLUMNS, ML_DATA_DIR, QualityMetrics, CHANNEL_COLUMNS


class QuickPredictor:
    def __init__(self, model_dir: Path = ML_DATA_DIR / "models"):
        """Инициализация предиктора с путём к моделям."""
        self.model_dir = model_dir
        self.pipeline = None
        self.mode = False  # False: без каналов, True: с каналами

    def set_mode(self, analyze_channels: bool):
        """Устанавливает режим анализа."""
        self.mode = analyze_channels
        if self.mode:
            self.model_path = self.model_dir / "channels" / "model.joblib"
        else:
            self.model_path = self.model_dir / "combined" / "model.joblib"

    def load(self) -> bool:
        """Загружает модель в зависимости от режима."""
        if self.mode:
            model_path = self.model_dir / "channels" / "model.joblib"
        else:
            model_path = self.model_dir / "combined" / "model.joblib"

        if not model_path.exists():
            logging.warning(f"Файл модели не найден: {model_path}")
            return False
        self.pipeline = joblib.load(model_path)
        return True

    def train(self, features_csv: str, targets_csv: str) -> None:
        """
        Обучает модели на данных.
        Создаёт две модели: одну для общего анализа, другую для анализа по каналам.
        """
        if not os.path.exists(features_csv) or not os.path.exists(targets_csv):
            logging.error("Не найдены файлы с фичами/таргетами.")
            return

        # Загрузка данных
        df_features = pd.read_csv(features_csv)
        df_targets = pd.read_csv(targets_csv)

        # Предобработка фичей
        preprocessor = self._get_preprocessor()
        X_processed = preprocessor.fit_transform(df_features)

        # Разделение данных на два режима
        mask = df_features['analyze_channels'] == 0
        X_combined = X_processed[mask]
        X_channels = X_processed[~mask]

        # Подготовка таргетов
        y_combined = df_targets[['psnr', 'ssim', 'ms_ssim']][mask]
        y_channels = df_targets[
            [col for col in df_targets.columns if any(col.startswith(f"{m.value}_") for m in QualityMetrics)]
        ][~mask]

        # Обучение моделей
        combined_model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=42)
        ).fit(X_combined, y_combined)

        channels_model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=42)
        ).fit(X_channels, y_channels)

        # Сохранение моделей
        (self.model_dir / "combined").mkdir(parents=True, exist_ok=True)
        (self.model_dir / "channels").mkdir(parents=True, exist_ok=True)

        joblib.dump(preprocessor, self.model_dir / "preprocessor.joblib")
        joblib.dump(combined_model, self.model_dir / "combined" / "model.joblib")
        joblib.dump(channels_model, self.model_dir / "channels" / "model.joblib")
        logging.info("Модели обучены и сохранены.")

    def _get_preprocessor(self) -> ColumnTransformer:
        """Создаёт препроцессор фичей."""
        numeric_features = [
            'contrast', 'variance', 'entropy',
            'wavelet_energy', 'glcm_contrast',
            'glcm_energy', 'scale_factor',
            'original_width', 'original_height',
            'num_channels'
        ]
        categorical_features = ['method']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'
        )
        return preprocessor

    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Предсказывает метрики на одном примере (словарь).
        Возвращает соответствующие метрики в зависимости от режима.
        """
        if not self.pipeline:
            raise ValueError("Модель не загружена. Сначала вызовите load().")

        df = pd.DataFrame([features])  # DataFrame из одного примера
        processed = df  # Предполагается, что фичи уже соответствуют препроцессору

        pred = self.pipeline.predict(processed)[0]

        if self.mode:
            # Возвращаем метрики по каналам
            predictions = {}
            metric_channel_pairs = [
                (metric.value, ch) for metric in QualityMetrics for ch in CHANNEL_COLUMNS
            ]
            for i, (metric, ch) in enumerate(metric_channel_pairs):
                predictions[f"{metric}_{ch}"] = pred[i]
            return predictions
        else:
            # Возвращаем общие метрики
            return {
                'psnr': pred[0],
                'ssim': pred[1],
                'ms_ssim': pred[2]
            }


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