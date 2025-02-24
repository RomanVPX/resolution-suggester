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

from config import ML_DATA_DIR, QualityMetrics, CHANNEL_COLUMNS

class QuickPredictor:
    def __init__(self, model_dir: Path = ML_DATA_DIR / "models"):
        """Инициализация предиктора с путём к моделям."""
        self.model_dir = model_dir
        self.combined_model = None    # Модель для общего анализа (без каналов)
        self.channels_model = None    # Модель для анализа по каналам
        self.preprocessor = None      # Препроцессор фич, одинаковый для обоих режимов
        self.mode = False             # False: без каналов, True: с каналами

    def set_mode(self, analyze_channels: bool):
        """Устанавливает режим анализа: с каналами (True) или без (False)."""
        self.mode = analyze_channels

    def load(self) -> bool:
        """Загружает препроцессор и модель согласно текущему режиму."""
        preprocessor_path = self.model_dir / "preprocessor.joblib"
        if not preprocessor_path.exists():
            logging.warning(f"Файл препроцессора не найден: {preprocessor_path}")
            return False
        self.preprocessor = joblib.load(preprocessor_path)

        if self.mode:
            model_path = self.model_dir / "channels" / "model.joblib"
        else:
            model_path = self.model_dir / "combined" / "model.joblib"

        if not model_path.exists():
            logging.warning(f"Файл модели не найден: {model_path}")
            return False

        if self.mode:
            self.channels_model = joblib.load(model_path)
        else:
            self.combined_model = joblib.load(model_path)
        return True

    def train(self, features_csv: str, targets_csv: str) -> None:
        """
        Обучает модели на данных.
        Создаёт две модели: одну для общего анализа, другую для анализа по каналам.
        """
        if not os.path.exists(features_csv) or not os.path.exists(targets_csv):
            logging.error("Не найдены файлы с фичами/таргетами.")
            return

        df_features = pd.read_csv(features_csv)
        df_targets = pd.read_csv(targets_csv)

        preprocessor = self._get_preprocessor()
        X_processed = preprocessor.fit_transform(df_features)

        mask = df_features['analyze_channels'] == 0
        X_combined = X_processed[mask]
        X_channels = X_processed[~mask]

        y_combined = df_targets[['psnr', 'ssim', 'ms_ssim']][mask]
        y_channels = df_targets[
            [col for col in df_targets.columns if any(col.startswith(f"{m.value}_") for m in QualityMetrics)]
        ][~mask]

        combined_model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=42)
        ).fit(X_combined, y_combined)

        channels_model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=42)
        ).fit(X_channels, y_channels)

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
        Если режим установлен в True (анализ по каналам), возвращает:
            {'psnr_R': value, 'psnr_G': value, ..., 'min_psnr': value, ...}
        Иначе – возвращает общий результат:
            {'psnr': value, 'ssim': value, 'ms_ssim': value}
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor не загружен.")
        df = pd.DataFrame([features])
        processed = self.preprocessor.transform(df)

        if self.mode:
            if self.channels_model is None:
                raise ValueError("Модель для анализа по каналам не загружена.")
            pred = self.channels_model.predict(processed)[0]
            predictions = {}
            metric_channel_pairs = [(metric.value, ch) for metric in QualityMetrics for ch in CHANNEL_COLUMNS]
            for i, (metric, ch) in enumerate(metric_channel_pairs):
                predictions[f"{metric}_{ch}"] = pred[i]
            for metric in QualityMetrics:
                min_val = min(predictions[f"{metric.value}_{ch}"] for ch in CHANNEL_COLUMNS)
                predictions[f"min_{metric.value}"] = min_val
            return predictions
        else:
            if self.combined_model is None:
                raise ValueError("Модель для общего анализа не загружена.")
            pred = self.combined_model.predict(processed)[0]
            return {
                'psnr': pred[0],
                'ssim': pred[1],
                'ms_ssim': pred[2]
            }

#############################################################################
# Простейшая функция-обёртка для извлечения признаков из np.ndarray
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
def extract_features_of_original_img(img: np.ndarray) -> dict:
    """Извлекает признаки только из оригинального изображения."""
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)  # конвертируем в grayscale
    else:
        img_gray = img
    contrast = float(np.std(img_gray))
    variance = float(np.var(img_gray))
    entropy = shannon_entropy(img_gray)
    coefficients = pywt.dwt2(img_gray, 'haar')
    c_a, (c_h, c_v, c_d) = coefficients
    wavelet_energy = np.sum(c_a**2 + c_h**2 + c_v**2 + c_d**2) / img_gray.size
    logging.debug("Wavelet energy: %f", wavelet_energy)
    glcm = graycomatrix((img_gray * 255).astype(np.uint8),
                        distances=[1],
                        angles=[0],
                        symmetric=True,
                        normed=True)
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
