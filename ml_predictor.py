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

from config import ML_DATA_DIR, QualityMetrics

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

        # Фильтруем только те строки, в которых все таргет-значения конечны.
        # Это можно сделать так:
        y = df_targets.to_numpy()
        mask = np.isfinite(y).all(axis=1)
        if not mask.all():
            logging.info("Будут удалены строки с бесконечными значениями в таргетах.")
        # Для общего случая
        mask_combined = (df_features['analyze_channels'] == 0) & mask
        X_combined = X_processed[mask_combined]
        y_combined = df_targets[['psnr', 'ssim', 'ms_ssim']].to_numpy()[mask_combined]

        # Для канального случая
        mask_channels = (df_features['analyze_channels'] != 0) & mask
        X_channels = X_processed[mask_channels]
        # Выбираем все колонки, начинающиеся с метрики (например, "psnr_", "ssim_", "ms_ssim"...),
        # предполагаем, что они идут в нужном порядке
        y_channels = df_targets[
            [col for col in df_targets.columns if any(col.startswith(f"{m.value}") for m in QualityMetrics)]
        ].to_numpy()[mask_channels]

        # Обучение моделей
        combined_model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=42)
        ).fit(X_combined, y_combined)

        channels_model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=42)).fit(X_channels, y_channels)

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
            'original_width', 'original_height'
        ]
        categorical_features = ['method', 'channel'] # <- добавляем 'channel' в категориальные фичи
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

        if self.mode: # channels
            if self.channels_model is None:
                raise ValueError("Модель для анализа по каналам не загружена.")
            pred = self.channels_model.predict(processed)[0]
            # Возвращаем словарь, где ключи - названия метрик, значения - предсказанные значения.
            return {metric.value: val for metric, val in zip(QualityMetrics, pred)}
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
def extract_features_of_original_img(img: np.ndarray, channel_name: str = None) -> dict:
    """Извлекает признаки из 2D-изображения (канала)."""
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    if img.ndim != 2:
        raise ValueError("extract_features_of_original_img: входное изображение должно быть 2D (канал)")

    # Статистические признаки
    features = {'contrast': float(np.std(img)), 'variance': float(np.var(img)), 'entropy': shannon_entropy(img)}

    # Вейвлет-признаки (Haar)
    coefficients = pywt.dwt2(img, 'haar')
    c_a, (c_h, c_v, c_d) = coefficients
    features['wavelet_energy'] = np.sum(c_a**2 + c_h**2 + c_v**2 + c_d**2) / img.size
    logging.debug(f"Wavelet energy ({channel_name}): {features['wavelet_energy']:.4f}")

    # GLCM-признаки
    glcm = graycomatrix((img * 255).astype(np.uint8),
                        distances=[1],
                        angles=[0],
                        symmetric=True,
                        normed=True)
    features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_energy']   = graycoprops(glcm, 'energy')[0, 0]

    return features
