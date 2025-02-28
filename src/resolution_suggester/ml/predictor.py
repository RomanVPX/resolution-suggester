# ml/predictor.py
from ..i18n import _
import logging
import os
from pathlib import Path
from typing import Any, Dict
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import pywt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import ML_MODELS_DIR, QualityMetrics, PSNR_IS_LARGE_AS_INF


class QuickPredictor:
    def __init__(self, model_dir: Path = ML_MODELS_DIR):
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
            model_path = self.model_dir / "model_channels.joblib"
        else:
            model_path = self.model_dir / "model_combined.joblib"

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

        # Замена бесконечных значений на большое число (для PSNR)
        df_targets.replace(np.inf, PSNR_IS_LARGE_AS_INF, inplace=True)

        preprocessor = self._get_preprocessor()
        x_processed = preprocessor.fit_transform(df_features)

        # Удаление строк с бесконечными значениями
        y = df_targets.to_numpy()
        mask = np.isfinite(y).all(axis=1)
        if not mask.all():
            logging.info("Будут удалены строки с бесконечными значениями в таргетах.")
        mask_combined = (df_features['analyze_channels'] == 0) & mask
        x_combined = x_processed[mask_combined]
        y_combined = df_targets[['psnr', 'ssim', 'ms_ssim', 'tdpr']].to_numpy()[mask_combined]
        mask_channels = (df_features['analyze_channels'] != 0) & mask
        x_channels = x_processed[mask_channels]
        y_channels = df_targets[
            [col for col in df_targets.columns if any(col.startswith(f"{m.value}") for m in QualityMetrics)]
        ].to_numpy()[mask_channels]

        # Обучение моделей
        combined_model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=42)
        ).fit(x_combined, y_combined)

        channels_model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=42)).fit(x_channels, y_channels)

        self.model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(preprocessor, self.model_dir / "preprocessor.joblib")
        joblib.dump(combined_model, self.model_dir / "model_combined.joblib")
        joblib.dump(channels_model, self.model_dir / "model_channels.joblib")
        logging.info("Модели обучены и сохранены.")

    @staticmethod
    def _get_preprocessor() -> ColumnTransformer:
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
        Predicts metrics using models.
        Returns dictionary with keys corresponding to metrics (from config.QualityMetrics)
        """
        try:
            if self.preprocessor is None:
                raise ValueError(_("Preprocessor is not loaded"))
            df = pd.DataFrame([features])
            processed = self.preprocessor.transform(df)
            if self.mode:  # channels
                if self.channels_model is None:
                    raise ValueError(_("Channel model is not loaded"))
                pred = self.channels_model.predict(processed)[0]
                return {metric: val for metric, val in zip(QualityMetrics, pred)}
            else:
                if self.combined_model is None:
                    raise ValueError(_("Combined model is not loaded"))
                pred = self.combined_model.predict(processed)[0]
                return {
                    'psnr': pred[0],
                    'ssim': pred[1],
                    'ms_ssim': pred[2],
                    'tdpr': pred[3]
                }
        except Exception as e:
            logging.error(f"Ошибка при предсказании: {e}")
            logging.debug(f"Размерность входных данных: {processed.shape}, ключи признаков: {list(features.keys())}")
            return {m.value: 0.0 for m in QualityMetrics}


@lru_cache(maxsize=32)
def calculate_wavelet_features(img_bytes, height, width):
    """
    Кэшированное вычисление вейвлет-признаков.

    Args:
        img_bytes: Байтовое представление изображения
        height, width: Размеры изображения

    Returns:
        Энергия вейвлет-преобразования
    """
    # Восстанавливаем массив из байтов
    img = np.frombuffer(img_bytes, dtype=np.float32).reshape(height, width)

    # Вычисляем вейвлет-коэффициенты
    coefficients = pywt.dwt2(img, 'haar')
    c_a, (c_h, c_v, c_d) = coefficients
    wavelet_energy = np.sum(c_a ** 2 + c_h ** 2 + c_v ** 2 + c_d ** 2) / img.size

    return wavelet_energy


@lru_cache(maxsize=32)
def calculate_glcm_features(img_bytes, height, width):
    """
    Кэшированное вычисление GLCM-признаков.

    Args:
        img_bytes: Байтовое представление изображения
        height, width: Размеры изображения

    Returns:
        Кортеж GLCM-признаков (контраст, энергия)
    """
    # Восстанавливаем массив из байтов
    img = np.frombuffer(img_bytes, dtype=np.float32).reshape(height, width)

    # Преобразуем для GLCM (требуется uint8)
    img_uint8 = (img * 255).astype(np.uint8)

    # Вычисляем GLCM-матрицу и признаки
    glcm = graycomatrix(img_uint8,
                        distances=[1],
                        angles=[0],
                        symmetric=True,
                        normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    return contrast, energy


def extract_features_of_original_img(img: np.ndarray) -> dict:
    """
    Извлекает признаки из изображения с использованием кэширования для тяжелых вычислений.

    Args:
        img: Изображение как numpy массив

    Returns:
        Словарь признаков
    """
    # Приводим к формату 2D-изображения (канала)
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    if img.ndim != 2:
        raise ValueError("extract_features_of_original_img: входное изображение должно быть 2D (канал)")

    # Убеждаемся, что данные имеют правильный тип для корректной сериализации
    img = img.astype(np.float32)

    # Подготавливаем изображение для кэширования
    img_bytes = np.ascontiguousarray(img).tobytes()
    height, width = img.shape

    # Статистические признаки (вычисляются быстро, не кэшируем)
    features = {'contrast': float(np.std(img)), 'variance': float(np.var(img)), 'entropy': shannon_entropy(img),
                'wavelet_energy': calculate_wavelet_features(img_bytes, height, width)}

    # Кэшированные GLCM-признаки
    glcm_contrast, glcm_energy = calculate_glcm_features(img_bytes, height, width)
    features['glcm_contrast'] = glcm_contrast
    features['glcm_energy'] = glcm_energy

    return features
