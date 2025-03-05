# core/image_analyzer.py
import os

import pandas as pd

from ..i18n import _
import argparse
import concurrent.futures
from typing import Tuple

import numpy as np

# from ..i18n import _
from PIL import Image
import logging
from typing import Optional

from ..config import INTERPOLATION_METHOD_UPSCALE, InterpolationMethods, QualityMetrics, \
    PSNR_IS_LARGE_AS_INF, INTERMEDIATE_DIR, QualityLevelHints, QUALITY_LEVEL_HINTS_DESCRIPTIONS
from ..core.image_processing import get_resize_function
from ..ml.predictor import QuickPredictor, extract_features_of_original_img
from ..utils.reporters import IReporter
from .image_loader import load_image
from .metrics import compute_resolutions, calculate_metrics
from ..utils.reporting import QualityHelper, ConsoleReporter


class ImageAnalyzer:
    """
    Class encapsulating image analysis logic.
    Provides methods for analyzing individual files and file groups.
    """

    def __init__(self, args: argparse.Namespace, reporters: list[IReporter] = None):
        """
        Initializes the image analyzer.

        Args:
            args: Command line arguments
            reporters: List of reporting objects (CSV, JSON, etc.)
        """
        self.args = args
        self.reporters = reporters or []
        self.predictor = None

        # Если используется ML-предсказание, инициализируем предиктор
        if args.ml:
            self.predictor = self._initialize_predictor()

        # Инициализируем функции изменения размера, если они будут использоваться
        self.resize_fn = None
        self.resize_fn_upscale = None
        if not args.ml:
            try:
                self.resize_fn = get_resize_function(args.interpolation)
                self.resize_fn_upscale = get_resize_function(INTERPOLATION_METHOD_UPSCALE)
            except ValueError as e:
                logging.error(f"Ошибка при инициализации функции ресайза: {e}")
                raise

    def _initialize_predictor(self) -> Optional[QuickPredictor]:
        """Initializes and configures the predictor for ML predictions."""
        predictor = QuickPredictor()
        predictor.set_mode(self.args.channels)
        if not predictor.load():
            logging.info(_("ML model not found, the actual metrics will be calculated."))
            return None
        return predictor

    def analyze_files(self, files: list[str]) -> None:
        """
        Analyzes a list of files.

        Args:
            files: List of file paths to analyze
        """
        if self.args.no_parallel:
            self._analyze_files_sequential(files)
        else:
            self._analyze_files_parallel(files)

    def _analyze_files_sequential(self, files: list[str]) -> None:
        """Sequential file analysis."""
        from tqdm import tqdm
        for file_path in tqdm(files, desc="Обработка файлов", leave=False):
            try:
                results, meta = self.analyze_file(file_path)
                if results:
                    self._report_results(file_path, results, meta)
            except Exception as e:
                logging.error(f"Ошибка обработки файла {file_path}: {e}")
                logging.debug("Детали:", exc_info=True)

    def _analyze_files_parallel(self, files: list[str]) -> None:
        """Parallel file analysis using ProcessPoolExecutor."""
        # Преобразуем аргументы в словарь для передачи
        args_dict = vars(self.args)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.threads) as executor:
            future_to_file = {
                executor.submit(process_file_for_analyzer, args_dict, file_path): file_path
                for file_path in files
            }
            # Используем tqdm для отображения прогресса
            from tqdm import tqdm
            for future in tqdm(concurrent.futures.as_completed(future_to_file),
                               total=len(files), desc="Обработка файлов", leave=False):
                file_path = future_to_file[future]
                try:
                    results, meta = future.result()
                    if results:
                        self._report_results(file_path, results, meta)
                except Exception as e:
                    logging.error(f"Ошибка получения результата для {file_path}: {e}")


    def analyze_file(self, file_path: str) -> Tuple[Optional[list], Optional[dict]]:
        """
        Analyzes a single image.

        Args:
            file_path: Path to the image file

        Returns:
            Tuple (results, metadata) or (None, None) in case of error
        """
        try:
            # Загрузка изображения
            image_load_result = load_image(file_path)
            if image_load_result.error or image_load_result.data is None:
                logging.error(f"Ошибка загрузки изображения {file_path}: {image_load_result.error}")
                return None, None

            img_original = image_load_result.data
            max_val = image_load_result.max_value
            channels = image_load_result.channels
            height, width = img_original.shape[:2]

            # Проверка минимального размера
            if height < self.args.min_size or width < self.args.min_size:
                logging.info(
                    f"Пропуск {file_path} (размер {width}x{height}) - меньше, чем min_size = {self.args.min_size}")
                return None, None

            # Начинаем с оригинального разрешения
            results = [self._create_original_entry(width, height, channels)]

            # Вычисляем список разрешений для анализа
            resolutions = compute_resolutions(width, height, self.args.min_size)

            # Анализируем каждое разрешение
            for (w, h) in resolutions:
                if w == width and h == height:  # Пропускаем оригинальное разрешение
                    continue

                # Анализ с реальным изменением размера (не ML)
                if not self.predictor:
                    results_entry = self._analyze_resize_real(img_original, max_val, channels, w, h, width, height,
                                                              file_path)
                # Анализ с ML-предсказанием
                else:
                    results_entry = self._analyze_resize_ml(img_original, channels, w, h, width, height)

                results.append(results_entry)

            return results, {'max_val': max_val, 'channels': channels}

        except MemoryError:
            logging.error(f"Недостаточно памяти для обработки файла {file_path}")
            return None, None
        except Exception as e:
            logging.error(f"Ошибка при анализе файла {file_path}: {e}")
            logging.debug("Детали:", exc_info=True)
            return None, None

    def _analyze_resize_real(self, img_original, max_val, channels, w, h, orig_width, orig_height, file_path):
        """Analyzes resize with real metrics calculation."""

        img_downscaled = self.resize_fn(img_original, w, h)
        if self.args.save_im_down:
            self._save_intermediate(img_downscaled, file_path, w, h,
                                    InterpolationMethods(self.args.interpolation), '')

        img_upscaled = self.resize_fn_upscale(img_downscaled, orig_width, orig_height)
        if self.args.save_im_up:
            self._save_intermediate(img_upscaled, file_path, w, h,
                                    InterpolationMethods(self.args.interpolation), 'upscaled')

        if self.args.channels: # Поканальный анализ
            channels_metrics = calculate_metrics(
                QualityMetrics(self.args.metric),
                img_original, img_upscaled, max_val,
                channels, no_gpu=self.args.no_gpu
            )
            channels_metrics = postprocess_metric_value(channels_metrics, self.args.metric)
            min_metric = min(channels_metrics.values())
            hint = QualityHelper.get_hint(min_metric, QualityMetrics(self.args.metric))
            return f"{w}x{h}", channels_metrics, min_metric, hint
        else:
            metric_value = calculate_metrics(
                QualityMetrics(self.args.metric),
                img_original, img_upscaled, max_val,
                no_gpu=self.args.no_gpu
            )
            metric_value = postprocess_metric_value(metric_value, self.args.metric)
            hint = QualityHelper.get_hint(metric_value, QualityMetrics(self.args.metric))
            return f"{w}x{h}", metric_value, hint


    def _analyze_resize_ml(self, img_original, channels, w, h, orig_width, orig_height):
        """Analyzes resize using ML prediction with optimized batch processing."""
        # Common features for all channels
        common_features = {
            'scale_factor': (w / orig_width + h / orig_height) / 2,
            'original_width': orig_width,
            'original_height': orig_height,
            'method': self.args.interpolation,
        }

        if self.args.channels:  # Поканальный анализ
            # Extract features for all channels
            features_list = []
            channel_indices = []

            for c in channels:
                channel_idx = channels.index(c)
                channel_indices.append(channel_idx)

                # Extract features for each channel
                channel_features = extract_features_of_original_img(img_original[..., channel_idx])
                channel_features.update(common_features)
                channel_features['channel'] = c
                features_list.append(channel_features)

            # Batch prediction - create one DataFrame for all channels
            if not features_list:
                return f"{w}x{h}", {}, 0.0, ""

            # Create a single DataFrame with all features
            df = pd.DataFrame(features_list)
            # Process and predict in a single batch
            metric_index = list(m.value for m in QualityMetrics).index(self.args.metric.value)

            # Run batch prediction
            try:
                predictions = self.predictor.predict_batch(df)

                channels_metrics = {}
                for i, c in enumerate(channels):
                    val = predictions[i][metric_index]
                    channels_metrics[c] = postprocess_metric_value(val, self.args.metric)

                min_metric = min(channels_metrics.values())
                hint = QualityHelper.get_hint(min_metric, QualityMetrics(self.args.metric))
                return f"{w}x{h}", channels_metrics, min_metric, hint

            except Exception as e:
                logging.error(f"Error in batch prediction: {e}")
                logging.debug("Details:", exc_info=True)

                # Fallback to original method if batch prediction fails
                channels_metrics = {}
                for i, c in enumerate(channels):
                    features = features_list[i]
                    prediction = self.predictor.predict(features)
                    val = prediction.get(self.args.metric.value, 0.0)
                    channels_metrics[c] = postprocess_metric_value(val, self.args.metric)

                min_metric = min(channels_metrics.values()) if channels_metrics else 0.0
                hint = QualityHelper.get_hint(min_metric, QualityMetrics(self.args.metric))
                return f"{w}x{h}", channels_metrics, min_metric, hint
        else:
            features = extract_features_of_original_img(img_original)
            features.update(common_features)
            features['channel'] = 'combined'
            prediction = self.predictor.predict(features)
            metric_value = prediction.get(self.args.metric.value, 0.0)
            metric_value = postprocess_metric_value(metric_value, self.args.metric)
            hint = QualityHelper.get_hint(metric_value, QualityMetrics(self.args.metric))
            return f"{w}x{h}", metric_value, hint


    def _create_original_entry(self, width, height, channels):
        """Creates an entry for the original image."""
        base_entry = (f"{width}x{height}",)
        channel_value = float('inf') if self.args.metric == QualityMetrics.PSNR else float(1.0)
        hint_original = QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.ORIGINAL]
        if self.args.channels and channels:
            return *base_entry, {c: channel_value for c in channels}, channel_value, hint_original
        return *base_entry, channel_value, hint_original


    def _report_results(self, file_path, results, meta):
        """Outputs and saves analysis results."""
        # Вывод в консоль
        ConsoleReporter.print_file_header(file_path, QualityMetrics(self.args.metric))
        if meta['max_val'] < 0.001:
            logging.warning(f"Низкое максимальное значение: {meta['max_val']:.3e}")
        ConsoleReporter.print_quality_table(
            results, self.args.channels, meta.get('channels'),
            QualityMetrics(self.args.metric)
        )

        # Создаём график, если нужно
        if hasattr(self.args, 'chart') and self.args.chart:
            chart_path = self.generate_chart(file_path, results, meta)
            if chart_path:
                try:
                    from rich.console import Console
                    from rich.text import Text
                    Console().print(
                        Text(f"📊 {_("Graph saved")}: ", style="bold green") +
                        Text(f"{chart_path}", style="underline blue")
                    )
                except ImportError:
                    from rich.console import Console
                    from rich.text import Text
                    print(f"📊 {_("Graph saved")}: {chart_path}")

        # Запись в репортеры
        for rep in self.reporters:
            rep.write_results(os.path.basename(file_path), results, self.args.channels)


    @staticmethod
    def _save_intermediate(img_array, file_path, width, height, interpolation, suffix):
        """Saves intermediate result as PNG."""
        file_path_dir = INTERMEDIATE_DIR
        if not os.path.exists(file_path_dir):
            os.makedirs(file_path_dir, exist_ok=True)

        output_filename = (
                os.path.splitext(os.path.basename(file_path))[0] +
                f"_({interpolation.value}_{width}x{height}){'_' + suffix if suffix else ''}.png"
        )
        output_path = os.path.join(file_path_dir, output_filename)

        arr_for_save = img_array
        if arr_for_save.ndim == 3 and arr_for_save.shape[2] == 1:
            arr_for_save = arr_for_save.squeeze(axis=-1)

        arr_uint8 = np.clip(arr_for_save * 255.0, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(arr_uint8)
        pil_img.save(output_path, format="PNG", optimize=True)


    def generate_chart(self, file_path: str, results: list, meta: dict) -> Optional[str]:
        """
        Generates a chart showing quality dependency on resolution.

        Args:
            file_path: Path to the image file
            results: Quality analysis results
            meta: Image metadata

        Returns:
            Path to the created chart or None
        """
        if not self.args.chart:
            return None

        try:
            from ..utils.visualization import generate_quality_chart, get_chart_filename

            file_basename = os.path.basename(file_path)
            chart_path = get_chart_filename(
                os.path.splitext(file_basename)[0],
                QualityMetrics(self.args.metric),
                self.args.channels
            )

            title = f"{_("Quality")} ({self.args.metric.upper()}) {_("depending on the resolution")}\n{file_basename}"
            theme = getattr(self.args, 'theme', 'dark')

            chart_file = generate_quality_chart(
                results,
                chart_path,
                title=title,
                metric_type=QualityMetrics(self.args.metric),
                analyze_channels=self.args.channels,
                channels=meta.get('channels'),
                theme=theme
            )

            logging.debug(f"{_("Graph saved")}: {chart_file}")
            return chart_file
        except Exception as e:
            logging.error(f"{_("Error when generating chart")}: {e}")
            logging.debug("Details:", exc_info=True)
            return None


def process_file_for_analyzer(args_dict, file_path):
    """
    Wrapper function for processing a file in a separate process.

    Args:
        args_dict: Dictionary with arguments for creating ImageAnalyzer
        file_path: Path to the file for analysis
    """
    try:
        # Создаём анализатор из словаря аргументов
        args = argparse.Namespace(**args_dict)
        analyzer = ImageAnalyzer(args)
        return analyzer.analyze_file(file_path)
    except Exception as e:
        logging.error(f"{_("Error processing file")} {file_path}: {e}")
        logging.debug("Details:", exc_info=True)
        return None, None


def postprocess_metric_value(metric_value, metric_type):
    """
    Method for postprocessing metric value.
    """
    # Для словаря (поканальные метрики)
    if isinstance(metric_value, dict):
        if QualityMetrics(metric_type) == QualityMetrics.PSNR:
            return {channel: float('inf') if value >= PSNR_IS_LARGE_AS_INF else value
                    for channel, value in metric_value.items()}
        else:
            # Клемпим значения в диапазон [0;1] для не-PSNR метрик
            return {channel: max(0.0, min(1.0, value))
                    for channel, value in metric_value.items()}

    # Для скалярного значения
    if isinstance(metric_value, (int, float, np.number)):
        if QualityMetrics(metric_type) == QualityMetrics.PSNR:
            if metric_value >= PSNR_IS_LARGE_AS_INF:
                return float('inf')
            return metric_value
        else:
            # Клемпим значения в диапазон [0;1] для не-PSNR метрик
            return max(0.0, min(1.0, metric_value))

    # Неверный тип данных
    raise TypeError(f"Ожидается число или словарь, получено: {type(metric_value).__name__}")