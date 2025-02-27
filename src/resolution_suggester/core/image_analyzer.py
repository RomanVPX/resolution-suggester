# core/image_analyzer.py
import os
import argparse
import concurrent.futures
from typing import Tuple

import numpy as np

# from ..i18n import _
from PIL import Image
import logging
from typing import Optional

from resolution_suggester.config import INTERPOLATION_METHOD_UPSCALE, InterpolationMethods, QualityMetrics, \
    PSNR_IS_LARGE_AS_INF, INTERMEDIATE_DIR
from resolution_suggester.core.image_processing import get_resize_function
from resolution_suggester.ml.predictor import QuickPredictor, extract_features_of_original_img
from resolution_suggester.utils.reporters import IReporter
from .image_loader import load_image
from .metrics import compute_resolutions, calculate_metrics
from ..utils.reporting import QualityHelper, ConsoleReporter


class ImageAnalyzer:
    """
    Класс, инкапсулирующий логику анализа изображений.
    Предоставляет методы для анализа отдельных файлов и групп файлов.
    """

    def __init__(self, args: argparse.Namespace, reporters: list[IReporter] = None):
        """
        Инициализирует анализатор изображений.

        Args:
            args: Аргументы командной строки
            reporters: Список объектов для отчетности (CSV, JSON и т.д.)
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
        """Инициализирует и настраивает предиктор для ML-предсказаний."""
        predictor = QuickPredictor()
        predictor.set_mode(self.args.channels)
        if not predictor.load():
            logging.info("ML-модель не найдена, будем вычислять реальные метрики.")
            return None
        return predictor

    def analyze_files(self, files: list[str]) -> None:
        """
        Анализирует список файлов.

        Args:
            files: Список путей к файлам для анализа
        """
        if self.args.no_parallel:
            self._analyze_files_sequential(files)
        else:
            self._analyze_files_parallel(files)

    def _analyze_files_sequential(self, files: list[str]) -> None:
        """Последовательный анализ файлов."""
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
        """Параллельный анализ файлов с использованием ProcessPoolExecutor."""
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
        Анализирует одно изображение.

        Args:
            file_path: Путь к файлу изображения

        Returns:
            Кортеж (результаты, метаданные) или (None, None) в случае ошибки
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
        """Анализирует изменение размера с реальным вычислением метрик."""
        # Уменьшаем изображение
        img_downscaled = self.resize_fn(img_original, w, h)

        # Сохраняем промежуточный результат, если нужно
        if self.args.save_im_down:
            self._save_intermediate(img_downscaled, file_path, w, h,
                                    InterpolationMethods(self.args.interpolation), '')

        # Увеличиваем обратно для сравнения с оригиналом
        img_upscaled = self.resize_fn_upscale(img_downscaled, orig_width, orig_height)

        # Сохраняем промежуточный результат увеличения, если нужно
        if self.args.save_im_up:
            self._save_intermediate(img_upscaled, file_path, w, h,
                                    InterpolationMethods(self.args.interpolation), 'upscaled')

        # Вычисляем метрики в зависимости от режима (с каналами или без)
        if self.args.channels:
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
        """Анализирует изменение размера с использованием ML-предсказания."""
        if self.args.channels:
            # Поканальный анализ
            channels_metrics = {}
            for c in channels:
                features = extract_features_of_original_img(img_original[..., channels.index(c)])
                features.update({
                    'scale_factor': (w / orig_width + h / orig_height) / 2,
                    'original_width': orig_width,
                    'original_height': orig_height,
                    'channel': c,
                    'method': self.args.interpolation,
                })
                prediction = self.predictor.predict(features)
                val = prediction.get(self.args.metric.value, 0.0)
                channels_metrics[c] = float('inf') if val >= PSNR_IS_LARGE_AS_INF else val

            min_metric = min(channels_metrics.values())
            hint = QualityHelper.get_hint(min_metric, QualityMetrics(self.args.metric))
            return f"{w}x{h}", channels_metrics, min_metric, hint
        else:
            # Общий анализ
            features = extract_features_of_original_img(img_original)
            features.update({
                'scale_factor': (w / orig_width + h / orig_height) / 2,
                'original_width': orig_width,
                'original_height': orig_height,
                'method': self.args.interpolation,
                'channel': 'combined'
            })
            prediction = self.predictor.predict(features)
            metric_value = prediction.get(self.args.metric.value, 0.0)
            metric_value = float('inf') if metric_value >= PSNR_IS_LARGE_AS_INF else metric_value
            hint = QualityHelper.get_hint(metric_value, QualityMetrics(self.args.metric))
            return f"{w}x{h}", metric_value, hint

    def _create_original_entry(self, width, height, channels):
        """Создает запись для оригинального изображения."""
        base_entry = (f"{width}x{height}",)
        if self.args.channels and channels:
            return *base_entry, {c: float('inf') for c in channels}, float('inf'), "Оригинал"
        return *base_entry, float('inf'), "Оригинал"

    def _report_results(self, file_path, results, meta):
        """Выводит и сохраняет результаты анализа."""
        # Вывод в консоль
        ConsoleReporter.print_file_header(file_path, QualityMetrics(self.args.metric))
        if meta['max_val'] < 0.001:
            logging.warning(f"Низкое максимальное значение: {meta['max_val']:.3e}")
        ConsoleReporter.print_quality_table(
            results, self.args.channels, meta.get('channels'),
            QualityMetrics(self.args.metric)
        )

        # Запись в репортеры
        for rep in self.reporters:
            rep.write_results(os.path.basename(file_path), results, self.args.channels)

    @staticmethod
    def _save_intermediate(img_array, file_path, width, height, interpolation, suffix):
        """Сохраняет промежуточный результат как PNG."""
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


def process_file_for_analyzer(args_dict, file_path):
    """
    Функция-обертка для обработки файла в отдельном процессе.

    Args:
        args_dict: Словарь с аргументами для создания ImageAnalyzer
        file_path: Путь к файлу для анализа
    """
    try:
        # Создаем анализатор из словаря аргументов
        args = argparse.Namespace(**args_dict)
        analyzer = ImageAnalyzer(args)
        return analyzer.analyze_file(file_path)
    except Exception as e:
        logging.error(f"Ошибка обработки файла {file_path}: {e}")
        logging.debug("Детали:", exc_info=True)
        return None, None


def postprocess_metric_value(metric_value, metric_type):
    """
    Method for postprocessing metric value.
    """
    # Обрабатываем только PSNR
    if QualityMetrics(metric_type) != QualityMetrics.PSNR:
        return metric_value

    # Для словаря (поканальные метрики)
    if isinstance(metric_value, dict):
        return {channel: float('inf') if value >= PSNR_IS_LARGE_AS_INF else value
                for channel, value in metric_value.items()}

    # Для скалярного значения
    if isinstance(metric_value, (int, float, np.number)):
        return float('inf') if metric_value >= PSNR_IS_LARGE_AS_INF else metric_value

    # Неверный тип данных
    raise TypeError(f"Ожидается число или словарь, получено: {type(metric_value).__name__}")