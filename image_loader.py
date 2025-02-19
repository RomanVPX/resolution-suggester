# image_loader.py
import logging
import numpy as np
import pyexr
import os
from PIL import Image, ImageFile, UnidentifiedImageError
from typing import Dict, Optional
from dataclasses import dataclass

ImageFile.LOAD_TRUNCATED_IMAGES = True

BIT_DEPTH_16 = 65535.0
BIT_DEPTH_8 = 255.0

MODE_CHANNEL_MAP: Dict[str, list[str]] = {
    'L': ['L'],
    'LA': ['L', 'A'],
    'RGB': ['R', 'G', 'B'],
    'RGBA': ['R', 'G', 'B', 'A'],
    'I;16': ['L'],
    'I;16L': ['L'],
    'I;16B': ['L'],
    # 'CMYK': ['C', 'M', 'Y', 'K'],
    # 'YCbCr': ['Y', 'Cb', 'Cr'],
    # 'LAB': ['L', 'a', 'b']
}

@dataclass
class ImageLoadResult:
    data: Optional[np.ndarray]
    max_value: Optional[float]
    channels: Optional[list[str]]
    error: Optional[str] = None

def load_image(file_path: str, normalize_exr: bool = False) -> ImageLoadResult:
    """
    Загружает изображение из файла и возвращает массив numpy, максимальное значение и каналы.
    Поддерживаемые форматы: EXR, PNG, TGA.

    Args:
        file_path: Путь к файлу изображения
        normalize_exr: Приводить ли EXR к диапазону [0..1] (опционально)

    Returns:
        ImageLoadResult: поля (data, max_value, channels, error)
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()

        # Existence check
        if not os.path.exists(file_path):
            return ImageLoadResult(None, None, None, f"Файл не найден: {file_path}")

        # Size check
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return ImageLoadResult(None, None, None, f"Пустой файл: {file_path}")

        match ext:
            case '.exr':
                return load_exr(file_path, normalize_exr)
            case '.png' | '.tga' | '.jpg' | '.jpeg':
                return load_raster(file_path)
            case _:
                msg = f"Неподдерживаемый формат файла: {file_path}"
                logging.warning(msg)
                return ImageLoadResult(None, None, None, msg)

    except MemoryError:
        logging.error("Недостаточно памяти для загрузки изображения %s", file_path)
        return ImageLoadResult(None, None, None, "Недостаточно памяти для загрузки изображения")
    except Exception as e:
        logging.error("Ошибка при чтении %s: %s", file_path, str(e))
        return ImageLoadResult(None, None, None, str(e))

def load_exr(file_path: str, normalize_exr: bool) -> ImageLoadResult:
    """Загружает EXR файл с обработкой каналов (опционально нормализуя к [0, 1])."""
    try:
        exr_file = pyexr.open(file_path)
        try:
            channels = exr_file.channels if hasattr(exr_file, 'channels') else []
            img = exr_file.get().astype(np.float32)

            if not channels:
                # Определяем число каналов, если не удалось вычитать явно
                num_channels = img.shape[2] if img.ndim > 2 else 1
                if num_channels > 1:
                    channels = ['R', 'G', 'B', 'A'][:num_channels]
                else:
                    channels = ['L']

            # Если задано normalize_exr=True, приводим к диапазону [0..1]
            if normalize_exr:
                min_val = np.min(img)
                max_val = np.max(img)
                range_val = max_val - min_val
                if range_val > 1e-7:
                    img = (img - min_val) / range_val
                    max_val = 1.0
                    logging.debug("EXR нормализован в [0, 1], min=%.3f, max=%.3f у %s" % (min_val, max_val, file_path))
                else:
                    # Избежать деления на ноль в вырожденном случае
                    max_val = 1.0
                    logging.debug("Нормализация EXR пропущена (малый диапазон), max={:.3f} у {}".format(max_val, file_path))
            else:
                max_val = np.max(np.abs(img))

            return ImageLoadResult(img, float(max_val), channels)
        finally:
            exr_file.close() # Ensure EXR file is closed
    except Exception as e:
        logging.error("Ошибка обработки EXR %s: %s", file_path, str(e))
        return ImageLoadResult(None, None, None, str(e))

def load_raster(file_path: str) -> ImageLoadResult:
    """
    Загружает PNG/TGA и нормализует данные в диапазон [0..1].
    Если изображение grayscale, то расширяем до (H, W, 1) для согласованности.
    """
    try:
        # Используем контекстный менеджер, чтобы корректно закрыть файл
        with Image.open(file_path) as img:
            if img.mode not in MODE_CHANNEL_MAP:
                img = img.convert('RGB')

            divisor = BIT_DEPTH_16 if img.mode.startswith('I;16') else BIT_DEPTH_8
            img_array = np.array(img).astype(np.float32) / divisor

            if img_array.ndim == 2:
                img_array = np.expand_dims(img_array, axis=-1)

            channels = MODE_CHANNEL_MAP[img.mode]
            # Максимальное значение после нормализации всегда 1.0 для PNG/TGA
            return ImageLoadResult(img_array, 1.0, channels)
    except FileNotFoundError:
        logging.error("Файл не найден: %s", file_path)
        return ImageLoadResult(None, None, None, f"Файл не найден: {file_path}")
    except UnidentifiedImageError:
        logging.error("Невозможно декодировать изображение: %s", file_path)
        return ImageLoadResult(None, None, None, f"Невозможно декодировать изображение: {file_path}")
    except Exception as e:
        logging.error("Ошибка обработки растрового изображения %s: %s", file_path, e)
        return ImageLoadResult(None, None, None, str(e))
