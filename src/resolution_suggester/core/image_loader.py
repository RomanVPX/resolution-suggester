# core/image_loader.py
import logging
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pyexr
from PIL import Image, ImageFile, UnidentifiedImageError

from ..config import TINY_EPSILON
from ..i18n import _

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
    data: np.ndarray | None
    max_value: float | None
    channels: list[str] | None
    error: str | None = None


def load_image(file_path: str, normalize_exr: bool = False) -> ImageLoadResult:
    """
    Loads an image from a file and returns a numpy array, max value, and channels.
    Supported formats: EXR, PNG, TGA, JPG, JPEG.

    Args:
        file_path: Path to the image file.
        normalize_exr: Whether to normalize EXR images to [0, 1].

    Returns:
        ImageLoadResult: fields (data, max_value, channels, error)
    """
    # Existence check
    if not os.path.exists(file_path):
        error_msg = f"{_('File not found')}: {file_path}"
        logging.error(error_msg)
        return ImageLoadResult(None, None, None, error_msg)

    ext = os.path.splitext(file_path)[1].lower()

    # Size check
    file_size = os.path.getsize(file_path)
    if file_size < 16:
        error_msg = _("File may be corrupted or is not an image: {}") + f" {file_path} " + _("(file size: {} bytes)").format(file_size)
        logging.error(error_msg)
        return ImageLoadResult(None, None, None, error_msg)

    try:
        if ext == '.exr':
            return load_exr(file_path, normalize_exr)
        elif ext in {'.png', '.tga', '.jpg', '.jpeg'}:
            return load_raster(file_path)
        else:
            error_msg = _("Unsupported file format: {}").format(file_path)
            logging.warning(error_msg)
            return ImageLoadResult(None, None, None, error_msg)

    except MemoryError:
        error_msg = f"{_('Not enough memory to load image')}: {file_path}"
        logging.error(error_msg)
        return ImageLoadResult(None, None, None, error_msg)
    except Exception as e:
        error_msg = f"{_('Error reading')} {file_path}, {str(e)}"
        logging.error(error_msg)
        return ImageLoadResult(None, None, None, str(e))


def load_exr(file_path: str, normalize_exr: bool) -> ImageLoadResult | None:
    """Loads an EXR file with channel processing (optionally normalizing to [0, 1])."""
    try:
        exr_file = pyexr.open(file_path)
        try:
            channels = exr_file.channels if hasattr(exr_file, 'channels') else []
            img = exr_file.get().astype(np.float32)

            if not channels:
                num_channels = img.shape[2] if img.ndim > 2 else 1
                channels = ['R', 'G', 'B', 'A'][:num_channels] if num_channels > 1 else ['L']

            if normalize_exr:
                min_val = np.min(img)
                max_val = np.max(img)
                range_val = max_val - min_val
                if range_val > TINY_EPSILON:
                    img = (img - min_val) / range_val
                    max_val = 1.0
                    logging.debug(f"EXR normalized to [0, 1], min={min_val:.3f}, max={max_val:.3f} from {file_path}")
                else:
                    # Избежать деления на ноль в вырожденном случае
                    max_val = 1.0
                    logging.debug(f"EXR normalization skipped (small range), max={max_val:.3f} from {file_path}")
            else:
                max_val = np.max(np.abs(img))

            return ImageLoadResult(img, float(max_val), channels)
        finally:
            exr_file.close()  # Ensure EXR file is closed
    except Exception as e:
        # Не логируем ошибку здесь, она будет обработана в load_image
        return ImageLoadResult(None, None, None, f"{_('Error processing EXR')}: {str(e)}")


def load_raster(image_path: str) -> ImageLoadResult:
    """
    Loads PNG/TGA/JPG images and normalizes data to range [0, 1].
    If the image is grayscale, expands to (H, W, 1) for consistency.
    """
    img = None
    try:
        img = Image.open(image_path)
        if img.mode not in MODE_CHANNEL_MAP:
            img = img.convert('RGB') # Конвертация в RGB для неподдерживаемых режимов

        mode = img.mode  # фиксируем режим после возможного преобразования
        divisor = BIT_DEPTH_16 if img.mode.startswith('I;16') else BIT_DEPTH_8
        img_array = np.array(img).astype(np.float32) / divisor

        # Проверяем, что изображение имеет как минимум 3 измерения
        if img_array.ndim == 2:
            img_array = img_array[:, :, np.newaxis]
        channels = MODE_CHANNEL_MAP.get(mode, ['R', 'G', 'B'])

        # Максимальное значение после нормализации всегда равно 1,0
        return ImageLoadResult(img_array, 1.0, channels)

    except FileNotFoundError:
        return ImageLoadResult(None, None, None, f"{_('File not found')}")
    except UnidentifiedImageError:
        return ImageLoadResult(None, None, None, f"{_('Unable to decode image')}")
    except Exception as e:
        return ImageLoadResult(None, None, None, str(e))
    finally:
        if img is not None and hasattr(img, 'close'):
            img.close()