# core/metrics.py
import math

from ..i18n import _
import numpy as np
import torch
from numba import njit, prange
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.feature import canny
from skimage.filters import sobel
from skimage.morphology import dilation, footprint_rectangle
from ..config import MIN_DOWNSCALE_SIZE, TINY_EPSILON, QualityMetrics

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _get_adaptive_ms_ssim_params(h: int, w: int) -> tuple[tuple[float, ...], int]:
    """Динамически подбирает параметры MS-SSIM под размер изображения"""
    min_dim = min(h, w)

    if min_dim >= 160:
        return (0.0448, 0.2856, 0.3001, 0.2363, 0.1333), 11  # оригинальные параметры (5 scales, kernel 11)
    elif min_dim >= 80:
        return (0.25, 0.25, 0.25, 0.25), 7   # 4 scales, kernel 7
    elif min_dim >= 40:
        return (0.333, 0.333, 0.334), 5   # 3 scales, kernel 5
    else:
        return (0.5, 0.5), 3   # минимальные параметры для мелочи


def calculate_ms_ssim_pytorch(
        original: np.ndarray,
        processed: np.ndarray,
        max_val: float,
        no_gpu: bool = False
) -> float:
    """
    Calculates MS-SSIM between two images using PyTorch.
    """
    # Инициализируем вычислитель для каждого вызова для очистки памяти
    # Нормализация изображения
    max_val = max(max_val, TINY_EPSILON)
    if max_val > 1.0 + TINY_EPSILON:
        original = original.astype(np.float32) / max_val
        processed = processed.astype(np.float32) / max_val
    else:
        original = original.astype(np.float32)
        processed = processed.astype(np.float32)

    # Добавляем размерность батча и каналов
    if original.ndim == 2:
        original = original[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        processed = processed[np.newaxis, np.newaxis, :, :]
    elif original.ndim == 3:
        original = original.transpose(2, 0, 1)  # (C, H, W)
        processed = processed.transpose(2, 0, 1)
        original = original[np.newaxis, :, :, :]  # (1, C, H, W)
        processed = processed[np.newaxis, :, :, :]
    else:
        raise ValueError(_("MS-SSIM: unsupported images dimensions"))

    torch_device = device if not no_gpu else torch.device("cpu")

    weights, kernel_size = _get_adaptive_ms_ssim_params(original.shape[-2], original.shape[-1])
    msssim_calc = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, betas=weights,
                                                             kernel_size=kernel_size).to(torch_device)
    # Переводим в тензоры и переносим на устройство
    original_tensor = torch.from_numpy(original).to(torch_device)
    processed_tensor = torch.from_numpy(processed).to(torch_device)

    # Вычисление MS-SSIM
    with torch.no_grad():
        with torch.autocast(device_type=torch_device.type, dtype=torch.float16):
            ms_ssim_val = msssim_calc(original_tensor, processed_tensor).item()
    # Явное освобождение ресурсов
    del original_tensor, processed_tensor
    # Очистка кэша для MPS, если функция доступна:
    if torch_device.type == 'mps':
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        else:
            # Иначе просто пропускаем, чтобы не вызывать ошибку
            pass
    elif torch_device.type == 'cuda':
        # Для CUDA вызываем очистку без проверок
        torch.cuda.empty_cache()

    return float(ms_ssim_val)


def calculate_ms_ssim_pytorch_channels(
        original: np.ndarray,
        processed: np.ndarray,
        max_val: float,
        channels: list[str],
        no_gpu: bool = False
) -> dict[str, float]:
    """
    Calculates MS-SSIM for each channel separately.
    """
    results = {}
    for i, ch in enumerate(channels):
        orig_ch = original[..., i] if original.ndim == 3 else original
        proc_ch = processed[..., i] if processed.ndim == 3 else processed
        results[ch] = calculate_ms_ssim_pytorch(orig_ch, proc_ch, max_val, no_gpu)
    return results


@njit(cache=True)
def calculate_psnr(
        original: np.ndarray,
        processed: np.ndarray,
        max_val: float
) -> float:

    diff = original - processed
    mse = np.mean(diff * diff)

    log_max = 20 * np.log10(max_val)
    return log_max - 10 * np.log10(mse)


def calculate_psnr_channels(
        original: np.ndarray,
        processed: np.ndarray,
        max_val: float,
        channels: list[str]
) -> dict[str, float]:
    return {
        channel: calculate_psnr(original[..., i], processed[..., i], max_val)
        for i, channel in enumerate(channels)
    }


def calculate_tdpr(
        original: np.ndarray,
        processed: np.ndarray,
        edge_sigma: float = 1.0,
        dilation_size: int = 1
) -> float:
    """
    Calculates the texture detail preservation coefficient (TDPR).
    """
    # Обнаруживаем края на оригинальном изображении
    original_edges = detect_texture_edges(original, sigma=edge_sigma)

    # Расширяем маску краев для учета небольших смещений
    if dilation_size > 0:
        # Создаем квадратный footprint размером dilation_size x dilation_size
        footprint = footprint_rectangle((dilation_size, dilation_size))
        original_edges_dilated = dilation(original_edges, footprint)
    else:
        original_edges_dilated = original_edges

    # Обнаруживаем края на обработанном изображении
    processed_edges = detect_texture_edges(processed, sigma=edge_sigma)

    # Считаем количество краев в оригинальном и обработанном изображениях
    original_edge_count = np.sum(original_edges)

    # Если в оригинальном изображении нет краев, возвращаем 1.0 (идеальное сохранение)
    if original_edge_count == 0:
        return 1.0

    # Находим пересечение краев (сохраненные детали)
    preserved_edges = np.logical_and(original_edges_dilated, processed_edges)
    preserved_edge_count = np.sum(preserved_edges)

    tdpr = preserved_edge_count / original_edge_count

    return float(tdpr)


def calculate_tdpr_channels(
        original: np.ndarray,
        processed: np.ndarray,
        channels: list[str],
        edge_sigma: float = 1.0
) -> dict[str, float]:
    """
    Calculates TDPR for each channel separately.
    """
    results = {}

    # Если изображение одноканальное
    if original.ndim == 2:
        results['L'] = calculate_tdpr(
            original, processed, edge_sigma
        )
        return results

    # Для многоканальных изображений
    for i, ch in enumerate(channels):
        orig_ch = original[..., i]
        proc_ch = processed[..., i]

        results[ch] = calculate_tdpr(
            orig_ch, proc_ch, edge_sigma
        )

    return results


def detect_texture_edges(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Detects edges in the image that represent textural details.
    """
    # Нормализуем изображение, если оно не в диапазоне [0, 1]
    if image.max() > 1.0 + TINY_EPSILON:
        normalized = image / image.max()
    else:
        normalized = image.copy()

    # Если изображение многоканальное, конвертируем в оттенки серого
    if normalized.ndim == 3:
        gray = np.mean(normalized, axis=2)
    else:
        gray = normalized

    # Вычисляем порог на основе статистики изображения
    img_std = np.std(gray)
    img_mean = np.mean(gray)

    # Проверяем, имеет ли изображение достаточную вариацию для Canny
    if img_std < 0.01:
        sobel_edges = sobel(gray)
        sobel_edges_np = np.asarray(sobel_edges)
        max_val = np.max(sobel_edges_np)
        if max_val < TINY_EPSILON:
            max_val = TINY_EPSILON
        threshold = 0.05 * max_val
        return sobel_edges_np > threshold

    # Устанавливаем безопасные пороги, гарантируя, что low < high
    low_threshold = max(0.01, img_mean * 0.5)
    high_threshold = max(low_threshold + 0.01, img_mean * 1.0)

    try:
        edges = canny(gray, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        return edges
    except ValueError:
        # В случае ошибки с порогами используем метод Собеля как запасной вариант
        sobel_edges = sobel(gray)
        sobel_edges_np = np.asarray(sobel_edges)
        max_val = np.max(sobel_edges_np)
        if max_val < TINY_EPSILON:
            max_val = TINY_EPSILON
        threshold = 0.05 * max_val
        return sobel_edges_np > threshold


def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Формирует 1D-ядро Гаусса длиной size с параметром sigma.
    """
    kernel_1d = np.empty(size, dtype=np.float32)
    half = (size - 1) / 2.0

    sum_val = 0.0
    for i in range(size):
        x = i - half
        val = math.exp(-0.5 * (x*x) / (sigma*sigma))
        kernel_1d[i] = val
        sum_val += val

    # Нормируем, чтобы сумма коэффициентов была = 1
    for i in range(size):
        kernel_1d[i] /= sum_val

    return kernel_1d


@njit(cache=True)
def reflect_coordinate(x: int, size: int) -> int:
    """
    Reflective remapping of index x within [0..size-1]
    (similar to OpenCV BORDER_REFLECT_101).
    """
    if x < 0:
        return -x - 1
    if x >= size:
        return 2*size - x - 1
    return x


@njit(parallel=True, cache=True)
def convolve_1d_horiz(src: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """
    Horizontal 1D convolution of the image src (H,W).
    Reflective padding horizontally.
    """
    h, w = src.shape
    ksize = kernel_1d.size
    pad = ksize // 2

    out = np.zeros_like(src)
    for i in prange(h):
        for j in prange(w):
            accum = 0.0
            for k in prange(ksize):
                jj = j + k - pad
                jj = reflect_coordinate(jj, w)
                accum += src[i, jj] * kernel_1d[k]
            out[i, j] = accum
    return out


@njit(parallel=True, cache=True)
def convolve_1d_vert(src: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """
    Horizontal 1D convolution of the image src (H,W).
    Reflective padding vertically.
    """
    h, w = src.shape
    ksize = kernel_1d.size
    pad = ksize // 2

    out = np.zeros_like(src)
    for j in prange(w):
        for i in prange(h):
            accum = 0.0
            for k in prange(ksize):
                ii = i + k - pad
                ii = reflect_coordinate(ii, h)
                accum += src[ii, j] * kernel_1d[k]
            out[i, j] = accum
    return out


def filter_2d_separable(img: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """
    Separable Gaussian Convolution for (H,W) or (H,W,C).
    1) Generate a 1D kernel kernel_1d.
    2) Horizontal convolution.
    3) Vertical convolution.
    """
    # Создаём 1D ядро Гаусса
    kernel_1d = gaussian_kernel_1d(size, sigma)

    if img.ndim == 2:
        # (H,W) — одиночный канал
        tmp = convolve_1d_horiz(img, kernel_1d)
        out = convolve_1d_vert(tmp, kernel_1d)
        return out

    # (H,W,C) — многоканальное изображение
    h, w, c = img.shape
    result = np.zeros_like(img)
    for ch in range(c):
        tmp = convolve_1d_horiz(img[..., ch], kernel_1d)
        out = convolve_1d_vert(tmp, kernel_1d)
        result[..., ch] = out
    return result


def _calculate_ssim_gauss_single(
        original: np.ndarray,
        processed: np.ndarray,
        k_1: float = 0.01,
        k_2: float = 0.03,
        l: float  = 1.0,
        window_size: int = 11,
        sigma: float = 1.5
) -> float:

    # Фильтруем средние
    mu_x = filter_2d_separable(original, window_size, sigma)
    mu_y = filter_2d_separable(processed, window_size, sigma)

    # Фильтруем X^2, Y^2 и X*Y
    sigma_x = filter_2d_separable(original*original, window_size, sigma) - mu_x*mu_x
    sigma_y = filter_2d_separable(processed*processed, window_size, sigma) - mu_y*mu_y
    sigma_xy = filter_2d_separable(original*processed, window_size, sigma) - mu_x*mu_y

    c1 = (k_1 * l) ** 2
    c2 = (k_2 * l) ** 2

    numerator   = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

    ssim_map = numerator / np.maximum(denominator, TINY_EPSILON)
    return float(np.mean(ssim_map))


def calculate_ssim_gauss(
        original: np.ndarray,
        processed: np.ndarray,
        max_val: float,
        k_1: float = 0.01,
        k_2: float = 0.03,
        window_size: int = 11,
        sigma: float = 1.5
) -> float:
    # Нормализуем, если max_val > 1
    if max_val > 1.0 + TINY_EPSILON:
        original  = original  / max_val
        processed = processed / max_val

    if original.ndim == 2:
        return _calculate_ssim_gauss_single(original, processed, k_1, k_2, 1.0, window_size, sigma)
    if original.ndim == 3:
        c = original.shape[2]
        ssim_sum = 0.0
        for i in range(c):
            ssim_sum += _calculate_ssim_gauss_single(
                original[..., i],
                processed[..., i],
                k_1, k_2, 1.0, window_size, sigma
            )
        return ssim_sum / c

    raise ValueError(_("SSIM: unsupported images dimensions"))


def calculate_ssim_gauss_channels(
        original: np.ndarray,
        processed: np.ndarray,
        max_val: float,
        channels: list[str],
        window_size: int = 11,
        sigma: float = 1.5
) -> dict[str, float]:
    # Нормализуем, если max_val > 1
    if max_val > 1.0 + TINY_EPSILON:
        original  = original  / max_val
        processed = processed / max_val

    # Если (H,W), то один канал:
    if original.ndim == 2:
        return {
            'L': _calculate_ssim_gauss_single(
                original, processed,
                window_size=window_size, sigma=sigma
            )
        }

    # (H,W,C)
    ssim_dict: dict[str, float] = {}
    for i, ch in enumerate(channels):
        ssim_val = _calculate_ssim_gauss_single(
            original[..., i],
            processed[..., i],
            window_size=window_size, sigma=sigma
        )
        ssim_dict[ch] = ssim_val

    return ssim_dict


def calculate_metrics(
        quality_metric: QualityMetrics,
        original: np.ndarray,
        processed: np.ndarray,
        max_val: float,
        channels: list[str] = None,
        no_gpu: bool = False
) -> None | dict[str, float] | float:

    if original.shape != processed.shape:
        raise ValueError(_("calculate_metrics: The shapes of the images must match!"))
    if original.ndim != processed.ndim:
        raise ValueError(_("calculate_metrics: The dimensions of the images must match!"))

    if channels is None:
        match quality_metric:
            case QualityMetrics.PSNR:
                return calculate_psnr(original, processed, max_val)
            case QualityMetrics.SSIM:
                return calculate_ssim_gauss(original, processed, max_val)
            case QualityMetrics.MS_SSIM:
                return calculate_ms_ssim_pytorch(original, processed, max_val, no_gpu=no_gpu)
            case QualityMetrics.TDPR:
                return calculate_tdpr(original, processed)
    else:
        match quality_metric:
            case QualityMetrics.PSNR:
                return calculate_psnr_channels(original, processed, max_val, channels)
            case QualityMetrics.SSIM:
                return calculate_ssim_gauss_channels(original, processed, max_val, channels)
            case QualityMetrics.MS_SSIM:
                return calculate_ms_ssim_pytorch_channels(original, processed, max_val, channels, no_gpu=no_gpu)
            case QualityMetrics.TDPR:
                return calculate_tdpr_channels(original, processed, channels)

    raise ValueError(f"{_('Unsupported quality metric')}: {quality_metric}")


def compute_resolutions(
        original_width: int,
        original_height: int,
        min_size: int = MIN_DOWNSCALE_SIZE,
        divider: int = 2
) -> list[tuple[int, int]]:
    min_size = max(MIN_DOWNSCALE_SIZE, min_size)
    resolutions = []
    w, h = original_width, original_height

    if w < min_size or h < min_size:
        return resolutions

    while w >= min_size and h >= min_size:
        resolutions.append((w, h))
        w //= divider
        h //= divider

    return resolutions
