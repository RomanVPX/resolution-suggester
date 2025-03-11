# tests/tests_main.py
import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from resolution_suggester.config import ML_DATASETS_DIR, PSNR_IS_LARGE_AS_INF, QualityMetrics
from resolution_suggester.core.image_analyzer import postprocess_metric_value
from resolution_suggester.main import run_dataset_generation, run_model_subcommand


@pytest.mark.parametrize("value, metric_type, expected", [
    # Test PSNR values above threshold
    (PSNR_IS_LARGE_AS_INF, QualityMetrics.PSNR, float('inf')),
    (PSNR_IS_LARGE_AS_INF + 1, QualityMetrics.PSNR, float('inf')),
    (PSNR_IS_LARGE_AS_INF - 1, QualityMetrics.PSNR, PSNR_IS_LARGE_AS_INF - 1),
    (1000.0, QualityMetrics.PSNR, float('inf')),

    # Test if PSNR values below threshold
    (PSNR_IS_LARGE_AS_INF - 1, QualityMetrics.PSNR, PSNR_IS_LARGE_AS_INF - 1),
    (100.0, QualityMetrics.PSNR, 100.0),
    (0.0, QualityMetrics.PSNR, 0.0),
    (-1.0, QualityMetrics.PSNR, -1.0),

    # Testing other metrics (should clamp to [0;1]])
    (100, QualityMetrics.SSIM, 1.0),
    (0.99, QualityMetrics.MS_SSIM, 0.99),
    (-0.343, QualityMetrics.MS_SSIM, 0.0),
    (1.0, QualityMetrics.MS_SSIM, 1.0),
    (0.0, QualityMetrics.MS_SSIM, 0.0),
    (-0.0, QualityMetrics.MS_SSIM, 0.0),
])
def test_postprocess_metric_value_scalar(value, metric_type, expected):
    """Test postprocess_metric_value with scalar values"""
    result = postprocess_metric_value(value, metric_type)

    if expected == float('inf'):
        assert result == float('inf')
    else:
        assert result == expected


def test_postprocess_metric_value_invalid_type():
    """Test handling of invalid types"""
    with pytest.raises(TypeError):
        postprocess_metric_value([1, 2, 3], QualityMetrics.PSNR)

    with pytest.raises(TypeError):
        postprocess_metric_value(None, QualityMetrics.PSNR)


def test_postprocess_metric_value_dict():
    """Test postprocess_metric_value with dictionary values"""
    # Словарь для PSNR
    metrics_psnr = {'R': 140.0, 'G': 100.0, 'B': PSNR_IS_LARGE_AS_INF, 'A': 50.0}
    result_psnr = postprocess_metric_value(metrics_psnr, QualityMetrics.PSNR)
    assert result_psnr['R'] == float('inf')
    assert result_psnr['G'] == 100.0
    assert result_psnr['B'] == float('inf')
    assert result_psnr['A'] == 50.0

    # Словарь для SSIM
    metrics_ssim = {'R': 0.95, 'G': 0.98, 'B': 0.99}
    result_ssim = postprocess_metric_value(metrics_ssim, QualityMetrics.SSIM)
    assert result_ssim == metrics_ssim  # Должен вернуться без изменений

    # Словарь для SSIM
    metrics_ssim = {'R': 1.001, 'G': 0.98, 'B': 0.99}
    result_ssim = postprocess_metric_value(metrics_ssim, QualityMetrics.SSIM)
    assert result_ssim == {'R': 1.00, 'G': 0.98, 'B': 0.99}  # Должен клемпнуть

def test_postprocess_metric_value_nan():
    """Test handling of NaN values"""
    result = postprocess_metric_value(float('nan'), QualityMetrics.PSNR)
    assert np.isnan(result)

    # Словарь с NaN
    metrics_with_nan = {'R': float('nan'), 'G': 100.0}
    result_dict = postprocess_metric_value(metrics_with_nan, QualityMetrics.PSNR)
    assert np.isnan(result_dict['R'])
    assert result_dict['G'] == 100.0


@patch('resolution_suggester.main.generate_dataset')
@patch('resolution_suggester.main.logging')
def test_run_dataset_generation(mock_logging, mock_generate_dataset):
    """Test dataset generation function"""
    # Подготовка мок-объектов
    mock_generate_dataset.return_value = ('features.csv', 'targets.csv')
    
    # Создаём тестовые аргументы
    args = argparse.Namespace()
    files = ['/path/to/test.png']
    
    # Вызов тестируемой функции
    result = run_dataset_generation(files, args)
    
    # Проверки
    mock_generate_dataset.assert_called_once_with(files, args)
    mock_logging.info.assert_called_once()
    assert result == ('features.csv', 'targets.csv')


@patch('resolution_suggester.main.run_dataset_generation')
@patch('resolution_suggester.main.QuickPredictor')
@patch('resolution_suggester.main.logging')
@patch('pathlib.Path.exists')
def test_run_model_subcommand_generate_and_train(mock_path_exists, mock_logging, mock_predictor, mock_run_dataset):
    """Test model subcommand with dataset generation and training"""
    # Подготовка мок-объектов
    mock_run_dataset.return_value = ('features.csv', 'targets.csv')
    mock_predictor_instance = MagicMock()
    mock_predictor.return_value = mock_predictor_instance
    
    # Создаём тестовые аргументы
    args = argparse.Namespace()
    args.generate_dataset = True
    args.train_ml = True
    files = ['/path/to/test.png']
    
    # Вызов тестируемой функции
    run_model_subcommand(files, args)
    
    # Проверки
    mock_run_dataset.assert_called_once_with(files, args)
    mock_predictor.assert_called_once()
    mock_predictor_instance.train.assert_called_once_with('features.csv', 'targets.csv')
    mock_logging.info.assert_called()


@patch('resolution_suggester.main.QuickPredictor')
@patch('resolution_suggester.main.logging')
@patch('pathlib.Path.exists')
def test_run_model_subcommand_train_only(mock_path_exists, mock_logging, mock_predictor):
    """Test model subcommand with training only (existing dataset)"""
    # Подготовка мок-объектов
    mock_path_exists.return_value = True
    mock_predictor_instance = MagicMock()
    mock_predictor.return_value = mock_predictor_instance
    
    # Создаём тестовые аргументы
    args = argparse.Namespace()
    args.generate_dataset = False
    args.train_ml = True
    files = ['/path/to/test.png']
    
    # Вызов тестируемой функции
    run_model_subcommand(files, args)
    
    # Проверки
    mock_predictor.assert_called_once()
    mock_predictor_instance.train.assert_called_once()
    mock_logging.info.assert_called()


@patch('resolution_suggester.main.QuickPredictor')
@patch('resolution_suggester.main.logging')
@patch('pathlib.Path.exists')
def test_run_model_subcommand_train_only_no_dataset(mock_path_exists, mock_logging, mock_predictor):
    """Test model subcommand with training only but missing dataset"""
    # Подготовка мок-объектов
    mock_path_exists.return_value = False
    
    # Создаём тестовые аргументы
    args = argparse.Namespace()
    args.generate_dataset = False
    args.train_ml = True
    files = ['/path/to/test.png']
    
    # Вызов тестируемой функции
    run_model_subcommand(files, args)
    
    # Проверки
    mock_predictor.assert_not_called()
    mock_logging.error.assert_called_once()


@patch('resolution_suggester.main.logging')
def test_run_model_subcommand_no_operation(mock_logging):
    """Test model subcommand with no operation specified"""
    # Создаём тестовые аргументы
    args = argparse.Namespace()
    args.generate_dataset = False
    args.train_ml = False
    files = ['/path/to/test.png']
    
    # Вызов тестируемой функции
    run_model_subcommand(files, args)
    
    # Проверки
    mock_logging.info.assert_called_once()
