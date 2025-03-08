# tests/utils/test_cli.py
import argparse
import pytest
import sys
import logging
from unittest.mock import patch, MagicMock

from resolution_suggester.utils.cli import (
    validate_paths, 
    handle_backward_compatibility,
    parse_arguments,
    create_parser
)


def test_validate_paths(tmp_path):
    # Создаём тестовые файлы и директории
    valid_dir = tmp_path / "valid_dir"
    valid_dir.mkdir()

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # Создаём PNG файл в директории (поддерживаемое расширение)
    valid_file_in_dir = valid_dir / "test_in_dir.png"
    valid_file_in_dir.write_text("dummy content")

    # Создаём TXT файл в директории (неподдерживаемое расширение)
    invalid_file_in_dir = valid_dir / "test_in_dir.txt"
    invalid_file_in_dir.write_text("dummy content")

    # Создаём PNG файл вне директории
    valid_file = tmp_path / "test.png"
    valid_file.write_text("dummy content")

    # Создаём TXT файл вне директории
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("dummy content")

    # Тестируем валидный файл с поддерживаемым расширением
    paths = [str(valid_file)]
    valid_paths = validate_paths(paths)
    assert len(valid_paths) == 1
    assert valid_paths[0] == str(valid_file)

    # Тестируем невалидный файл с неподдерживаемым расширением
    with pytest.raises(ValueError):
        validate_paths([str(invalid_file)])

    # Тестируем валидную директорию с файлами
    paths = [str(valid_dir)]
    valid_paths = validate_paths(paths)
    assert len(valid_paths) == 1
    assert valid_paths[0] == str(valid_file_in_dir)

    # Тестируем пустую директорию
    with pytest.raises(ValueError):
        validate_paths([str(empty_dir)])

    # Тестируем комбинацию валидных и невалидных путей
    paths = [str(valid_file), str(invalid_file), str(valid_dir)]
    valid_paths = validate_paths(paths)
    # Должны быть включены только файлы с поддерживаемыми расширениями
    assert len(valid_paths) == 2
    assert set(valid_paths) == {str(valid_file), str(valid_file_in_dir)}

    # Тестируем несуществующий путь
    with pytest.raises(ValueError):
        validate_paths([str(tmp_path / "nonexistent")])


def test_handle_backward_compatibility_model_command():
    # Создаем аргументы для режима model со старым стилем команд
    args = argparse.Namespace()
    args.default_paths = ['/path/to/images']
    args.generate_dataset_default = True
    args.train_ml_default = True
    args.threads_default = 4
    args.no_parallel_default = False
    args.no_gpu_default = True
    args.min_size_default = 256
    args.lpips_net_default = 'alex'
    
    # Проверяем обработку совместимости
    result = handle_backward_compatibility(args)
    
    # Проверяем, что команда правильно преобразована
    assert result.subcommand == 'model'
    assert result.paths == ['/path/to/images']
    assert result.generate_dataset is True
    assert result.train_ml is True
    assert result.threads == 4
    assert result.no_parallel is False
    assert result.no_gpu is True
    assert result.min_size == 256
    assert result.lpips_net == 'alex'


def test_handle_backward_compatibility_analyze_command():
    # Создаем аргументы для режима analyze со старым стилем команд
    args = argparse.Namespace()
    args.default_paths = ['/path/to/images']
    args.channels_default = True
    args.csv_output_default = True
    args.json_output_default = False
    args.chart_default = True
    args.theme_default = 'dark'
    args.metric_default = 'psnr'
    args.lpips_net_default = 'alex'
    args.interpolation_default = 'bilinear'
    args.min_size_default = 256
    args.threads_default = 4
    args.save_im_down_default = True
    args.save_im_up_default = False
    args.save_im_all_default = False
    args.no_parallel_default = False
    args.no_gpu_default = True
    args.ml_default = False
    args.compare_ml_default = True
    
    # Проверяем обработку совместимости
    result = handle_backward_compatibility(args)
    
    # Проверяем, что команда правильно преобразована
    assert result.subcommand == 'analyze'
    assert result.paths == ['/path/to/images']
    assert result.channels is True
    assert result.csv_output is True
    assert result.json_output is False
    assert result.chart is True
    assert result.theme == 'dark'
    assert result.metric == 'psnr'
    assert result.lpips_net == 'alex'
    assert result.interpolation == 'bilinear'
    assert result.min_size == 256
    assert result.threads == 4
    assert result.save_im_down is True
    assert result.save_im_up is False
    assert result.save_im_all is False
    assert result.no_parallel is False
    assert result.no_gpu is True
    assert result.ml is False
    assert result.compare_ml is True


def test_handle_backward_compatibility_explicit_subcommand():
    # Создаем аргументы с явно указанной подкомандой
    args = argparse.Namespace()
    args.subcommand = 'model'
    args.paths = ['/path/to/images']
    args.generate_dataset = True
    args.train_ml = True
    
    # Проверяем обработку совместимости
    result = handle_backward_compatibility(args)
    
    # Проверяем, что команда не изменилась
    assert result.subcommand == 'model'
    assert result.paths == ['/path/to/images']
    assert result.generate_dataset is True
    assert result.train_ml is True


@patch('sys.exit')
@patch('logging.error')
def test_handle_backward_compatibility_no_paths(mock_logging_error, mock_exit):
    # Создаем аргументы без путей
    args = argparse.Namespace()
    args.default_paths = []
    
    # Проверяем обработку совместимости для команды analyze
    handle_backward_compatibility(args)
    
    # Проверяем, что была ошибка и выход
    mock_logging_error.assert_called_once()
    mock_exit.assert_called_once_with(1)


@patch('sys.exit')
@patch('logging.error')
def test_handle_backward_compatibility_no_paths_for_model(mock_logging_error, mock_exit):
    # Создаем аргументы без путей для модели
    args = argparse.Namespace()
    args.default_paths = []
    args.generate_dataset_default = True
    
    # Проверяем обработку совместимости для команды model
    handle_backward_compatibility(args)
    
    # Проверяем, что была ошибка и выход
    mock_logging_error.assert_called_once()
    mock_exit.assert_called_once_with(1)


def test_create_parser():
    parser = create_parser()
    
    # Проверяем, что в парсере есть подкоманды
    subparsers = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break
    
    assert subparsers is not None
    
    # Проверяем, что есть подкоманды analyze и model
    assert 'analyze' in subparsers.choices
    assert 'model' in subparsers.choices
    
    # Проверяем, что у подкоманды model есть нужные параметры
    model_parser = subparsers.choices['model']
    model_actions = {action.dest: action for action in model_parser._actions}
    
    assert 'generate_dataset' in model_actions
    assert 'train_ml' in model_actions
    assert 'no_gpu' in model_actions
    assert 'threads' in model_actions
    assert 'no_parallel' in model_actions
    
    # Проверяем, что у подкоманды analyze есть нужные параметры
    analyze_parser = subparsers.choices['analyze']
    analyze_actions = {action.dest: action for action in analyze_parser._actions}
    
    assert 'channels' in analyze_actions
    assert 'csv_output' in analyze_actions
    assert 'ml' in analyze_actions
    assert 'compare_ml' in analyze_actions


@patch('sys.argv', ['res-suggest', 'model', '--generate-dataset', '--train-ml', '/path/to/images'])
@patch('resolution_suggester.i18n.setup_localization')
@patch('argparse.ArgumentParser.parse_args')
def test_parse_arguments_model_subcommand(mock_parse_args, mock_setup_localization):
    # Создаем мок-объект для аргументов
    mock_args = argparse.Namespace()
    mock_args.subcommand = 'model'
    mock_args.paths = ['/path/to/images']
    mock_args.generate_dataset = True
    mock_args.train_ml = True
    mock_args.lang = 'auto'
    mock_parse_args.return_value = mock_args
    
    args = parse_arguments()
    
    assert args.subcommand == 'model'
    assert args.paths == ['/path/to/images']
    assert args.generate_dataset is True
    assert args.train_ml is True


@patch('sys.argv', ['res-suggest', 'analyze', '-c', '-o', '/path/to/images'])
@patch('resolution_suggester.i18n.setup_localization')
@patch('argparse.ArgumentParser.parse_args')
def test_parse_arguments_analyze_subcommand(mock_parse_args, mock_setup_localization):
    # Создаем мок-объект для аргументов
    mock_args = argparse.Namespace()
    mock_args.subcommand = 'analyze'
    mock_args.paths = ['/path/to/images']
    mock_args.channels = True
    mock_args.csv_output = True
    mock_args.lang = 'auto'
    mock_parse_args.return_value = mock_args
    
    args = parse_arguments()
    
    assert args.subcommand == 'analyze'
    assert args.paths == ['/path/to/images']
    assert args.channels is True
    assert args.csv_output is True


@patch('sys.argv', ['res-suggest', '-c', '-o', '/path/to/images'])
@patch('resolution_suggester.i18n.setup_localization')
@patch('argparse.ArgumentParser.parse_args')
def test_parse_arguments_default_subcommand(mock_parse_args, mock_setup_localization):
    # Создаем мок-объект для аргументов
    mock_args = argparse.Namespace()
    mock_args.subcommand = None
    mock_args.default_paths = ['/path/to/images']
    mock_args.channels_default = True
    mock_args.csv_output_default = True
    mock_args.lang = 'auto'
    mock_parse_args.return_value = mock_args
    
    with patch('resolution_suggester.utils.cli.handle_backward_compatibility') as mock_handle:
        # Настраиваем поведение обработчика обратной совместимости
        mock_result = argparse.Namespace()
        mock_result.subcommand = 'analyze'
        mock_result.paths = ['/path/to/images']
        mock_result.channels = True
        mock_result.csv_output = True
        mock_result.lang = 'auto'
        mock_handle.return_value = mock_result
        
        args = parse_arguments()
        
        assert args.subcommand == 'analyze'
        assert args.paths == ['/path/to/images']
        assert args.channels is True
        assert args.csv_output is True


@patch('sys.argv', ['res-suggest', '--generate-dataset', '/path/to/images'])
@patch('resolution_suggester.i18n.setup_localization')
@patch('argparse.ArgumentParser.parse_args')
def test_parse_arguments_backward_compatibility_model(mock_parse_args, mock_setup_localization):
    # Создаем мок-объект для аргументов
    mock_args = argparse.Namespace()
    mock_args.subcommand = None
    mock_args.default_paths = ['/path/to/images']
    mock_args.generate_dataset_default = True
    mock_args.lang = 'auto'
    mock_parse_args.return_value = mock_args
    
    with patch('resolution_suggester.utils.cli.handle_backward_compatibility') as mock_handle:
        # Настраиваем поведение обработчика обратной совместимости
        mock_result = argparse.Namespace()
        mock_result.subcommand = 'model'
        mock_result.paths = ['/path/to/images']
        mock_result.generate_dataset = True
        mock_result.lang = 'auto'
        mock_handle.return_value = mock_result
        
        args = parse_arguments()
        
        assert args.subcommand == 'model'
        assert args.paths == ['/path/to/images']
        assert args.generate_dataset is True
