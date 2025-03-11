# tests/utils/test_cli.py
import argparse
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from resolution_suggester.utils.cli import create_parser, parse_arguments, validate_paths


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
    mock_args.min_size = 16
    mock_args.threads = 4
    mock_args.no_parallel = False
    mock_args.no_gpu = False
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
    mock_args.json_output = False
    mock_args.ml = False
    mock_args.compare_ml = False
    mock_args.save_im_down = False
    mock_args.save_im_up = False
    mock_args.save_im_all = False
    mock_args.min_size = 16
    mock_args.threads = 4
    mock_args.lang = 'auto'
    mock_parse_args.return_value = mock_args
    
    args = parse_arguments()
    
    assert args.subcommand == 'analyze'
    assert args.paths == ['/path/to/images']
    assert args.channels is True
    assert args.csv_output is True


@patch('sys.argv', ['res-suggest', 'model', '--train-ml', '/path/to/images'])
@patch('resolution_suggester.i18n.setup_localization')
@patch('argparse.ArgumentParser.parse_args')
def test_parse_arguments_model_train_only(mock_parse_args, mock_setup_localization):
    # Создаем мок-объект для аргументов
    mock_args = argparse.Namespace()
    mock_args.subcommand = 'model'
    mock_args.paths = ['/path/to/images']
    mock_args.generate_dataset = False
    mock_args.train_ml = True
    mock_args.lang = 'auto'
    mock_args.min_size = 16
    mock_args.threads = 4
    mock_args.no_parallel = False
    mock_args.no_gpu = False
    mock_parse_args.return_value = mock_args
    
    args = parse_arguments()
    
    assert args.subcommand == 'model'
    assert args.paths == ['/path/to/images']
    assert args.generate_dataset is False
    assert args.train_ml is True


@patch('sys.stdout')
@patch('sys.argv', ['res-suggest', '--help'])
@patch('resolution_suggester.i18n.setup_localization')
@patch('resolution_suggester.utils.cli.sys.exit')
def test_parse_arguments_main_help(mock_exit, mock_setup_localization, mock_stdout, capfd):
    # Патчим так, чтобы выход не происходил
    mock_exit.side_effect = RuntimeError("Exit called")
    
    # Проверяем, что вызывается sys.exit при запросе помощи
    with pytest.raises(RuntimeError, match="Exit called"):
        parse_arguments()
    
    # Проверяем, что был вызов sys.exit
    mock_exit.assert_called()
