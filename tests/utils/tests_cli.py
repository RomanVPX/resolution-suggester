# tests/utils/test_cli.py
import pytest

from resolution_suggester.utils.cli import validate_paths


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
