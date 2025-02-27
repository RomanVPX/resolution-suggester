#!/usr/bin/env python3
# scripts/update_translations.py
"""
Утилита для обновления файлов переводов (PO/MO) в проекте.
Извлекает строки для перевода из исходного кода, обновляет
существующие переводы и создаёт новые при необходимости.
"""
import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("translation_updater")


def check_dependencies() -> bool:
    """Проверяет наличие необходимых внешних программ."""
    dependencies = ["xgettext", "msgmerge", "msgfmt"]
    missing = []

    for cmd in dependencies:
        if shutil.which(cmd) is None:
            missing.append(cmd)

    if missing:
        logger.error(f"Не найдены необходимые программы: {', '.join(missing)}")
        logger.error("Установите пакет gettext перед использованием этого скрипта.")
        return False

    return True


def run_subprocess(cmd: List[str], error_msg: str) -> Tuple[bool, Optional[str]]:
    """Запускает внешнюю команду и обрабатывает возможные ошибки.

    Args:
        cmd: Список команды и аргументов для subprocess.run
        error_msg: Сообщение об ошибке для вывода в случае проблемы

    Returns:
        Tuple[bool, Optional[str]]: (успех, вывод команды)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"{error_msg}: {result.stderr}")
            logger.debug(f"Команда: {' '.join(cmd)}")
            return False, None

        return True, result.stdout
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}")
        logger.debug(f"Команда: {' '.join(cmd)}")
        return False, None


def extract_strings(project_root: Path, source_dir: Path, pot_file: str) -> bool:
    """Извлекает строки для перевода из исходного кода в POT-файл."""
    logger.info("Извлечение строк для перевода...")

    # Собираем все Python файлы из исходной директории
    py_files = []
    for py_file in source_dir.glob("**/*.py"):
        rel_path = py_file.relative_to(project_root)
        py_files.append(str(rel_path))

    if not py_files:
        logger.warning(f"Не найдены Python файлы в {source_dir}")
        return False

    # Формируем команду xgettext
    cmd = [
              "xgettext",
              "--language=Python",
              "--keyword=_",
              f"--output={pot_file}",
              "--from-code=UTF-8",
              "--add-comments=TRANSLATORS:",
              "--sort-by-file",
              f"--directory={project_root}",
          ] + py_files

    success, _ = run_subprocess(cmd, "Ошибка при извлечении строк")
    if not success:
        return False

    # Исправляем кодировку в POT-файле
    try:
        with open(pot_file, 'r', encoding='utf-8') as f:
            pot_content = f.read()

        if 'charset=CHARSET' in pot_content:
            pot_content = pot_content.replace('charset=CHARSET', 'charset=UTF-8')

            with open(pot_file, 'w', encoding='utf-8') as f:
                f.write(pot_content)
    except Exception as e:
        logger.error(f"Ошибка при обработке POT-файла: {e}")
        return False

    return True


def process_english_translations(po_file: Path) -> bool:
    """Автоматически заполняет английские переводы копированием исходного текста."""
    logger.info(f"Обработка английских переводов: {po_file}")

    try:
        with open(po_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Разделяем файл на записи (более надёжный способ, чем простой split)
        pattern = r'(msgid\s+(".*?"(?:\s+".*?")*))[\r\n]+(msgstr\s+(".*?"(?:\s+".*?")*))'

        def replace_match(match):
            msgid = match.group(1)
            msgstr = match.group(3)

            if msgstr == 'msgstr ""':
                # Заменяем пустой msgstr на значение msgid
                return f"{msgid}\n{msgid.replace('msgid', 'msgstr')}"
            return match.group(0)

        # Заменяем пустые переводы
        updated_content = re.sub(pattern, replace_match, content, flags=re.DOTALL)

        with open(po_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        return True
    except Exception as e:
        logger.error(f"Ошибка при обработке английских переводов: {e}")
        return False


def compile_po_to_mo(po_file: Path, mo_file: Path) -> bool:
    """Компилирует PO-файл в бинарный MO-файл."""
    logger.info(f"Компиляция {po_file} в {mo_file}")

    cmd = ["msgfmt", "--statistics", "-o", str(mo_file), str(po_file)]
    success, output = run_subprocess(cmd, f"Ошибка при компиляции {po_file}")

    if success and output:
        logger.info(f"Статистика для {po_file}: {output.strip()}")

    return success


def update_translation(lang: str, locale_dir: Path, pot_file: str) -> bool:
    """Обновляет перевод для указанного языка."""
    logger.info(f"Обновление перевода для языка: {lang}")

    lang_dir = locale_dir / lang / "LC_MESSAGES"
    lang_dir.mkdir(parents=True, exist_ok=True)

    po_file = lang_dir / "messages.po"
    mo_file = lang_dir / "messages.mo"

    if po_file.exists():
        # Проверяем и исправляем кодировку в существующем PO-файле
        try:
            with open(po_file, 'r', encoding='utf-8') as f:
                po_content = f.read()

            if 'charset=CHARSET' in po_content:
                po_content = po_content.replace('charset=CHARSET', 'charset=UTF-8')
                with open(po_file, 'w', encoding='utf-8') as f:
                    f.write(po_content)
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {po_file}: {e}")
            return False

        # Обновляем существующий PO-файл
        merge_cmd = [
            "msgmerge",
            "--update",
            "--backup=none",
        ]

        # Для английского отключаем нечёткое сопоставление
        if lang == "en":
            merge_cmd.append("--no-fuzzy-matching")

        merge_cmd.extend([str(po_file), pot_file])

        success, _ = run_subprocess(
            merge_cmd,
            f"Ошибка при обновлении {po_file}"
        )
        if not success:
            return False
    else:
        # Создаём новый PO-файл из шаблона POT
        logger.info(f"Создание нового файла перевода: {po_file}")
        try:
            with open(pot_file, 'r', encoding='utf-8') as src:
                with open(po_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        except Exception as e:
            logger.error(f"Ошибка при создании файла {po_file}: {e}")
            return False

    # Для английского языка автоматически заполняем переводы
    if lang == "en":
        if not process_english_translations(po_file):
            return False

    # Компилируем PO в MO
    return compile_po_to_mo(po_file, mo_file)


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description="Обновляет файлы переводов проекта."
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=["en", "ru"],
        help="Список языков для обновления (по умолчанию: en ru)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный режим вывода"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Проверяем наличие необходимых зависимостей
    if not check_dependencies():
        return 1

    # Определяем пути проекта
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "src" / "resolution_suggester"
    locale_dir = source_dir / "locales"
    locale_dir.mkdir(exist_ok=True)

    # Создаём временный POT-файл
    with tempfile.NamedTemporaryFile(suffix='.pot', delete=False) as temp_file:
        temp_pot = temp_file.name

    try:
        # Извлекаем строки для перевода
        if not extract_strings(project_root, source_dir, temp_pot):
            return 1

        # Обновляем переводы для каждого языка
        for lang in args.languages:
            if not update_translation(lang, locale_dir, temp_pot):
                logger.error(f"Не удалось обновить переводы для языка: {lang}")
            else:
                logger.info(f"Переводы для языка {lang} успешно обновлены")

        return 0

    finally:
        # Удаляем временный файл
        if os.path.exists(temp_pot):
            os.unlink(temp_pot)


if __name__ == "__main__":
    sys.exit(main())