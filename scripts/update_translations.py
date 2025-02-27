#!/usr/bin/env python3
# scripts/update_translations.py
"""
Utility for updating translation files (PO/MO) in the project.
Extracts strings for translation from the source code, updates
existing translations, and creates new ones when necessary.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("translation_updater")


def check_dependencies() -> bool:
    """Checks for required external programs."""
    dependencies = ["xgettext", "msgmerge", "msgfmt"]
    missing = []

    for cmd in dependencies:
        if shutil.which(cmd) is None:
            missing.append(cmd)

    if missing:
        logger.error(f"Required programs not found: {', '.join(missing)}")
        logger.error("Install the gettext package before using this script.")
        return False

    return True


def run_subprocess(cmd: List[str], error_msg: str) -> Tuple[bool, Optional[str]]:
    """Runs an external command and handles possible errors.

    Args:
        cmd: List of command and arguments for subprocess.run
        error_msg: Error message to display in case of problems

    Returns:
        Tuple[bool, Optional[str]]: (success, command output)
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
            logger.debug(f"Command: {' '.join(cmd)}")
            return False, None

        return True, result.stdout
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}")
        logger.debug(f"Command: {' '.join(cmd)}")
        return False, None


def extract_strings(project_root: Path, source_dir: Path, pot_file: str) -> bool:
    """Extracts strings for translation from source code to a POT file."""
    logger.info("Extracting strings for translation...")

    py_files = []
    for py_file in source_dir.glob("**/*.py"):
        rel_path = py_file.relative_to(project_root)
        py_files.append(str(rel_path))

    if not py_files:
        logger.warning(f"No Python files found in {source_dir}")
        return False

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

    success, _ = run_subprocess(cmd, "Error extracting strings")
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
        logger.error(f"Error processing POT file: {e}")
        return False

    return True


def process_english_translations(po_file: Path) -> bool:
    """Automatically fills English translations by copying the source text."""
    logger.info(f"Processing English translations: {po_file}")

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
        logger.error(f"Error processing English translations: {e}")
        return False


def compile_po_to_mo(po_file: Path, mo_file: Path) -> bool:
    """Compiles a PO file into a binary MO file."""
    logger.info(f"Compiling {po_file} to {mo_file}")

    cmd = ["msgfmt", "--statistics", "-o", str(mo_file), str(po_file)]
    success, output = run_subprocess(cmd, f"Error compiling {po_file}")

    if success and output:
        logger.info(f"Statistics for {po_file}: {output.strip()}")

    return success


def update_translation(lang: str, locale_dir: Path, pot_file: str) -> bool:
    """Updates the translation for the specified language."""
    logger.info(f"Updating translation for language: {lang}")

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
            logger.error(f"Error processing file {po_file}: {e}")
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
            f"Error updating {po_file}"
        )
        if not success:
            return False
    else:
        # Создаём новый PO-файл из шаблона POT
        logger.info(f"Creating new translation file: {po_file}")
        try:
            with open(pot_file, 'r', encoding='utf-8') as src:
                with open(po_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        except Exception as e:
            logger.error(f"Error creating file {po_file}: {e}")
            return False

    # Для английского языка автоматически заполняем переводы
    if lang == "en":
        if not process_english_translations(po_file):
            return False

    return compile_po_to_mo(po_file, mo_file)


def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser(
        description="Updates project translation files."
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=["en", "ru"],
        help="List of languages to update (default: en ru)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output mode"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not check_dependencies():
        return 1

    project_root = Path(__file__).parent.parent
    source_dir = project_root / "src" / "resolution_suggester"
    locale_dir = source_dir / "locales"
    locale_dir.mkdir(exist_ok=True)

    # Создаём временный POT-файл
    with tempfile.NamedTemporaryFile(suffix='.pot', delete=False) as temp_file:
        temp_pot = temp_file.name

    try:
        if not extract_strings(project_root, source_dir, temp_pot):
            return 1

        for lang in args.languages:
            if not update_translation(lang, locale_dir, temp_pot):
                logger.error(f"Failed to update translations for language: {lang}")
            else:
                logger.info(f"Translations for language {lang} successfully updated")

        return 0

    finally:
        if os.path.exists(temp_pot):
            os.unlink(temp_pot)


if __name__ == "__main__":
    sys.exit(main())