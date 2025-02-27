#!/usr/bin/env python3
# scripts/update_translations.py
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent

    source_dir = project_root / "src" / "resolution_suggester"
    locale_dir = source_dir / "locales"
    locale_dir.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.pot', delete=False) as temp_file:
        temp_pot = temp_file.name

    try:
        cmd = [
            "xgettext",
            "--language=Python",
            "--keyword=_",
            f"--output={temp_pot}",
            "--from-code=UTF-8",
            f"--directory={project_root}",
        ]

        for py_file in source_dir.glob("**/*.py"):
            rel_path = py_file.relative_to(project_root)
            cmd.append(str(rel_path))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Ошибка при извлечении строк: {result.stderr}")
            return

        with open(temp_pot, 'r', encoding='utf-8') as f:
            pot_content = f.read()

        pot_content = pot_content.replace('charset=CHARSET', 'charset=UTF-8')

        with open(temp_pot, 'w', encoding='utf-8') as f:
            f.write(pot_content)

        for lang in ["en", "ru"]:
            lang_dir = locale_dir / lang / "LC_MESSAGES"
            lang_dir.mkdir(parents=True, exist_ok=True)

            po_file = lang_dir / "messages.po"

            if po_file.exists():
                with open(po_file, 'r', encoding='utf-8') as f:
                    po_content = f.read()

                if 'charset=CHARSET' in po_content:
                    po_content = po_content.replace('charset=CHARSET', 'charset=UTF-8')
                    with open(po_file, 'w', encoding='utf-8') as f:
                        f.write(po_content)

                if lang == "en":
                    merge_cmd = [
                        "msgmerge",
                        "--update",
                        "--no-fuzzy-matching",
                        "--backup=none",
                        str(po_file),
                        temp_pot
                    ]
                else:
                    merge_cmd = [
                        "msgmerge",
                        "--update",
                        "--backup=none",
                        str(po_file),
                        temp_pot
                    ]

                subprocess.run(merge_cmd)
            else:
                with open(temp_pot, 'r', encoding='utf-8') as src:
                    with open(po_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())

            if lang == "en":
                script_content = """
import re
import sys

def process_po_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = re.split(r'\\n\\n', content)

    for i in range(len(entries)):
        entry = entries[i]

        if entry.startswith('#~') or 'msgid ""' in entry and 'msgstr ""' in entry:
            continue

        msgid_match = re.search(r'msgid\\s+(".*?"(?:\\s+".*?")*)', entry, re.DOTALL)
        msgstr_match = re.search(r'msgstr\\s+(".*?"(?:\\s+".*?")*)', entry, re.DOTALL)

        if msgid_match and msgstr_match:
            msgid = msgid_match.group(1)
            msgstr = msgstr_match.group(1)

            if msgstr == '""':
                # Заменяем пустой msgstr на msgid
                entry = entry.replace('msgstr ""', f'msgstr {msgid}')
                entries[i] = entry

    new_content = '\\n\\n'.join(entries)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    process_po_file(sys.argv[1])
"""

                with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as script_file:
                    script_path = script_file.name
                    script_file.write(script_content)

                try:
                    subprocess.run(["python", script_path, str(po_file)])
                finally:
                    if os.path.exists(script_path):
                        os.unlink(script_path)

            mo_file = lang_dir / "messages.mo"
            subprocess.run(["msgfmt", "-o", str(mo_file), str(po_file)])

            print(f"Updated translations for language: {lang}")

    finally:
        if os.path.exists(temp_pot):
            os.unlink(temp_pot)


if __name__ == "__main__":
    main()
