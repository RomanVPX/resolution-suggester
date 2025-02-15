#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! python3 -m venv .venv; then
  echo "Ошибка при создании виртуального окружения. Убедитесь, что установлен python3 и venv."
  exit 1
fi

source .venv/bin/activate

if ! pip install -r requirements.txt; then
  echo "Ошибка при установке зависимостей из requirements.txt. Убедитесь, что файл requirements.txt находится в директории скрипта."
  exit 1
fi

echo "\nВиртуальное окружение .venv успешно создано и активировано."
echo "Зависимости установлены.\n"
echo "Теперь вы можете запускать скрипт resolution_suggester.py."
echo "Для запуска скрипта в активированном окружении, просто выполните:"
echo "python resolution_suggester.py <путь_к_файлу_или_директории> [опции]"
echo "\nКогда закончите работу, для выхода из виртуального окружения выполните команду: deactivate"
