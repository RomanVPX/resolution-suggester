#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    if ! python3 -m venv .venv; then
        echo "Ошибка при создании виртуального окружения. Убедитесь, что установлен python3 и venv."
        exit 1
    fi
    echo "Виртуальное окружение создано."
else
    echo "Виртуальное окружение уже существует."
fi

source .venv/bin/activate || exit 1

if ! pip install -r requirements.txt; then
    echo "Ошибка при установке зависимостей из requirements.txt. Убедитесь, что файл requirements.txt находится в директории скрипта."
    exit 1
fi

cat > activate.sh << EOL
#!/bin/bash
source "${SCRIPT_DIR}/.venv/bin/activate"
printf "\nВиртуальное окружение .venv успешно активировано.\n"
printf "Для запуска скрипта в активированном окружении, просто выполните:\n"
printf "python resolution_suggester.py <путь_к_файлу_или_директории> [опции]\n"
printf "\nДля выхода из виртуального окружения, выполните команду: deactivate\n"
EOL

chmod +x activate.sh

printf "\nУстановка завершена успешно!\n"
printf "Для активации окружения:\n"
printf "source ./activate.sh\n"
