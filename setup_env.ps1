$ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
Set-Location -Path $ScriptDir

if (-not (python -m venv .venv)) {
    Write-Error "Ошибка при создании виртуального окружения. Убедитесь, что Python установлен и добавлен в PATH."
    exit 1
}

. .venv/Scripts/activate

if (-not (pip install -r requirements.txt)) {
    Write-Error "Ошибка при установке зависимостей из requirements.txt. Убедитесь, что файл requirements.txt находится в директории скрипта."
    exit 1
}

Write-Host "`nВиртуальное окружение .venv успешно создано и активировано."
Write-Host "Зависимости установлены.`n"
Write-Host "Теперь вы можете запускать скрипт resolution_suggester.py."
Write-Host "Для запуска скрипта в активированном окружении, просто выполните:"
Write-Host "python resolution_suggester.py <путь_к_файлу_или_директории> [опции]"
Write-Host "`nКогда закончите работу, для выхода из виртуального окружения выполните команду: deactivate"
