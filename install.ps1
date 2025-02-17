# install.ps1
$ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
Set-Location -Path $ScriptDir

if (-not (Test-Path -Path ".venv")) {
    if (-not (python -m venv .venv)) {
        Write-Error "Ошибка при создании виртуального окружения. Убедитесь, что Python установлен и добавлен в PATH."
        exit 1
    }
    Write-Host "Виртуальное окружение создано."
} else {
    Write-Host "Виртуальное окружение уже существует."
}

. .venv/Scripts/activate

if (-not (pip install -r requirements.txt)) {
    Write-Error "Ошибка при установке зависимостей из requirements.txt. Убедитесь, что файл requirements.txt находится в директории скрипта."
    exit 1
}

@"
`$ScriptDir = "$ScriptDir"
Set-Location -Path `$ScriptDir
. .venv/Scripts/activate

Write-Host "`nВиртуальное окружение .venv успешно активировано."
Write-Host "Для запуска скрипта в активированном окружении, просто выполните:"
Write-Host "python resolution_suggester.py <путь_к_файлу_или_директории> [опции]"
Write-Host "`nДля выхода из виртуального окружения, выполните команду: deactivate"
"@ | Out-File -FilePath "activate.ps1" -Encoding UTF8

Write-Host "`nУстановка завершена успешно!"
Write-Host "Для активации окружения выполните:"
Write-Host ". ./activate.ps1"
