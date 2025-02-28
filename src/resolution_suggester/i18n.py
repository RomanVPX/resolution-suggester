# i18n.py
import gettext
import locale
import sys
from pathlib import Path
from typing import Optional

_ = lambda s: s


def detect_language_from_args() -> Optional[str]:
    """
    Проверяет аргументы командной строки на наличие параметра --lang
    до инициализации argparse.

    Returns:
        Код языка ('en', 'ru') или None, если язык не указан
    """
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--lang' and i < len(sys.argv):
            lang = sys.argv[i + 1]
            if lang in ['en', 'ru']:
                return lang
        elif arg.startswith('--lang='):
            lang = arg.split('=')[1]
            if lang in ['en', 'ru']:
                return lang
    return None


def setup_localization(force_lang: Optional[str] = None) -> callable:
    """
    Настраивает локализацию приложения.

    Args:
        force_lang: Принудительно указанный язык (например, 'en', 'ru')
                   Если None, определяется из системных настроек

    Returns:
        Функция перевода gettext
    """
    global _

    locale_dir = Path(__file__).parent / 'locales'

    if force_lang:
        lang_code = force_lang
    else:
        cmd_lang = detect_language_from_args()
        if cmd_lang:
            lang_code = cmd_lang
        else:
            try:
                system_locale = locale.getlocale()
                if system_locale is None:
                    system_locale = 'en_US'

                lang_code = system_locale[:2]

                if not (locale_dir / lang_code).exists():
                    lang_code = 'en'
            except:
                lang_code = 'en'

    translation = gettext.translation('messages',
                                      localedir=str(locale_dir),
                                      languages=[lang_code],
                                      fallback=True)

    _ = translation.gettext

    return _

_ = setup_localization()
