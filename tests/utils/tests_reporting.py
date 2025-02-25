# tests/utils/test_reporting.py
import pytest

from resolution_suggester.config import QualityMetrics, QUALITY_LEVEL_HINTS_DESCRIPTIONS, QualityLevelHints
from resolution_suggester.utils.reporting import QualityHelper


def test_get_hint():
    # Тестируем все пороговые значения для PSNR
    assert (QualityHelper.get_hint(50, QualityMetrics.PSNR) ==
            QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.EXCELLENT])
    assert (QualityHelper.get_hint(40, QualityMetrics.PSNR) ==
            QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.VERY_GOOD])
    assert (QualityHelper.get_hint(30, QualityMetrics.PSNR) ==
            QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.GOOD])
    assert (QualityHelper.get_hint(29, QualityMetrics.PSNR) ==
            QUALITY_LEVEL_HINTS_DESCRIPTIONS[QualityLevelHints.NOTICEABLE_LOSS])

    # Проверяем, что функция выбрасывает исключение для неизвестной метрики
    with pytest.raises(ValueError):
        QualityHelper.get_hint(100, "unknown_metric")
