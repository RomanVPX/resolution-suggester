# tests/utils/test_reporters.py
from resolution_suggester.config import QualityMetrics
from resolution_suggester.utils.reporters import CSVReporter


def test_csv_reporter(tmp_path):
    # Создаём тестовые данные
    output_path = tmp_path / "test_output.csv"
    results = [
        ("1024x1024", float('inf'), "Оригинал"),
        ("512x512", 45.0, "практически идентичные изображения"),
        ("256x256", 35.0, "очень хорошее качество")
    ]

    # Тестируем запись в CSV
    with CSVReporter(str(output_path), QualityMetrics.PSNR) as reporter:
        reporter.write_header(False)
        reporter.write_results("test.png", results, False)

    # Проверяем, что файл создан и содержит правильные данные
    assert output_path.exists()
    content = output_path.read_text()
    assert "test.png" in content
    assert "1024x1024" in content
    assert "512x512" in content
    assert "45.00" in content
