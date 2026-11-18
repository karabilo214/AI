"""Модели распознавания речи (ADR-0004): файл, адрес, размер и SHA-256.

Файлы лежат в папке моделей, подпапка `stt/`; скачиваются менеджером загрузок 4.3
с проверкой SHA-256 до активации.
"""

from dataclasses import dataclass
from pathlib import Path

FOLDER = "stt"
LANGUAGES = ("auto", "ru", "en", "uk")


@dataclass(frozen=True)
class SttModel:
    id: str
    file: str
    size: int
    sha256: str

    repo: str = "ggerganov/whisper.cpp"

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.file}"

    def path(self, models_dir: Path) -> Path:
        return Path(models_dir) / FOLDER / self.file


MODELS = {
    m.id: m
    for m in (
        SttModel(
            "large-v3-turbo-q5_0",
            "ggml-large-v3-turbo-q5_0.bin",
            574_041_195,
            "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        ),
        SttModel(
            "small-q5_1",
            "ggml-small-q5_1.bin",
            190_085_487,
            "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
        ),
    )
}
DEFAULT_MODEL = "large-v3-turbo-q5_0"

# Детектор речи Silero для whisper.cpp: без него на тишине whisper выдумывает текст
# («Продолжение следует...»). Скачивается вместе с любой моделью распознавания.
VAD = SttModel(
    "silero-v5.1.2",
    "ggml-silero-v5.1.2.bin",
    885_098,
    "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    repo="ggml-org/whisper-vad",
)


def installed(model: SttModel, models_dir: Path) -> bool:
    """Модель готова: на месте и её файл, и детектор речи."""
    return model.path(models_dir).is_file() and VAD.path(models_dir).is_file()


def join_segments(texts: list[str]) -> str:
    """Склеить фрагменты whisper как есть: слова внутри фрагмента уже начинаются с пробела,
    а граница фрагмента может прийти посреди слова (склейка через пробел или «\\n» рвёт его)."""
    return " ".join("".join(texts).split())
