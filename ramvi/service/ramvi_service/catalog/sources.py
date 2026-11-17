"""Какие источники читает каталог по умолчанию (4.2).

ТЗ: «каталог шире списка совместимости не делать» — только Qwen3 (qwen3 и
qwen3moe). Неподдерживаемые файлы этих источников всё равно отсеивает `support.py`.
"""

HUGGINGFACE_REPOS = [
    "Qwen/Qwen3-0.6B-GGUF",
    "Qwen/Qwen3-1.7B-GGUF",
    "Qwen/Qwen3-4B-GGUF",
    "Qwen/Qwen3-8B-GGUF",
    "Qwen/Qwen3-14B-GGUF",
    "Qwen/Qwen3-32B-GGUF",
    "Qwen/Qwen3-30B-A3B-GGUF",
]

OLLAMA_TAGS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:32b",
    "qwen3:30b",
    "qwen3:0.6b-fp16",
    "qwen3:1.7b-fp16",
    "qwen3:4b-fp16",
    "qwen3:8b-fp16",
    "qwen3:14b-fp16",
]
