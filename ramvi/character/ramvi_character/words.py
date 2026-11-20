"""Word handling shared by search, consolidation and forgetting."""

import re

STOP = frozenset(
    "и в во на не что это как а но или ли же бы то за по из у о об от до для с со к ко"
    " я ты он она мы вы они мне меня тебе тебя его её ее нам вас их мой твой наш ваш"
    " был была были быть есть было будет очень там тут где когда так уже ещё еще"
    " the a an and or but is are was were be to of in on at for with it this that"
    " i you he she we they me my your our their do does did not what how".split()
)
WORD = re.compile(r"\w+", re.UNICODE)


def terms(value: str, limit: int = 16) -> list[str]:
    result = []
    for word in WORD.findall(value.casefold()):
        if word in STOP or len(word) < 2 or (word.isdigit() and len(word) < 3):
            continue
        if word not in result:
            result.append(word)
    return result[:limit]


def stem(word: str) -> str:
    """A short prefix catches Russian word endings (файл/файлы/файлом)."""
    return word[: max(4, len(word) - 2)] if len(word) >= 5 else word


def stems(value: str) -> set[str]:
    return {stem(w) for w in terms(value, 64)}


def norm(value: str) -> str:
    return " ".join(value.casefold().split())
