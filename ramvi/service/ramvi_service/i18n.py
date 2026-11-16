"""3.14 (CORE-I18N-01): сообщения пользователю — ключ + параметры, тексты в каталогах.

Сервис не собирает фраз: он отдаёт `{"code": "<ключ>", "params": {...}}`, текст на
выбранном языке показывает оболочка. Каталоги — `ramvi/i18n/<язык>.json` (общие для
сервиса и оболочки) и `<модуль>/i18n/<язык>.json` у модулей. Параметр в тексте — `{имя}`;
значением параметра может быть другое сообщение (`Msg`), оно переводится тем же языком.

`check_catalogs` — правила, которые проверяет CI (`tools/check_i18n.py`): у всех языков
один набор ключей, переводы не пустые, у каждого ключа одинаковый набор параметров.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LANGUAGES = ("ru", "en")
# ramvi/service/ramvi_service/i18n.py -> ramvi/i18n; в переносимом комплекте — то же место.
CORE_DIR = Path(__file__).resolve().parents[2] / "i18n"
PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
KEY = re.compile(r"[a-z][a-z0-9_]*(\.[a-z0-9_]+)+")


@dataclass(frozen=True)
class Msg:
    """Сообщение для перевода: ключ каталога и параметры."""

    code: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return {"code": self.code, "params": {k: to_json(v) for k, v in self.params.items()}}


def msg(code: str, **params: Any) -> Msg:
    return Msg(code, params)


def to_json(value: Any) -> Any:
    """Значение параметра для JSON: вложенные `Msg` — объекты `{code, params}`."""
    if isinstance(value, Msg):
        return value.json()
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, Mapping):
        return {k: to_json(v) for k, v in value.items()}
    return value


class UserError(Exception):
    """Ошибка, которую видит пользователь: ключ и параметры вместо готовой фразы."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(code)
        self.code, self.params = code, params

    @property
    def message(self) -> Msg:
        return Msg(self.code, self.params)


def load_catalog(language: str, folder: Path = CORE_DIR) -> dict[str, str]:
    """Каталог языка; нет файла — пустой (тогда показывается сам ключ)."""
    path = Path(folder) / f"{language}.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
    ):
        raise ValueError(f"{path}: expected an object of strings")
    return raw


def render(message: Msg | str, catalog: Mapping[str, str], **params: Any) -> str:
    """Текст сообщения; вложенные `Msg` в параметрах переводятся тем же каталогом."""
    if isinstance(message, Msg):
        code, values = message.code, {**message.params, **params}
    else:
        code, values = message, params
    template = catalog.get(code)
    if template is None:
        return code

    def value(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        item = values[name]
        if isinstance(item, Msg):
            return render(item, catalog)
        if isinstance(item, (list, tuple)):
            return "; ".join(render(v, catalog) if isinstance(v, Msg) else str(v) for v in item)
        return "" if item is None else str(item)

    return PARAM.sub(value, template)


def params_of(template: str) -> set[str]:
    return set(PARAM.findall(template))


def check_catalogs(
    catalogs: Mapping[str, Mapping[str, str]],
    *,
    languages: tuple[str, ...] = LANGUAGES,
    where: str = "core",
    allowed: re.Pattern[str] | None = None,
) -> list[str]:
    """Ошибки каталогов: нет языка, разный набор ключей, пустой перевод, разные параметры.

    `allowed` — какими могут быть ключи (модуль не переопределяет ключи ядра).
    """
    errors: list[str] = []
    missing = [lang for lang in languages if lang not in catalogs]
    errors += [f"{where}: no catalog for language {lang}" for lang in missing]
    present = [lang for lang in languages if lang in catalogs]
    keys = set().union(*(catalogs[lang].keys() for lang in present)) if present else set()
    for key in sorted(keys):
        if not KEY.fullmatch(key):
            errors.append(f"{where}: bad key {key!r}")
        if allowed is not None and not allowed.fullmatch(key):
            errors.append(f"{where}: key {key!r} is outside the module namespace")
        texts = {lang: catalogs[lang].get(key) for lang in present}
        for lang, text in texts.items():
            if text is None:
                errors.append(f"{where}: {lang} has no translation for {key}")
            elif not text.strip():
                errors.append(f"{where}: {lang} translation of {key} is empty")
        found = {lang: params_of(t) for lang, t in texts.items() if t}
        if len({frozenset(p) for p in found.values()}) > 1:
            detail = ", ".join(f"{lang}={sorted(p)}" for lang, p in found.items())
            errors.append(f"{where}: {key} has different parameters: {detail}")
    return errors
