"""3.14 (CORE-I18N-01): каталоги, ключи в коде, тексты вне каталога, переводы модулей."""

import json
from pathlib import Path
from typing import Any

import check_i18n
import pytest
from ramvi_service.app import build
from ramvi_service.contracts.modules import Manifest
from ramvi_service.i18n import (
    LANGUAGES,
    Msg,
    UserError,
    check_catalogs,
    load_catalog,
    msg,
    render,
    to_json,
)
from ramvi_service.modules.host import ModuleHost
from ramvi_service.sections import SectionRegistry
from test_chat import ENGINE, TOKEN, Client


def test_repository_passes_the_ci_check() -> None:
    assert check_i18n.catalog_errors() == []
    assert check_i18n.missing_keys() == []
    assert check_i18n.hardcoded_texts() == []
    assert check_i18n.main() == 0
    # Проверка правда видит ключи и Python, и C#, и разметки.
    places = {where.split("/")[0] for _, where in check_i18n.used_keys()}
    assert places == {"service", "desktop"}
    assert any(
        where.endswith(".axaml") or ".axaml:" in where for _, where in check_i18n.used_keys()
    )


def test_catalog_rules() -> None:
    good = {"ru": {"a.b": "Привет, {name}"}, "en": {"a.b": "Hello, {name}"}}
    assert check_catalogs(good) == []
    assert check_catalogs({"ru": good["ru"]}) == ["core: no catalog for language en"]
    assert check_catalogs({"ru": good["ru"], "en": {}}) == ["core: en has no translation for a.b"]
    assert check_catalogs({"ru": {"a.b": " "}, "en": {"a.b": "x"}}) == [
        "core: ru translation of a.b is empty"
    ]
    errors = check_catalogs({"ru": {"a.b": "{name}"}, "en": {"a.b": "{who}"}})
    assert errors == ["core: a.b has different parameters: ru=['name'], en=['who']"]
    assert check_catalogs({"ru": {"Bad": "x"}, "en": {"Bad": "y"}}) == ["core: bad key 'Bad'"]


def make_repo(tmp_path: Path, ru: dict[str, str], en: dict[str, str]) -> Path:
    (tmp_path / "i18n").mkdir()
    (tmp_path / "i18n" / "ru.json").write_text(json.dumps(ru, ensure_ascii=False), "utf-8")
    (tmp_path / "i18n" / "en.json").write_text(json.dumps(en), "utf-8")
    for folder in (*check_i18n.PYTHON, *check_i18n.CSHARP):
        (tmp_path / folder).mkdir(parents=True)
    return tmp_path


def test_check_finds_missing_keys_and_texts_outside_catalog(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"chat.send": "Отправить"}, {"chat.send": "Send"})
    service = root / check_i18n.PYTHON[0]
    (service / "a.py").write_text(
        '"""Докстринг по-русски можно."""\n'
        'X = "chat.send"\n'
        'raise ValueError("chat.gone")\n'
        'Y = "settings.json"\n'
        'Z = "Сообщение пользователю"\n',
        "utf-8",
    )
    shell = root / check_i18n.CSHARP[0]
    (shell / "A.cs").write_text(
        'var a = L["chat.send"]; // комментарий можно\n'
        'var b = L.Format("chat.none", ("x", 1));\n'
        'throw new Exception("Готово");\n',
        "utf-8",
    )
    (shell / "B.axaml").write_text(
        '<!-- комментарий -->\n<TextBlock Text="{l:T chat.missing}" />\n<TextBlock Text="Привет" />\n',
        "utf-8",
    )

    assert check_i18n.missing_keys(root) == [
        f"{check_i18n.PYTHON[0]}/a.py:3: key chat.gone is not in i18n/",
        f"{check_i18n.CSHARP[0]}/A.cs:2: key chat.none is not in i18n/",
        f"{check_i18n.CSHARP[0]}/B.axaml:2: key chat.missing is not in i18n/",
    ]
    texts = check_i18n.hardcoded_texts(root)
    assert [t.split(": user text")[0] for t in texts] == [
        f"{check_i18n.PYTHON[0]}/a.py:5",
        f"{check_i18n.CSHARP[0]}/A.cs:3",
        f"{check_i18n.CSHARP[0]}/B.axaml:3",
    ]


def test_check_validates_module_catalogs(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"chat.send": "Отправить"}, {"chat.send": "Send"})
    folder = root / "modules" / "agent" / "i18n"
    folder.mkdir(parents=True)
    (folder / "ru.json").write_text(
        json.dumps({"module.agent.hi": "Привет", "chat.send": "Чужой", "module.x.y": "z"}), "utf-8"
    )
    (folder / "en.json").write_text(json.dumps({"module.agent.hi": "Hi"}), "utf-8")
    errors = check_i18n.catalog_errors(root)
    assert "modules/agent: key 'module.x.y' is outside the module namespace" in errors
    assert "modules/agent: en has no translation for chat.send" in errors
    assert "modules/agent: key chat.send already exists in the core catalog" in errors


def test_nested_messages_render_and_serialize() -> None:
    ru, en = load_catalog("ru"), load_catalog("en")
    message = msg(
        "settings.invalid", field="theme", detail=msg("settings.bad_theme", allowed="a, b")
    )
    assert render(message, ru) == "Неверное значение поля theme: тема должна быть одной из: a, b"
    assert render(message, en).endswith("the theme must be one of: a, b")
    assert to_json({"x": [message]})["x"][0]["params"]["detail"] == {
        "code": "settings.bad_theme",
        "params": {"allowed": "a, b"},
    }
    assert render("нет.такого", ru) == "нет.такого"  # нет перевода — виден ключ
    error = UserError("download.paused")
    assert error.message == Msg("download.paused", {})


def test_both_languages_have_every_key() -> None:
    catalogs = {lang: load_catalog(lang) for lang in LANGUAGES}
    assert catalogs["ru"].keys() == catalogs["en"].keys()
    assert len(catalogs["ru"]) > 200


def section_manifest(module_id: str) -> Manifest:
    ui = {"section": {"view": "agent", "title_key": "section.agent", "icon": "agent"}}
    return Manifest(module_id, "1.0.0", "ramvi.tests.module_fixture:create", ui=ui)


@pytest.fixture
def registry(tmp_path: Path):  # type: ignore[no-untyped-def]
    host = ModuleHost(tmp_path)
    try:
        yield SectionRegistry(host)
    finally:
        host.close()


def test_module_registers_own_translations(registry: SectionRegistry) -> None:
    texts = {
        "ru": {"module.helper.greeting": "Привет, {name}"},
        "en": {"module.helper.greeting": "Hello, {name}"},
    }
    registry.install(section_manifest("helper"), texts=texts)
    assert registry.texts()["en"] == {"module.helper.greeting": "Hello, {name}"}
    registry.remove("helper")
    assert registry.texts() == {"ru": {}, "en": {}}


@pytest.mark.parametrize(
    "texts",
    [
        {"ru": {"module.helper.a": "x"}},  # нет английского
        {"ru": {"module.helper.a": "{n}"}, "en": {"module.helper.a": "{m}"}},  # разные параметры
        {"ru": {"module.other.a": "x"}, "en": {"module.other.a": "y"}},  # чужое пространство
        {"ru": {"chat.send": "x"}, "en": {"chat.send": "y"}},  # ключ ядра
    ],
)
def test_bad_module_translations_block_install(
    registry: SectionRegistry, texts: dict[str, dict[str, str]]
) -> None:
    with pytest.raises(ValueError, match="module helper"):
        registry.install(section_manifest("helper"), texts=texts)
    assert "helper" not in registry.host.status()


def write_module(root: Path, module_id: str, texts: dict[str, Any]) -> None:
    folder = root / "modules" / module_id
    (folder / "i18n").mkdir(parents=True)
    manifest = {
        "module_id": module_id,
        "version": "1.0.0",
        "entrypoint": "ramvi.tests.module_fixture:create",
        "ui": {"section": {"view": "agent", "title_key": "section.agent", "icon": "agent"}},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    for lang, catalog in texts.items():
        (folder / "i18n" / f"{lang}.json").write_text(
            json.dumps(catalog, ensure_ascii=False), "utf-8"
        )


def test_service_serves_module_translations_and_rejects_broken(tmp_path: Path) -> None:
    app = tmp_path / "app"
    write_module(
        app, "helper", {"ru": {"module.helper.hi": "Привет"}, "en": {"module.helper.hi": "Hi"}}
    )
    write_module(app, "broken", {"ru": {"module.broken.hi": "Привет"}})
    server, engines = build(tmp_path / "data", TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        client = Client(server.port)
        assert client.ok("GET", "/i18n") == {
            "modules": {"ru": {"module.helper.hi": "Привет"}, "en": {"module.helper.hi": "Hi"}}
        }
        assert server.sections is not None
        problems = server.sections.problems
        assert [Path(p["path"]).parent.name for p in problems] == ["broken"]
        assert (
            "en has no translation" in problems[0]["error"] or "no catalog" in problems[0]["error"]
        )
        # Вложенная причина ошибки настроек приходит ключом, а не готовой фразой.
        status, error = client.request("POST", "/settings", {"theme": "pink"})
        assert status == 400 and error["error"]["code"] == "settings.invalid"
        assert error["error"]["params"]["detail"]["code"] == "settings.bad_theme"
    finally:
        server.close()
        engines.stop()
