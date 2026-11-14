"""Контроли арифметики и отказа от несопоставимых данных отчёта миграции."""

import copy
import csv
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

report = importlib.import_module("tools.bench.report_migration")
generate_report = cast(Callable[[Path, Path, Path], dict[str, object]], report.generate_report)
RUN_FILES = (
    "wide_before_r1.json",
    "wide_after_r1.json",
    "wide_after_r2.json",
    "wide_before_r2.json",
)


def fixture_rows() -> list[dict[str, object]]:
    """Создать явно искусственные данные с ручной скоростью 20/2 = 10 ток/с."""
    return [
        {
            "cat": "math" if index < 15 else "code",
            "q": f"synthetic question {index}",
            "out": f"synthetic answer {index}",
            "n": 20,
            "gen_s": 2.0,
            "tps": 10.0,
            "ttft": 0.25,
            "wall": 2.5,
            "run": {
                "version": 1,
                "model": "qwen3:14b",
                "parameters": {
                    "temperature": 0,
                    "top_k": 1,
                    "seed": 0,
                    "num_ctx": 2048,
                    "think": False,
                    "num_predict": 200,
                },
                "server": {
                    "weights_id": "synthetic-weights",
                    "tokenizer": "synthetic-tokenizer",
                    "place": ["gpu", "stream", "hyb"],
                    "maxt": 2048,
                    "draft": True,
                    "k": None,
                    "force_greedy": False,
                    "code_sha256": "synthetic-code",
                },
            },
        }
        for index in range(30)
    ]


def save(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый ряд без потери unicode и конечных числовых значений."""
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Положить четыре одинаковых ряда и исторический JSON без манифеста."""
    results = tmp_path / "input"
    results.mkdir()
    rows = fixture_rows()
    for name in RUN_FILES:
        save(results / name, rows)
    ollama = tmp_path / "ollama.json"
    archived = copy.deepcopy(rows)
    for row in archived:
        del row["run"]
    save(ollama, archived)
    return results, ollama, tmp_path / "output"


def test_identical_rows_have_hand_calculated_ratio_one(tmp_path: Path) -> None:
    """Контроль: одинаковые измерения дают ровно 1,0 и 30/30 в каждой паре."""
    results, ollama, output = prepare(tmp_path)
    summary = generate_report(results, ollama, output)
    assert summary["after_before_ratio"] == 1.0
    assert summary["after_ollama_ratio"] == 1.0
    assert summary["pair_tps_ratios"] == [1.0, 1.0]
    assert summary["all_four_texts_same"] == 30
    assert summary["versus_ollama"] == {"faster": 0, "slower": 0, "equal": 30}
    pairs = cast(dict[str, int], summary["text_pairs"])
    assert len(pairs) == 6 and set(pairs.values()) == {30}
    before = cast(dict[str, float], summary["before"])
    assert before["tps"] == 10.0
    assert before["tokens"] == 1200
    assert before["generation_s"] == 120.0
    assert before["ttft_s"] == 0.25
    assert before["wall_s"] == 150.0
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary
    with (output / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 30
    text = (output / "WIDE30_MIGRATION_COMPARISON.md").read_text(encoding="utf-8")
    assert "ABBA" in text and "исторический" in text
    assert "**30/30**" in text and "1.0000×" in text


def test_pooled_speed_is_not_average_of_question_speeds(tmp_path: Path) -> None:
    """Контроль: 600/(15*1+15*4)=8, а не среднее (20+5)/2=12,5."""
    results, ollama, output = prepare(tmp_path)
    for name in ("wide_after_r1.json", "wide_after_r2.json"):
        rows = fixture_rows()
        for index, row in enumerate(rows):
            row["gen_s"] = 1.0 if index < 15 else 4.0
            row["tps"] = 20.0 if index < 15 else 5.0
        save(results / name, rows)
    summary = generate_report(results, ollama, output)
    after = cast(dict[str, float], summary["after"])
    assert after["tps"] == 8.0
    assert summary["after_before_ratio"] == 0.8
    assert summary["versus_ollama"] == {"faster": 15, "slower": 15, "equal": 0}


def test_text_difference_is_reported_without_hiding_other_rows(tmp_path: Path) -> None:
    """Поведение может не совпасть: отчёт обязан показать провал, а не отказать в записи."""
    results, ollama, output = prepare(tmp_path)
    rows = fixture_rows()
    rows[4]["out"] = "different synthetic answer"
    save(results / "wide_after_r2.json", rows)
    summary = generate_report(results, ollama, output)
    assert summary["all_four_texts_same"] == 29
    pairs = cast(dict[str, int], summary["text_pairs"])
    assert pairs["before_r1__after_r2"] == 29
    assert pairs["before_r1__after_r1"] == 30


@pytest.mark.parametrize("field", ["weights_id", "tokenizer", "place", "maxt", "draft", "k"])
def test_changed_execution_identity_is_rejected(tmp_path: Path, field: str) -> None:
    """Нельзя сравнивать перенос с другими весами, размещением или режимом генерации."""
    results, ollama, output = prepare(tmp_path)
    rows = fixture_rows()
    run = cast(dict[str, object], rows[0]["run"])
    server = cast(dict[str, object], run["server"])
    server[field] = "changed"
    save(results / "wide_after_r1.json", rows)
    with pytest.raises(ValueError, match="server"):
        generate_report(results, ollama, output)
    assert not output.exists()


def test_changed_generation_parameters_are_rejected(tmp_path: Path) -> None:
    """Лимит ответа и sampling не должны незаметно отличаться между рядами."""
    results, ollama, output = prepare(tmp_path)
    rows = fixture_rows()
    run = cast(dict[str, object], rows[0]["run"])
    parameters = cast(dict[str, object], run["parameters"])
    parameters["num_predict"] = 32
    save(results / "wide_after_r1.json", rows)
    with pytest.raises(ValueError, match="параметры"):
        generate_report(results, ollama, output)


def test_code_change_between_repeats_is_rejected(tmp_path: Path) -> None:
    """Один вариант должен исполнять одни и те же исходники в обоих повторах."""
    results, ollama, output = prepare(tmp_path)
    rows = fixture_rows()
    for row in rows:
        run = cast(dict[str, object], row["run"])
        server = cast(dict[str, object], run["server"])
        server["code_sha256"] = "different-code"
    save(results / "wide_after_r2.json", rows)
    with pytest.raises(ValueError, match="исходники менялись"):
        generate_report(results, ollama, output)


@pytest.mark.parametrize("mutation", ["short", "reordered", "empty", "bad_speed", "nan"])
def test_incomplete_or_invalid_measurements_are_rejected(tmp_path: Path, mutation: str) -> None:
    """Проверка не засчитывает неполный ряд, иной порядок или пустые/неверные метрики."""
    results, ollama, output = prepare(tmp_path)
    rows = fixture_rows()
    if mutation == "short":
        rows.pop()
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "empty":
        rows[0]["out"] = ""
    elif mutation == "bad_speed":
        rows[0]["tps"] = 9999
    else:
        rows[0]["ttft"] = float("nan")
    save(results / "wide_after_r1.json", rows)
    with pytest.raises(ValueError):
        generate_report(results, ollama, output)
    assert not output.exists()
