"""Отчёт полного WIDE30: перенос RAMVI и исторический результат Ollama."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import cast

RUN_FILES = {
    "before_r1": "wide_before_r1.json",
    "after_r1": "wide_after_r1.json",
    "after_r2": "wide_after_r2.json",
    "before_r2": "wide_before_r2.json",
}
EXPECTED_PARAMETERS: dict[str, object] = {
    "temperature": 0,
    "top_k": 1,
    "seed": 0,
    "num_ctx": 2048,
    "think": False,
    "num_predict": 200,
}


@dataclass(frozen=True)
class Row:
    """Проверенные счётчики одного HTTP-ответа и манифест его запуска."""

    category: str
    question: str
    output: str
    tokens: int
    generation_s: float
    tps: float
    ttft_s: float
    wall_s: float
    run: dict[str, object] | None


def object_dict(value: object, label: str) -> dict[str, object]:
    """Отвергнуть JSON-значение, не являющееся объектом со строковыми ключами."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label}: ожидался объект JSON")
    return cast(dict[str, object], value)


def text_field(value: object, label: str) -> str:
    """Принять только непустую строку."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: нужна непустая строка")
    return value


def number_field(value: object, label: str, *, positive: bool = True) -> float:
    """Принять конечный числовой счётчик, не включая bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: ожидалось число")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{label}: неверный числовой счётчик")
    return result


def load_rows(path: Path, *, require_run: bool) -> list[Row]:
    """Прочитать строго полный ряд WIDE30, проверив арифметику скорости."""
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError(f"{path.name}: нужны все 30 вопросов")
    rows: list[Row] = []
    for index, value in enumerate(data, 1):
        record = object_dict(value, f"{path.name}:{index}")
        tokens = record.get("n")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"{path.name}:{index}: неверный eval_count")
        seconds = number_field(record.get("gen_s"), "gen_s")
        tps = number_field(record.get("tps"), "tps")
        if not math.isclose(tps, tokens / seconds, rel_tol=1e-6):
            raise ValueError(f"{path.name}:{index}: tps не равен n/gen_s")
        run = object_dict(record.get("run"), "run") if require_run else None
        rows.append(
            Row(
                text_field(record.get("cat"), "cat"),
                text_field(record.get("q"), "q"),
                text_field(record.get("out"), "out"),
                tokens,
                seconds,
                tps,
                number_field(record.get("ttft"), "ttft", positive=False),
                number_field(record.get("wall"), "wall"),
                run,
            )
        )
    if len({row.question for row in rows}) != 30:
        raise ValueError(f"{path.name}: вопросы должны быть различными")
    return rows


def validate_runs(runs: dict[str, list[Row]], ollama: list[Row]) -> None:
    """Проверить одинаковые задачи, веса, токенизатор, размещение и параметры."""
    reference_run = runs["before_r1"][0].run
    assert reference_run is not None
    reference_server = object_dict(reference_run.get("server"), "server")
    for field in ("weights_id", "tokenizer"):
        text_field(reference_server.get(field), field)
    placement = reference_server.get("place")
    if not isinstance(placement, list) or not placement:
        raise ValueError("server.place: нужно фактическое размещение слоёв")
    if reference_run.get("model") != "qwen3:14b":
        raise ValueError("ожидалась модель qwen3:14b")
    stable_code: dict[str, str] = {}
    for name, rows in runs.items():
        for index, row in enumerate(rows):
            old = ollama[index]
            if (row.question, row.category) != (old.question, old.category):
                raise ValueError(f"{name}: порядок, вопрос или категория отличаются от Ollama")
            run = row.run
            assert run is not None
            if run.get("version") != 1 or run.get("model") != reference_run.get("model"):
                raise ValueError(f"{name}: неверная версия манифеста или другая модель")
            if run.get("parameters") != EXPECTED_PARAMETERS:
                raise ValueError(f"{name}: параметры генерации отличаются от WIDE30")
            server = object_dict(run.get("server"), "server")
            for field in ("weights_id", "tokenizer", "place", "maxt", "draft", "k", "force_greedy"):
                if field not in server or server[field] != reference_server.get(field):
                    raise ValueError(f"{name}: отличается server.{field}")
            if server["maxt"] != 2048 or server["draft"] is not True:
                raise ValueError(f"{name}: нужен контекст 2048 и включённый угадыватель")
            code = text_field(server.get("code_sha256"), "code_sha256")
            side = name.split("_")[0]
            if stable_code.setdefault(side, code) != code:
                raise ValueError(f"{name}: исходники менялись между вопросами или повторами")


def pooled_speed(rows: list[Row]) -> float:
    """Скорость по суммарным токенам и времени, без среднего отдельных скоростей."""
    return sum(row.tokens for row in rows) / sum(row.generation_s for row in rows)


def metrics(rows: list[Row]) -> dict[str, float | int]:
    """Сводные измеренные счётчики ряда или объединённых рядов."""
    return {
        "tps": pooled_speed(rows),
        "ttft_s": statistics.median(row.ttft_s for row in rows),
        "wall_s": sum(row.wall_s for row in rows),
        "tokens": sum(row.tokens for row in rows),
        "generation_s": sum(row.generation_s for row in rows),
        "answers": len(rows),
    }


def count_ratios(ratios: list[float]) -> dict[str, int]:
    """Считать выигрыш, проигрыш и точное равенство отдельно."""
    return {
        "faster": sum(value > 1 for value in ratios),
        "slower": sum(value < 1 for value in ratios),
        "equal": sum(value == 1 for value in ratios),
    }


def comparison(
    runs: dict[str, list[Row]], ollama: list[Row]
) -> tuple[dict[str, object], list[dict[str, str | float | int | bool]]]:
    """Построить сводку, сравнение ответов всех шести пар и таблицу вопросов."""
    validate_runs(runs, ollama)
    before = runs["before_r1"] + runs["before_r2"]
    after = runs["after_r1"] + runs["after_r2"]
    table: list[dict[str, str | float | int | bool]] = []
    for index, old in enumerate(ollama):
        first = [runs[f"before_r{r}"][index] for r in (1, 2)]
        second = [runs[f"after_r{r}"][index] for r in (1, 2)]
        before_tps, after_tps = pooled_speed(first), pooled_speed(second)
        table.append(
            {
                "number": index + 1,
                "category": old.category,
                "question": old.question,
                "ollama_tps": old.tps,
                "before_r1_tps": first[0].tps,
                "after_r1_tps": second[0].tps,
                "after_r2_tps": second[1].tps,
                "before_r2_tps": first[1].tps,
                "before_tps": before_tps,
                "after_tps": after_tps,
                "after_before_ratio": after_tps / before_tps,
                "after_ollama_ratio": after_tps / old.tps,
                "pair1_ratio": second[0].tps / first[0].tps,
                "pair2_ratio": second[1].tps / first[1].tps,
                "ollama_ttft_s": old.ttft_s,
                "before_ttft_s": statistics.median(row.ttft_s for row in first),
                "after_ttft_s": statistics.median(row.ttft_s for row in second),
                "all_four_texts_same": len({row.output for row in first + second}) == 1,
            }
        )
    text_pairs = {
        f"{a}__{b}": sum(x.output == y.output for x, y in zip(runs[a], runs[b], strict=True))
        for a, b in itertools.combinations(RUN_FILES, 2)
    }
    categories: dict[str, object] = {}
    for category in dict.fromkeys(row.category for row in ollama):
        indices = [i for i, row in enumerate(ollama) if row.category == category]
        before_cat = [runs[f"before_r{r}"][i] for r in (1, 2) for i in indices]
        after_cat = [runs[f"after_r{r}"][i] for r in (1, 2) for i in indices]
        ollama_cat = [ollama[i] for i in indices]
        ratios = [float(table[i]["after_ollama_ratio"]) for i in indices]
        categories[category] = {
            "questions": len(indices),
            "before_tps": pooled_speed(before_cat),
            "after_tps": pooled_speed(after_cat),
            "ollama_tps": pooled_speed(ollama_cat),
            "after_before_ratio": pooled_speed(after_cat) / pooled_speed(before_cat),
            "median_after_ollama_ratio": statistics.median(ratios),
            "versus_ollama": count_ratios(ratios),
        }
    ollama_ratios = [float(row["after_ollama_ratio"]) for row in table]
    migration_ratios = [float(row["after_before_ratio"]) for row in table]
    summary: dict[str, object] = {
        "order": list(RUN_FILES),
        "questions": 30,
        "ollama": metrics(ollama),
        "before": metrics(before),
        "after": metrics(after),
        "runs": {name: metrics(rows) for name, rows in runs.items()},
        "after_before_ratio": pooled_speed(after) / pooled_speed(before),
        "after_ollama_ratio": pooled_speed(after) / pooled_speed(ollama),
        "median_question_after_before_ratio": statistics.median(migration_ratios),
        "median_question_after_ollama_ratio": statistics.median(ollama_ratios),
        "versus_ollama": count_ratios(ollama_ratios),
        "versus_before": count_ratios(migration_ratios),
        "pair_tps_ratios": [
            pooled_speed(runs[f"after_r{r}"]) / pooled_speed(runs[f"before_r{r}"]) for r in (1, 2)
        ],
        "all_four_texts_same": sum(bool(row["all_four_texts_same"]) for row in table),
        "text_pairs": text_pairs,
        "categories": categories,
        "manifests": {name: rows[0].run for name, rows in runs.items()},
    }
    return summary, table


def markdown(summary: dict[str, object], table: list[dict[str, str | float | int | bool]]) -> str:
    """Сформировать русский отчёт с явными границами исторического сравнения."""
    lines = [
        "# RAMVI: WIDE30 до и после переноса",
        "",
        "Qwen3-14B + угадыватель Qwen3-0.6B; 30 вопросов, до 200 токенов, "
        "greedy, seed 0, контекст 2048, think=False. Четыре полных ряда в порядке "
        "до → после → после → до (ABBA).",
        "",
        "| Вариант | Токенов/с | Первое слово, медиана, с | Время HTTP, с | Ответов |",
        "|---|---:|---:|---:|---:|",
    ]
    run_metrics = object_dict(summary["runs"], "runs")
    for name, value in [("Ollama, сохранённый замер", summary["ollama"]), *run_metrics.items()]:
        item = object_dict(value, "metrics")
        lines.append(
            f"| {name} | {float(cast(float, item['tps'])):.3f} | "
            f"{float(cast(float, item['ttft_s'])):.3f} | "
            f"{float(cast(float, item['wall_s'])):.1f} | {item['answers']} |"
        )
    for name, key in [("До, оба ряда вместе", "before"), ("После, оба ряда вместе", "after")]:
        item = object_dict(summary[key], key)
        lines.append(
            f"| {name} | {float(cast(float, item['tps'])):.3f} | "
            f"{float(cast(float, item['ttft_s'])):.3f} | "
            f"{float(cast(float, item['wall_s'])):.1f} (60 ответов) | 60 |"
        )
    ratio = float(cast(float, summary["after_before_ratio"]))
    versus_ollama = object_dict(summary["versus_ollama"], "versus_ollama")
    pair_ratios = cast(list[float], summary["pair_tps_ratios"])
    lines.extend(
        [
            "",
            f"После / до по суммарным токенам и времени: **{ratio:.4f}×** "
            f"({(ratio - 1) * 100:+.2f}%). Отдельные пары повторов: "
            f"{pair_ratios[0]:.4f}× и {pair_ratios[1]:.4f}×.",
            f"После / сохранённая Ollama: **{float(cast(float, summary['after_ollama_ratio'])):.4f}×**. "
            f"Быстрее на {versus_ollama['faster']}/30 вопросах, медленнее на "
            f"{versus_ollama['slower']}/30, равенство на {versus_ollama['equal']}/30.",
            f"Тексты во всех четырёх рядах совпали на **{summary['all_four_texts_same']}/30** вопросах. "
            "Совпадение текста проверяет сохранение поведения, а не истинность всех ответов.",
            "",
            "## Сравнение текстов по каждой паре",
            "",
            "| Пара | Совпало |",
            "|---|---:|",
        ]
    )
    for pair, same in object_dict(summary["text_pairs"], "text_pairs").items():
        lines.append(f"| {pair} | {same}/30 |")
    lines.extend(
        [
            "",
            "## Все 30 вопросов",
            "",
            "| № | Тип и вопрос | Ollama, ток/с | До 1 / до 2 | После 1 / после 2 | После / до | После / Ollama | Первое слово: Ollama / до / после, с |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in table:
        question = str(row["question"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['number']} | {row['category']}: {question} | "
            f"{float(row['ollama_tps']):.2f} | "
            f"{float(row['before_r1_tps']):.2f} / {float(row['before_r2_tps']):.2f} | "
            f"{float(row['after_r1_tps']):.2f} / {float(row['after_r2_tps']):.2f} | "
            f"{float(row['after_before_ratio']):.4f}× | "
            f"{float(row['after_ollama_ratio']):.3f}× | "
            f"{float(row['ollama_ttft_s']):.3f} / {float(row['before_ttft_s']):.3f} / "
            f"{float(row['after_ttft_s']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## По типам вопросов",
            "",
            "| Тип | Вопросов | До, ток/с | После, ток/с | После / до | Медиана после / Ollama |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, value in object_dict(summary["categories"], "categories").items():
        item = object_dict(value, "category")
        lines.append(
            f"| {category} | {item['questions']} | "
            f"{float(cast(float, item['before_tps'])):.3f} | "
            f"{float(cast(float, item['after_tps'])):.3f} | "
            f"{float(cast(float, item['after_before_ratio'])):.4f}× | "
            f"{float(cast(float, item['median_after_ollama_ratio'])):.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Методика и ограничения",
            "",
            "- Скорость — сумма eval_count / сумма eval_duration. Для сравнения по одному вопросу "
            "также объединяются оба повтора; среднее скоростей не используется.",
            "- TTFT — время первого непустого фрагмента по часам HTTP-клиента. Загрузка модели, "
            "прогрев и отдельные контроли не входят в измеряемые 30 запросов.",
            "- Перед сравнением проверены одинаковые задачи, параметры, веса, токенизатор и "
            "размещение слоёв. Внутри каждого варианта hash кода одинаков во всех рядах.",
            "- Ollama не запускалась; взят исторический JSON на тех же 30 вопросах. "
            "Это не одновременный A/B-замер с Ollama; её архив не содержит полного манифеста.",
            "- Как в прежнем сравнении, eval_count нашего API исключает первый токен; "
            "архив Ollama не выделяет его отдельно. Историческая методика сохранена.",
            "- Два повтора показывают наблюдаемый разброс; сами по себе они не доказывают "
            "статистическую значимость изменения скорости. Итог приёмки требует также "
            "контролей plain/draft, fp32 и проверки завершения процессов из журнала запуска.",
            "- Совпадение текстов с Ollama не является критерием переноса: эталон поведения — "
            "наш движок до переноса. Сравнение текстов всех шести пар приведено выше.",
            "",
            "Полные ответы остаются во входных JSON. Машиночитаемые итоги: `summary.json`, "
            "таблица: `comparison.csv`. SHA256 входных файлов и манифесты — в `summary.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_report(results: Path, ollama_path: Path, output: Path) -> dict[str, object]:
    """Проверить исходные файлы и записать Markdown, CSV и JSON без изменения входов."""
    runs = {
        name: load_rows(results / filename, require_run=True)
        for name, filename in RUN_FILES.items()
    }
    ollama = load_rows(ollama_path, require_run=False)
    summary, table = comparison(runs, ollama)
    paths = [ollama_path, *(results / name for name in RUN_FILES.values())]
    summary["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    report = markdown(summary, table)
    output.mkdir(parents=True, exist_ok=True)
    (output / "WIDE30_MIGRATION_COMPARISON.md").write_text(report, encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return summary


def main() -> None:
    """Запустить генератор по явно заданным путям результатов и архива Ollama."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--ollama", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = generate_report(args.results, args.ollama, args.out)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "questions",
                    "after_before_ratio",
                    "after_ollama_ratio",
                    "all_four_texts_same",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
