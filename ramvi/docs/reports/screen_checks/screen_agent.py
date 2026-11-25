"""9.36: прототип — помощник управляет программой на Mac по снимку окна.

Цикл: снимок выбранного окна → модель с картинкой (подписка ChatGPT через Codex CLI) решает
одно действие → мышь или клавиатура только внутри окна (`act`, Swift) → новый снимок.
Модель сама решает, когда цель достигнута («done») или нужно спросить («ask»).
Каждый шаг (снимок, ответ модели, действие, время) пишется в отчёт.

usage: screen_agent.py <act> <приложение[|имя окна]> <цель> <папка отчёта> [шагов, по умолчанию 6]
(имя владельца окна бывает локализованным: «Calculator|Калькулятор»)
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROMPT = """You control a desktop application on macOS for the user, like a careful human.
Goal: {goal}
Application: {app}. The attached image is a fresh screenshot of its window only,
{w}x{h} points; coordinates you give are in this image, origin top-left.
Previous steps (oldest first): {history}

Decide exactly ONE next action. Prefer keyboard shortcuts the application really has when
they are reliable; otherwise click precisely on the visible control. Never act outside the
window. "type" pastes text through the clipboard (for text fields); "key" presses keys
(shortcuts, or plain keys/digits for tools that read keystrokes); "g x 0.3 enter" presses
several in order. "wait" waits 10 seconds (loading). If the goal is already achieved in the screenshot, answer done. If you cannot tell
what to do or the screen is unexpected, answer ask.
Answer only with one JSON object:
{{"see": "<what matters on screen, short>", "action": "click|double_click|drag|type|key|scroll|wait|done|ask",
  "x": int, "y": int, "x2": int, "y2": int, "text": "<for type>", "keys": "<for key, e.g. cmd+s>",
  "dy": int, "say": "<one short sentence for the user>"}}"""


def run(*cmd: str, stdin: str | None = None, timeout: float = 180) -> str:
    done = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
    if done.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {done.stderr[-400:]}")
    return done.stdout


def window(act: str, app: str) -> tuple[int, int, int, int, int]:
    wid, x, y, w, h = (int(v) for v in run(act, "window", app).split())
    return wid, x, y, w, h


def ask_model(image: Path, prompt: str) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    out = run(
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-c",
        'approval_policy="never"',
        "-m",
        "gpt-6-sol",
        "-i",
        str(image),
        "-",
        stdin=prompt,
    )
    found = re.findall(r"\{.*\}", out, re.S)
    if not found:
        raise RuntimeError("no JSON in model answer: " + out[-300:])
    return json.loads(found[-1]), round(time.monotonic() - started, 1)


def point(
    decision: dict[str, Any], key_x: str, key_y: str, box: tuple[int, int, int, int]
) -> tuple[str, str]:
    """Точка из ответа модели (в окне) → координаты экрана; вне окна — отказ."""
    wx, wy, w, h = box
    px, py = int(decision[key_x]), int(decision[key_y])
    if not (0 <= px < w and 0 <= py < h):
        raise RuntimeError(f"point outside window: {px},{py}")
    return str(wx + px), str(wy + py)


def main() -> None:
    act, names, goal, folder = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    app, _, owner = names.partition("|")
    owner = owner or app
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    folder.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"app": app, "goal": goal, "steps": []}
    history: list[str] = []
    repeats: list[str] = []
    last: tuple[str, str] | None = None
    for number in range(1, steps + 1):
        run("osascript", "-e", f'tell application "{app}" to activate')
        time.sleep(0.6)
        wid, wx, wy, w, h = window(act, owner)
        image = folder / f"step{number}.png"
        # Область экрана окна, а не само окно: всплывающие панели и меню тоже видны.
        run("screencapture", "-x", f"-R{wx},{wy},{w},{h}", str(image))
        run("sips", "-z", str(h), str(w), str(image))  # пиксели Retina → точки окна
        prompt = PROMPT.format(
            goal=goal, app=app, w=w, h=h, history="; ".join(history[-6:]) or "none"
        )
        decision, seconds = ask_model(image, prompt)
        row = {"step": number, "window": [wx, wy, w, h], "model_s": seconds, **decision}
        report["steps"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        kind = decision.get("action")
        signature = json.dumps({k: decision.get(k) for k in ("action", "x", "y", "text", "keys")})
        repeats.append(signature)
        if repeats.count(signature) >= 3:
            report["result"] = "stuck"  # одно и то же действие в третий раз — по кругу; стоп
            break
        if kind in ("done", "ask"):
            report["result"] = kind
            break

        box = (wx, wy, w, h)

        def at(
            key_x: str, key_y: str, d: dict[str, Any] = decision, b: tuple[int, ...] = box
        ) -> tuple[str, str]:
            return point(d, key_x, key_y, (b[0], b[1], b[2], b[3]))

        if kind in ("click", "double_click", "scroll"):
            last = at("x", "y")
        if kind == "click":
            run(act, "click", *at("x", "y"))
        elif kind == "double_click":
            run(act, "click", *at("x", "y"), "double")
        elif kind == "drag":
            run(act, "drag", *at("x", "y"), *at("x2", "y2"))
        elif kind == "type":
            run(act, "type", str(decision.get("text", "")))
        elif kind == "key":
            # Клавиши часть программ (Blender) отдаёт области под мышью: курсор — на последнюю
            # точку действия или в середину окна.
            run(act, "move", *(last or (str(wx + w // 2), str(wy + h // 2))))
            # «g x 0.3 enter» — несколько нажатий по очереди
            for token in str(decision.get("keys", "")).split():
                try:
                    run(act, "key", token)
                except RuntimeError:
                    history.append(f"key {token!r} is not supported, nothing pressed")
                    break
        elif kind == "wait":
            time.sleep(10)
        elif kind == "scroll":
            run(act, "scroll", *at("x", "y"), str(int(decision.get("dy", 0))))
        else:
            report["result"] = f"unknown action {kind}"
            break
        history.append(
            f"{kind} {json.dumps({k: decision.get(k) for k in ('x', 'y', 'x2', 'y2', 'text', 'keys') if decision.get(k) not in (None, '')}, ensure_ascii=False)} ({decision.get('say', '')})"
        )
        time.sleep(0.8)
    else:
        report["result"] = "step limit"
    final = folder / "final.png"
    wid, *_ = window(act, owner)
    run("screencapture", "-x", "-o", f"-l{wid}", str(final))
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print("result:", report.get("result"))


if __name__ == "__main__":
    main()
