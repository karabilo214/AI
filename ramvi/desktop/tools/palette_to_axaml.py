"""Сгенерировать ресурсы темы Avalonia из общей палитры RAMVI (3.16).

Источник истины — `assets/branding/palette.json` (правило 7 PALETTE.md); этот файл
переводит его в `src/Ramvi.Desktop/Themes/Palette.axaml`, сохраняя имена токенов.
Тест `ThemeTests` падает, если файл разошёлся с палитрой: после правки палитры
запустить `python desktop/tools/palette_to_axaml.py`.
"""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PALETTE = ROOT / "assets" / "branding" / "palette.json"
TARGET = ROOT / "desktop" / "src" / "Ramvi.Desktop" / "Themes" / "Palette.axaml"
# 8.5 (UI-04): высокий контраст — своя тема (в оболочке ThemeVariant «HighContrast»,
# недостающие ресурсы Fluent берутся из Dark).
VARIANTS = {"light": "Light", "dark": "Dark", "high_contrast": "HighContrast"}


def shadows(theme: dict[str, str], name: str = "") -> dict[str, str]:
    """Неоморфизм: светлая тень сверху слева, тёмная снизу справа; inset — «вдавлено».

    8.5: в высоком контрасте тени на чёрном не видны — вместо них рамка цвета `border`
    (выпуклое — снаружи, вдавленное — внутри, толще), чтобы кнопки и поля были различимы.
    """
    if name == "high_contrast":
        line = theme["border"]
        return {
            "raised": f"0 0 0 1 {line}",
            "raised-sm": f"0 0 0 1 {line}",
            "inset": f"inset 0 0 0 2 {line}",
        }
    dark, light = theme["shadow-dark"], theme["shadow-light"]
    return {
        "raised": f"6 6 14 0 {dark}, -6 -6 14 0 {light}",
        "raised-sm": f"3 3 7 0 {dark}, -3 -3 7 0 {light}",
        "inset": f"inset 3 3 6 0 {dark}, inset -3 -3 6 0 {light}",
    }


def render(palette: dict[str, Any]) -> str:
    lines = [
        '<ResourceDictionary xmlns="https://github.com/avaloniaui"',
        '                    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"',
        '                    xmlns:app="using:Ramvi.Desktop">',
        "  <!-- Сгенерировано desktop/tools/palette_to_axaml.py из assets/branding/palette.json."
        " Не править вручную. -->",
        "  <ResourceDictionary.ThemeDictionaries>",
    ]
    for theme, variant in VARIANTS.items():
        # Свою тему Avalonia принимает только через x:Static (App.HighContrast).
        key = variant if variant in ("Light", "Dark") else f"{{x:Static app:App.{variant}}}"
        lines.append(f'    <ResourceDictionary x:Key="{key}">')
        for name, value in palette["themes"][theme].items():
            lines.append(f'      <Color x:Key="ramvi-{name}-color">{value}</Color>')
            lines.append(f'      <SolidColorBrush x:Key="ramvi-{name}" Color="{value}" />')
        for name, value in shadows(palette["themes"][theme], theme).items():
            lines.append(f'      <BoxShadows x:Key="ramvi-shadow-{name}">{value}</BoxShadows>')
        lines.append("    </ResourceDictionary>")
    lines.append("  </ResourceDictionary.ThemeDictionaries>")
    for name, value in palette["brand"].items():
        lines.append(f'  <Color x:Key="ramvi-brand-{name}">{value}</Color>')
    lines.append("</ResourceDictionary>")
    return "\n".join(lines) + "\n"


def main() -> int:
    palette = json.loads(PALETTE.read_text(encoding="utf-8"))
    if set(palette["themes"]) != set(VARIANTS):
        print(f"ожидались темы {sorted(VARIANTS)}", file=sys.stderr)
        return 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(render(palette), encoding="utf-8")
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
