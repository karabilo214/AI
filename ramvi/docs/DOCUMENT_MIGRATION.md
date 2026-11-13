# Перенос документов (пункт 1.2)

Предварительное состояние движка и ТЗ сохранено коммитом `29aa6da` и отправлено в origin/main. После этого выполнен пункт 1.2: ТЗ и отчёты перенесены в каталоги RAMVI. Код движка не перемещался.

PDF и DOCX перенесены побайтно, без изменения содержимого, рисунков или метаданных. В Markdown изменены только относительные ссылки между документами. Для отслеживаемых файлов использован git mv. Отчёт WIDE30 скопирован из архива измерений; исходный архив сохранён. JSON, логи и экспериментальные бинарники в продукт не добавлялись.

Пути и команды llm_app в исторических отчётах описывают исходные опыты. Обновление рабочих команд выполняется при переносе кода и проверочных инструментов по следующим пунктам плана.

Хеши источников относятся к состоянию `29aa6da`. Хеши назначения — к содержимому после исправления ссылок.

| Исходный путь | Путь внутри RAMVI | SHA256 источника | SHA256 назначения |
|---|---|---|---|
| `llm_app/docs/RAMVI_technical_spec_v1.docx` | `docs/spec/RAMVI_technical_spec_v1.docx` | `082793176564503e29cc230641056adcfa057f8b18e7a59a675383a03cdb69a1` | `082793176564503e29cc230641056adcfa057f8b18e7a59a675383a03cdb69a1` |
| `llm_app/docs/RAMVI_technical_spec_v1.pdf` | `docs/spec/RAMVI_technical_spec_v1.pdf` | `0bf28046b91e223a97ee415b8cfcaab5806ea41f6be7b4df9105c738c4a3c2fc` | `0bf28046b91e223a97ee415b8cfcaab5806ea41f6be7b4df9105c738c4a3c2fc` |
| `llm_app/docs/CODE_AUDIT_REPORT.md` | `docs/reports/CODE_AUDIT_REPORT.md` | `78b0ba4477bbe9f7b54a99c9fb8fb6d7e9555bbaf2fd970d79747b981763469c` | `78b0ba4477bbe9f7b54a99c9fb8fb6d7e9555bbaf2fd970d79747b981763469c` |
| `llm_app/docs/AUDIT_FIXES.md` | `docs/reports/AUDIT_FIXES.md` | `da4ad3cfe2ab7c78d0e12e9bc2433df91ef5d7b3f485222dff00eb82abc54821` | `da4ad3cfe2ab7c78d0e12e9bc2433df91ef5d7b3f485222dff00eb82abc54821` |
| `llm_app/docs/VERIFICATION_REPORT.md` | `docs/reports/VERIFICATION_REPORT.md` | `fdc30288dd8cb3604dc915861d08d442330b668ffb9bbdd85d2d3848612e4b20` | `fdc30288dd8cb3604dc915861d08d442330b668ffb9bbdd85d2d3848612e4b20` |
| `llm_app/docs/O02_OPTIMIZATION_REPORT.md` | `docs/reports/O02_OPTIMIZATION_REPORT.md` | `612a48d33b494c3cb216e3e4516abb9268927d64324f5502172c70acf0cb3225` | `1737b7d7ecc6c940b0f8e4f1124721754405fc2226e9cc07ed18eefe3bc7569e` |
| `llm_app/docs/PERFORMANCE_IMPROVEMENTS_REPORT.md` | `docs/reports/PERFORMANCE_IMPROVEMENTS_REPORT.md` | `7a784c1bb9df4f740085ae51e32711f7f4e8d27d4d9e9ed7ff146639cfc78209` | `199725f7eaf852453130d8927008a4740afe9cdbc6661fba1e3467b7e494e4e4` |
| `llm_app/docs/MEASURED_PLACEMENT.md` | `docs/architecture/MEASURED_PLACEMENT.md` | `8f3124914f156afd0b059bd0577ef41c2c27c2b7ba9a0ea55e047bfb509668e2` | `8f3124914f156afd0b059bd0577ef41c2c27c2b7ba9a0ea55e047bfb509668e2` |
| `llm_app/results/wide_current/WIDE30_OLLAMA_COMPARISON.md` | `docs/reports/WIDE30_OLLAMA_COMPARISON.md` | `ea89bf453dbe2defde6eec912dffb5a2b08f279afd9545e3aa39f729052699f1` | `ea89bf453dbe2defde6eec912dffb5a2b08f279afd9545e3aa39f729052699f1` |
