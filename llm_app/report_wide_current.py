"""Русский отчёт по завершённым WIDE30 и сохранённой Ollama."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

from .bench_validation import paired_rows,validate_speed,compare_run


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',default='llm_app/results/wide_current')
    args=ap.parse_args();root=Path(args.dir)
    baseline_path=Path('llm_app/results/wide_ollama_14b.json')
    previous_path=Path('llm_app/results/wide_ours_14b_fixed3.json')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    baseline=load(baseline_path);previous=load(previous_path)
    paths=sorted(root.glob('wide_current_14b_r*.json'))
    if len(paths)<2:raise ValueError('нужны два полных прогона')
    runs=[load(p) for p in paths]
    for rows in [baseline,previous,*runs]:
        if len(rows)!=30:raise ValueError('ожидались все 30 вопросов')
        paired_rows(rows,baseline,'q');validate_speed(rows)
    for rows in runs:
        for a,b in zip(rows,previous):compare_run(a['run'],b['run'])
    allrows=[v for rows in runs for v in rows]
    speed=lambda rows:sum(x['n'] for x in rows)/sum(x['gen_s'] for x in rows)
    base_speed=speed(baseline);old_speed=speed(previous);new_speed=speed(allrows)
    table=[]
    for i,old in enumerate(baseline):
        samples=[r[i] for r in runs]
        current=st.median(x['tps'] for x in samples)
        row=dict(number=i+1,category=old['cat'],question=old['q'],ollama_tps=old['tps'],
                 current_tps=current,ratio=current/old['tps'],ollama_ttft=old['ttft'],
                 current_ttft=st.median(x['ttft'] for x in samples),
                 repeat_same=all(x['out']==samples[0]['out'] for x in samples),
                 previous_same=all(x['out']==previous[i]['out'] for x in samples))
        row.update({f'run{j+1}_tps':r[i]['tps'] for j,r in enumerate(runs)})
        table.append(row)
    ratios=[x['ratio'] for x in table]
    worst=min(table,key=lambda x:x['ratio']);best=max(table,key=lambda x:x['ratio'])
    meta=load(root/'server_metadata.json');profile=load(root/'profile.json')
    controls=load(root/'controls.json')
    lines=['# 30 вопросов: текущий llm_app и Ollama','',
           f'Qwen3‑14B, угадыватель Qwen3‑0.6B. Два полных прогона по 30 вопросам, максимум 200 токенов на ответ. '
           'Температура 0, seed 0, контекст 2048, размышления выключены. Запросы и их порядок совпадают с сохранённым набором Ollama.','',
           '| Вариант | Скорость, токенов/с | Первое слово, медиана, с | Время 30 ответов, с |',
           '|---|---:|---:|---:|',
           f'| Ollama, сохранённый замер | {base_speed:.2f} | {st.median(x["ttft"] for x in baseline):.3f} | {sum(x["wall"] for x in baseline):.1f} |',
           f'| Наш движок до последних изменений | {old_speed:.2f} | {st.median(x["ttft"] for x in previous):.3f} | {sum(x["wall"] for x in previous):.1f} |']
    for i,rows in enumerate(runs):
        lines.append(f'| Наш текущий, прогон {i+1} | {speed(rows):.2f} | {st.median(x["ttft"] for x in rows):.3f} | {sum(x["wall"] for x in rows):.1f} |')
    lines += [f'| Наш текущий, оба прогона вместе | {new_speed:.2f} | {st.median(x["ttft"] for x in allrows):.3f} | {sum(x["wall"] for x in allrows)/len(runs):.1f} (среднее) |','',
       f'Текущая скорость относительно Ollama: **{new_speed/base_speed:.3f}×**; '
       f'относительно нашей предыдущей версии: **{new_speed/old_speed:.3f}×**. '
       f'Медиана отношения по вопросам — {st.median(ratios):.3f}×. '
       f'Не медленнее Ollama на {sum(v>=1 for v in ratios)}/30 вопросах.','',
       f'Худший случай: №{worst["number"]}, {worst["category"]}, {worst["ratio"]:.3f}×. '
       f'Лучший: №{best["number"]}, {best["category"]}, {best["ratio"]:.3f}×.','',
       '## По каждому вопросу','',
       '| № | Вопрос | Ollama, ток/с | Наш 1, ток/с | Наш 2, ток/с | Наш / Ollama | Первое слово: Ollama / наш, с |',
       '|---:|---|---:|---:|---:|---:|---:|']
    for row in table:
        q=row['question'].replace('|','\\|').replace('\n',' ')
        lines.append(f'| {row["number"]} | {q} | {row["ollama_tps"]:.2f} | {row["run1_tps"]:.2f} | '
                     f'{row["run2_tps"]:.2f} | {row["ratio"]:.3f}× | {row["ollama_ttft"]:.3f} / {row["current_ttft"]:.3f} |')
    lines += ['', '## По типам вопросов','', '| Тип | Медиана наш / Ollama |','|---|---:|']
    for cat in dict.fromkeys(x['category'] for x in table):
        lines.append(f'| {cat} | {st.median(x["ratio"] for x in table if x["category"]==cat):.3f}× |')
    lines += ['', '## Проверка ответов и условия','',
       f'- Тексты между двумя текущими прогонами совпали на {sum(x["repeat_same"] for x in table)}/30 вопросах.',
       f'- С нашей предыдущей версией тексты обоих прогонов совпали на {sum(x["previous_same"] for x in table)}/30 вопросах.',
       f'- До измерения проверены {len(controls)} контролей: ответы target без угадывателя и с ним совпали. '
       'Проверки с известным ответом: 2+2=4 и 137×29=3973.',
       '- Совпадение текста не является оценкой истинности всех ответов; это проверка сохранения поведения движка.',
       f'- Размещение: GPU {meta["place"].count("gpu")}, подкачка {meta["place"].count("stream")}, '
       f'гибрид {meta["place"].count("hyb")}, CPU-only {meta["place"].count("cpu")}.',
       f'- Prefill: {meta["prefill"]}, порог {meta["prefill_min_tokens"]} токенов. '
       'Эта проверка не заменяет отдельный тест длинных запросов.',
       '- Дополнительная калибровка измеренного планировщика упёрлась в RAM. '
       'Основной тест выполнен после перезапуска процесса, на штатном распределении слоёв; измеренные маршруты не включены.',
       '- Ollama не запускалась: использован сохранённый результат на том же наборе. Это историческое сравнение, '
       'а не одновременный замер двух серверов. Исторический JSON Ollama не содержит полного манифеста оборудования и версии.',
       '- Скорость вычислена как сумма eval_count / сумма eval_duration, а не среднее скоростей отдельных вопросов. '
       'Как и в предыдущем отчёте, счётчик нашего сервера исключает первый токен; архив Ollama не выделяет его отдельно.',
       '- Время первого слова измерено HTTP-клиентом. Загрузка модели, прогрев и контрольные запросы вынесены за измеряемый набор.',
       '- Во время обоих прогонов код и настройки не менялись. На каждый прогон сохранены все ответы и параметры.','',
       '## Файлы и воспроизводимость','',
       f'- Архив Ollama: `{baseline_path}`; SHA256 `{hashlib.sha256(baseline_path.read_bytes()).hexdigest()}`.',
       f'- Архив нашей предыдущей версии: `{previous_path}`.',
       f'- Веса: `{meta["weights_id"]}`; токенизатор `{meta["tokenizer"]}`.',
       f'- Манифест сервера: `server_metadata.json`; hash кода `{meta["code_sha256"]}`.',
       '- Полные ответы: `wide_current_14b_r1.json`, `wide_current_14b_r2.json`.',
       '- Таблица для обработки: `comparison.csv`; исходные контроли: `controls.json`.']
    length_file=root/'prompt_lengths.json'
    if length_file.exists():
        lengths=load(length_file)
        at=lines.index('## Файлы и воспроизводимость')
        lines[at:at]=[f'Длины полных запросов: {min(lengths)}–{max(lengths)} токена. Порог нового prefill достигают '
                      f'{sum(n>=256 for n in lengths)}/30: ускорение длинного prefill этим набором не измеряется.','']
    lines.insert(lines.index('## По каждому вопросу')-1,
        f'По текущему замеру скорость выше сохранённой Ollama на {(new_speed/base_speed-1)*100:.1f}%, '
        f'но ниже нашей предыдущей версии на {(1-new_speed/old_speed)*100:.1f}%. Последние изменения не подтвердили ускорение относительно прежнего нашего результата.')
    report=root/'WIDE30_OLLAMA_COMPARISON.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    summary=dict(ollama_tps=base_speed,previous_tps=old_speed,current_tps=new_speed,
                 run_tps=[speed(r) for r in runs],run_wall_s=[sum(x['wall'] for x in r) for r in runs],
                 ollama_wall_s=sum(x['wall'] for x in baseline),ollama_ttft=st.median(x['ttft'] for x in baseline),
                 current_ttft=st.median(x['ttft'] for x in allrows),ratio=new_speed/base_speed,
                 versus_previous=new_speed/old_speed,median_ratio=st.median(ratios),
                 wins=sum(v>=1 for v in ratios),worst=worst,best=best,
                 repeat_same=sum(x['repeat_same'] for x in table),previous_same=sum(x['previous_same'] for x in table))
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(report)

if __name__=='__main__':main()
