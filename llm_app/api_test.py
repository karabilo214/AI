"""Проверка API сервера официальными клиентами OpenAI и Anthropic и запросами Ollama:
python -m llm_app.api_test http://127.0.0.1:11435 [ключ]"""
import json
import sys
import urllib.error
import urllib.request

import anthropic
import openai

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11435"
KEY = sys.argv[2] if len(sys.argv) > 2 else "none"
ok_all = []


def check(name, cond, info=""):
    ok_all.append(bool(cond))
    print(f"[{'OK' if cond else 'ОШИБКА'}] {name}" + (f" — {info}" if info else ""), flush=True)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                           "required": ["city"]}}}

# --- ключ доступа ---
if KEY != "none":
    try:
        urllib.request.urlopen(urllib.request.Request(URL + "/v1/models"), timeout=30)
        check("без ключа — отказ", False)
    except urllib.error.HTTPError as e:
        check("без ключа — отказ 401", e.code == 401)

oc = openai.OpenAI(base_url=URL + "/v1", api_key=KEY)
models = oc.models.list()
model = models.data[0].id
check("OpenAI: список моделей", len(models.data) == 1, model)

r = oc.chat.completions.create(model=model, messages=[{"role": "user", "content": "What is the capital of France? One word."}],
                               max_tokens=20, temperature=0)
check("OpenAI: обычный ответ", "Paris" in (r.choices[0].message.content or ""), repr(r.choices[0].message.content)[:60] +
      f", слов {r.usage.completion_tokens}")

parts = []
for ch in oc.chat.completions.create(model=model, messages=[{"role": "user", "content": "Count from 1 to 5, comma separated."}],
                                     max_tokens=30, temperature=0, stream=True, stream_options={"include_usage": True}):
    if ch.choices and ch.choices[0].delta.content:
        parts.append(ch.choices[0].delta.content)
txt = "".join(parts)
check("OpenAI: потоковый ответ", "1" in txt and "5" in txt and len(parts) > 2, f"{len(parts)} кусков: {txt[:40]!r}")

msgs = [{"role": "user", "content": "What's the weather in Tokyo right now?"}]
r = oc.chat.completions.create(model=model, messages=msgs, tools=[WEATHER], max_tokens=200, temperature=0)
tc = r.choices[0].message.tool_calls
args = json.loads(tc[0].function.arguments) if tc else {}
check("OpenAI: вызов инструмента", bool(tc) and tc[0].function.name == "get_weather" and "tokyo" in args.get("city", "").lower(),
      f"finish={r.choices[0].finish_reason}, {tc[0].function.name if tc else None}({args})"
      + ("" if tc else f", ответ: {r.choices[0].message.content!r}"[:300]))
if tc:
    msgs += [r.choices[0].message.model_dump(exclude_none=True),
             {"role": "tool", "tool_call_id": tc[0].id, "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    r2 = oc.chat.completions.create(model=model, messages=msgs, tools=[WEATHER], max_tokens=80, temperature=0)
    a = r2.choices[0].message.content or ""
    check("OpenAI: ответ после результата инструмента", "18" in a, repr(a)[:80])

schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "hobbies": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "age", "hobbies"], "additionalProperties": False}
r = oc.chat.completions.create(model=model, messages=[{"role": "user", "content": "Invent a person: name, age and three hobbies."}],
                               response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema}},
                               max_tokens=120, temperature=0)
try:
    o = json.loads(r.choices[0].message.content)
    good = set(o) == {"name", "age", "hobbies"} and isinstance(o["age"], int) and all(isinstance(h, str) for h in o["hobbies"])
except Exception as e:
    o, good = str(e), False
check("OpenAI: JSON по схеме", good, str(o)[:90])

r = oc.chat.completions.create(model=model, messages=[{"role": "user", "content": "Give 2 fruits with colors as JSON."}],
                               response_format={"type": "json_object"}, max_tokens=300, temperature=0)
try:
    json.loads(r.choices[0].message.content)
    check("OpenAI: любой JSON", True, r.choices[0].message.content[:70])
except Exception as e:
    check("OpenAI: любой JSON", False, f"{e}: {r.choices[0].message.content[:70]!r}")

r = oc.chat.completions.create(model=model, messages=[{"role": "user", "content": "List the days of the week, one per line."}],
                               stop=["Wednesday"], max_tokens=60, temperature=0)
c = r.choices[0].message.content or ""
check("OpenAI: стоп-слово", "Wednesday" not in c and "Tuesday" in c, repr(c)[:60])

# --- Anthropic ---
ac = anthropic.Anthropic(base_url=URL, api_key=KEY)
m = ac.messages.create(model=model, max_tokens=30, extra_body={"temperature": 0}, system="Answer briefly.",
                       messages=[{"role": "user", "content": "What is 7*8?"}])
check("Anthropic: обычный ответ", "56" in m.content[0].text, f"{m.content[0].text!r}, stop={m.stop_reason}")

with ac.messages.stream(model=model, max_tokens=40, extra_body={"temperature": 0}, messages=[{"role": "user", "content": "Name three colors."}]) as s:
    txt = "".join(s.text_stream)
    fm = s.get_final_message()
check("Anthropic: потоковый ответ", len(txt) > 5 and fm.stop_reason in ("end_turn", "max_tokens"), f"{txt[:50]!r}")

atool = {"name": "get_weather", "description": "Get current weather for a city",
         "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
amsgs = [{"role": "user", "content": "What's the weather in Paris?"}]
m = ac.messages.create(model=model, max_tokens=200, extra_body={"temperature": 0}, tools=[atool], messages=amsgs)
tu = [b for b in m.content if b.type == "tool_use"]
check("Anthropic: tool_use", bool(tu) and tu[0].name == "get_weather" and m.stop_reason == "tool_use",
      f"stop={m.stop_reason}, {tu[0].input if tu else None}")
if tu:
    amsgs += [{"role": "assistant", "content": [b.model_dump() for b in m.content]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0].id, "content": "21 C, sunny"}]}]
    m2 = ac.messages.create(model=model, max_tokens=80, extra_body={"temperature": 0}, tools=[atool], messages=amsgs)
    t2 = "".join(b.text for b in m2.content if b.type == "text")
    check("Anthropic: ответ после tool_result", "21" in t2, repr(t2)[:80])

# --- Ollama ---
body = json.dumps({"model": model, "stream": False, "format": schema, "options": {"temperature": 0, "num_predict": 120},
                   "messages": [{"role": "user", "content": "Invent a person: name, age and three hobbies."}]}).encode()
req = urllib.request.Request(URL + "/api/chat", body, {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
o = json.loads(urllib.request.urlopen(req, timeout=600).read())
try:
    p = json.loads(o["message"]["content"])
    check("Ollama: format по схеме", set(p) == {"name", "age", "hobbies"}, str(p)[:80])
except Exception as e:
    check("Ollama: format по схеме", False, str(e))

print(f"ИТОГ: {sum(ok_all)}/{len(ok_all)}")
