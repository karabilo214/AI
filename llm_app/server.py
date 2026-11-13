"""HTTP-сервер для приложений и агентов: OpenAI (/v1/chat/completions, /v1/completions, /v1/models), Anthropic
(/v1/messages — Claude Code и клиенты Claude), Ollama (/api/chat, /api/generate, /api/tags). DeepSeek и Codex
ходят через формат OpenAI.

Возможности: потоковая выдача (SSE / ndjson), вызов инструментов (tools / tool_calls, tool_use / tool_result),
ответ строго в JSON или по JSON Schema (response_format / format — ограничение грамматикой, совместимо с
угадыванием), стоп-слова, рассуждения модели отдельно (reasoning_content / thinking), учёт слов (usage).
Сеть: --host 0.0.0.0, ключ --api-key (Authorization: Bearer ... или x-api-key), CORS, запросы — по очереди."""
import json
import queue
import hashlib
from pathlib import Path
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .request_validation import validate_request, validate_capabilities

TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def _text(content):
    """Содержимое сообщения (строка или список частей OpenAI/Anthropic) -> строка."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for p in content:
        if isinstance(p, str):
            out.append(p)
        elif p.get("type") in ("text", "input_text", "output_text"):
            out.append(p.get("text", ""))
    return "".join(out)


def _split(full, think):
    """Полный текст модели -> (рассуждение, ответ без вызовов инструментов, вызовы инструментов, хвост ответа
    может быть началом служебной метки)."""
    reasoning, content = "", full
    stripped = full.lstrip("\n")
    # Пока открывающая метка неполна, ничего из неё не отдаём клиенту.
    if stripped and "<think>".startswith(stripped):
        return "", "", [], True
    explicit = stripped.startswith("<think>")
    if think or explicit:
        start = full.find("<think>") + 7 if explicit else 0
        end = full.find("</think>", start)
        if end < 0:
            return full[start:].lstrip("\n"), "", [], True
        reasoning = full[start:end].lstrip("\n")
        content = full[end + 8:].lstrip("\n")
    else:
        # Не переопределяем уже выданный ответ задним числом как reasoning.
        content = content.replace("</think>", "")
    calls, pieces, last = [], [], 0
    for match in TOOL_RE.finditer(content):
        pieces.append(content[last:match.start()])
        try:
            obj = json.loads(match.group(1))
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
                raise ValueError("неверная форма вызова")
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("аргументы вызова должны быть объектом")
            calls.append({"name": obj["name"], "arguments": args})
        except (ValueError, TypeError):
            pieces.append(match.group(0))  # неправильный вывод модели остаётся обычным текстом
        last = match.end()
    tail = content[last:]
    opening = tail.find("<tool_call>")
    pieces.append(tail if opening < 0 else tail[:opening])
    return reasoning, "".join(pieces), calls, False


def _first_stop(content, stops):
    hits = [(content.find(stop), i, stop) for i, stop in enumerate(stops) if stop in content]
    return min(hits)[2] if hits else None


class App:
    def __init__(self, engine, tokenizer, name, draft=None, k=None, api_key=None, force_greedy=False):
        self.eng, self.tk, self.name, self.draft, self.k = engine, tokenizer, name, draft, k
        self.api_key, self.force_greedy = api_key, force_greedy
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(8)

    def metadata(self):
        engine = self.eng
        model_path = Path(engine.g.path) if hasattr(engine, "g") else None
        weights_id = model_path.name if model_path is not None and model_path.name.startswith("sha256-") else None
        if weights_id is None and hasattr(engine, "ck") and hasattr(engine.ck, "dir"):
            weights_id = Path(engine.ck.dir).name
        digest = hashlib.sha256()
        for file in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(file.name.encode()); digest.update(file.read_bytes())
        return {"weights_id": weights_id, "tokenizer": getattr(engine, "tokenizer_fingerprint", None),
                "maxt": getattr(engine, "maxt", None), "place": getattr(engine, "place", None),
                "code_sha256": digest.hexdigest(), "draft": self.draft is not None, "k": self.k,
                "force_greedy": self.force_greedy}

    # ---------------- подготовка запроса ----------------
    def build_ids(self, messages, tools=None, think=False):
        kw = {"tools": tools} if tools else {}
        text = self.tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think, **kw)
        return self.tk.encode(text, add_special_tokens=False)

    def chat_ids(self, messages, think=False):  # совместимость с прежним кодом
        return self.build_ids(messages, None, think)

    def constraint(self, schema_or_json):
        """schema_or_json: "json" — любой JSON; dict — JSON Schema; None — без ограничения."""
        if schema_or_json is None:
            return None
        from .constrain import Constraint
        V = getattr(self.eng, "V", None) or len(self.tk)
        return Constraint(self.tk, V, None if schema_or_json == "json" else schema_or_json)

    # ---------------- генерация ----------------
    QUEUE_SIZE = 8
    QUEUE_TIMEOUT = 10.0
    ENGINE_TIMEOUT = 60.0

    def run(self, *args, **kwargs):
        """Сетевая выдача отделена от вычислений ограниченной очередью."""
        events = queue.Queue(maxsize=self.QUEUE_SIZE)
        cancel, finished = threading.Event(), threading.Event()
        failures = []
        end = object()
        def put(value):
            deadline = time.monotonic() + self.QUEUE_TIMEOUT
            while not cancel.is_set():
                try:
                    events.put(value, timeout=0.01)
                    return True
                except queue.Full:
                    if time.monotonic() >= deadline:
                        failures.append(TimeoutError("потребитель не успевает читать ответ"))
                        cancel.set()
            return False
        def produce():
            gen = self._run(*args, **kwargs)
            try:
                for event in gen:
                    if not put(event):
                        break
            except Exception as exc:
                failures.append(exc)
            finally:
                try:
                    gen.close()
                except Exception as exc:
                    failures.append(exc)
                finally:
                    finished.set()
                    try:
                        events.put_nowait(end)
                    except queue.Full:
                        pass
        worker = threading.Thread(target=produce, name="llm-response", daemon=True)
        worker.start()
        last_event = time.monotonic()
        try:
            while True:
                if finished.is_set() and events.empty():
                    if failures:
                        raise failures[0]
                    break
                try:
                    item = events.get(timeout=0.05)
                except queue.Empty:
                    if finished.is_set():
                        if failures:
                            raise failures[0]
                        break
                    if time.monotonic() - last_event >= self.ENGINE_TIMEOUT:
                        raise TimeoutError("модель не ответила вовремя")
                    continue
                if item is end:
                    if failures:
                        raise failures[0]
                    break
                yield item
                last_event = time.monotonic()
        finally:
            cancel.set()
            worker.join(timeout=5)

    def _run(self, ids, max_tokens=512, temperature=0.0, top_p=1.0, seed=None, stop=None, think=False, cons=None):
        """Генератор событий: ("reasoning", кусок) | ("text", кусок) | ("end", итог)."""
        if self.force_greedy:
            temperature = 0.0
        stop = [s for s in (stop or []) if s]
        hold = max([len("<tool_call>")] + [len(s) for s in stop]) - 1
        st, gen, seen = {}, [], 0
        matched_stop = None
        sent_r, sent_c, finish = 0, 0, "length"
        with self.lock:
            t0 = time.perf_counter()
            for t in self.eng.generate(ids, max_tokens, temperature, top_p, seed,
                                       draft=self.draft if temperature <= 0 else None, k=self.k, stats=st, cons=cons):
                seen += 1
                if t in self.eng.ck.eos:
                    finish = "stop"
                    break
                gen.append(t)
                full = self.tk.decode(gen, skip_special_tokens=False)
                if full.endswith("�"):
                    continue
                r, c, calls, open_think = _split(full, think)
                hit = _first_stop(c, stop)
                if hit is not None:
                    c = c[:c.index(hit)]
                safe_r = max(sent_r, len(r) - 7) if open_think else len(r)
                if safe_r > sent_r:
                    yield "reasoning", r[sent_r:safe_r]
                    sent_r = safe_r
                safe = len(c) if hit is not None else max(sent_c, len(c) - hold)
                if not open_think and safe > sent_c:
                    yield "text", c[sent_c:safe]
                    sent_c = safe
                if hit is not None:
                    matched_stop = hit
                    finish = "stop_sequence"
                    break
                # cons.done() здесь не спрашиваем: проверка черновика принимает пачку слов сразу, и ограничение
                # может быть уже «завершено» концом пачки, пока мы выдаём её середину; конец придёт словом конца
            total = time.perf_counter() - t0
        full = self.tk.decode(gen, skip_special_tokens=False)
        r, c, calls, _ = _split(full, think)
        if finish == "stop_sequence":
            hit = _first_stop(c, stop)
            c = c[:c.index(hit)]
        if len(r) > sent_r:
            yield "reasoning", r[sent_r:]
        if len(c) > sent_c:
            yield "text", c[sent_c:]
        if calls:
            finish = "tool_calls"
        if cons is not None and finish == "stop":
            cons.validate(c)
        yield "end", {"text": c, "reasoning": r, "tool_calls": calls, "finish": finish, "prompt_n": len(ids),
                      "gen_n": seen, "decode_n": max(0, seen - 1),
                      "stop_sequence": matched_stop, "first_token_ms": st.get("first_token_ms", st.get("prompt_ms", 0)),
                      "prompt_ms": st.get("prompt_ms", 0), "gen_ms": st.get("gen_ms", 0),
                      "total_s": total}


# ---------------- преобразование форматов ----------------
def _tool_result(content, call_id=None, name=None, is_error=False):
    if not call_id:
        return _text(content)
    return json.dumps({"tool_call_id": call_id, "name": name or "", "is_error": bool(is_error),
                       "result": _maybe_json(_text(content))}, ensure_ascii=False)


def openai_messages(msgs):
    out, names = [], {}
    for m in msgs:
        role = "system" if m.get("role") == "developer" else m.get("role")
        o = {"role": role, "content": _text(m.get("content"))}
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc["function"]
                call_id = tc.get("id")
                if call_id:
                    names[call_id] = fn["name"]
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": fn["name"], "arguments": _maybe_json(fn.get("arguments"))}})
            o["tool_calls"] = calls
        if role == "tool":
            call_id = m.get("tool_call_id")
            name = m.get("name") or names.get(call_id, "")
            if call_id and call_id not in names:
                raise ValueError("результат неизвестного вызова инструмента")
            o.update(name=name, tool_call_id=call_id)
            o["content"] = _tool_result(m.get("content"), call_id, name, m.get("is_error", False))
        out.append(o)
    return out


def _maybe_json(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except json.JSONDecodeError:
            return a
    return a


def anthropic_messages(req):
    out, names = [], {}
    if req.get("system"):
        out.append({"role": "system", "content": _text(req["system"])})
    for m in req.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        texts, calls, results = [], [], []
        for block in content:
            typ = block.get("type")
            if typ == "text":
                texts.append(block.get("text", ""))
            elif typ == "tool_use":
                call_id = block["id"]
                names[call_id] = block["name"]
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": block["name"], "arguments": block.get("input", {})}})
            elif typ == "tool_result":
                call_id = block["tool_use_id"]
                if call_id not in names:
                    raise ValueError("результат неизвестного вызова инструмента")
                results.append({"role": "tool", "name": names[call_id], "tool_call_id": call_id,
                                "content": _tool_result(block.get("content"), call_id, names[call_id], block.get("is_error", False))})
        if m["role"] == "assistant":
            obj = {"role": "assistant", "content": "".join(texts)}
            if calls:
                obj["tool_calls"] = calls
            out.append(obj)
        else:
            out.extend(results)
            if texts:
                out.append({"role": "user", "content": "".join(texts)})
    return out


def anthropic_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("input_schema", {"type": "object"})}} for t in tools or []]


def _schema_from_openai(rf):
    if not rf:
        return None
    if rf.get("type") == "json_object":
        return {"type": "object"}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema", {})
        return js.get("schema", js)
    return None


def _schema_from_ollama(fmt):
    if not fmt:
        return None
    return "json" if fmt == "json" else fmt


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            self.request.settimeout(15)
            super().setup()

        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, obj, code=200):
            if getattr(self, "_stream_started", False):
                return self._stream_error(obj)
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            if getattr(self, "close_connection", False):
                self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.wfile.write(b)

        def _start_stream(self, ctype, protocol="openai"):
            self._stream_started = True
            self._stream_protocol = "ollama" if ctype == "application/x-ndjson" else protocol
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True

        def _stream_error(self, obj):
            self.close_connection = True
            if self._stream_protocol == "ollama":
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
            elif self._stream_protocol == "anthropic":
                self._sse({"type": "error", **obj}, "error")
            else:
                self._sse(obj)
                self._sse("[DONE]")

        def _sse(self, obj, event=None):
            s = (f"event: {event}\n" if event else "") + "data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)) + "\n\n"
            self.wfile.write(s.encode())
            self.wfile.flush()

        def _auth(self):
            if not app.api_key:
                return True
            h = self.headers.get("Authorization", "")
            if h == f"Bearer {app.api_key}" or self.headers.get("x-api-key") == app.api_key:
                return True
            self.close_connection = True  # POST-тело ещё не прочитано: соединение переиспользовать нельзя
            self._json({"error": {"message": "invalid api key", "type": "authentication_error"}}, 401)
            return False

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/health", "/"):
                return self._json({"status": "ok", "model": app.name, "run_metadata": app.metadata()})
            if not self._auth():
                return
            if p == "/v1/models":
                return self._json({"object": "list", "data": [{"id": app.name, "object": "model", "owned_by": "llm_app"}]})
            if p == "/api/tags":
                return self._json({"models": [{"name": app.name, "model": app.name}]})
            if p == "/api/version":
                return self._json({"version": "0.1.0-llm_app"})
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not app.slots.acquire(blocking=False):
                self.close_connection = True
                return self._json({"error": {"message": "очередь запросов заполнена"}}, 503)
            try:
                return self._post()
            finally:
                app.slots.release()

        def _post(self):
            self._stream_started = False
            if not self._auth():
                return
            try:
                length = self.headers.get("Content-Length", "0")
                if not length.isascii() or not length.isdecimal() or self.headers.get("Transfer-Encoding"):
                    raise ValueError("некорректный Content-Length или Transfer-Encoding")
                length = int(length)
                if length > 2 * 2**20:
                    self.close_connection = True
                    return self._json({"error": {"message": "тело запроса слишком велико"}}, 413)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("неполное тело запроса")
                req = validate_request(json.loads(body.decode("utf-8") or "{}"))
            except (ValueError, UnicodeError, TypeError) as exc:
                self.close_connection = True
                return self._json({"error": {"message": str(exc)}}, 400)
            except OSError:
                self.close_connection = True
                return
            p = self.path.split("?")[0]
            try:
                validate_capabilities(req, app.name, getattr(app.eng, "maxt", 2048), p)
                if p == "/v1/chat/completions":
                    return self.openai_chat(req)
                if p == "/v1/completions":
                    return self.openai_completion(req)
                if p == "/v1/messages":
                    return self.anthropic(req)
                if p in ("/api/chat", "/api/generate"):
                    return self.ollama(req, p == "/api/chat")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
                return
            except TimeoutError as exc:
                return self._json({"error": {"message": str(exc), "type": "timeout_error"}}, 504)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, NotImplementedError, RuntimeError) as e:
                return self._json({"error": {"message": str(e), "type": "invalid_request_error"}}, 400)
            self._json({"error": "not found"}, 404)

        # ---------------- OpenAI ----------------
        def openai_chat(self, req):
            choice = req.get("tool_choice", "auto")
            if choice not in ("auto", "none"):
                raise NotImplementedError("принудительный/конкретный tool_choice пока не поддерживается")
            tools = req.get("tools") if choice != "none" else None
            think = req.get("reasoning_effort") not in (None, "", "none") or bool(req.get("think"))
            schema = _schema_from_openai(req.get("response_format"))
            if schema is not None:
                think = False
            ids = app.build_ids(openai_messages(req.get("messages", [])), tools, think)
            stop = req.get("stop")
            stop = [stop] if isinstance(stop, str) else (stop or [])
            gen = app.run(ids, int(req.get("max_completion_tokens", req.get("max_tokens", 1024))),
                          float(req.get("temperature", 1.0) if req.get("temperature") is not None else 1.0),
                          float(req.get("top_p", 1.0) or 1.0), req.get("seed"), stop, think, app.constraint(schema))
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            fr = {"stop": "stop", "stop_sequence": "stop", "length": "length", "tool_calls": "tool_calls"}

            def calls_of(fin):
                return [{"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)
                                      if not isinstance(c["arguments"], str) else c["arguments"]}} for c in fin["tool_calls"]]

            usage = lambda f: {"prompt_tokens": f["prompt_n"], "completion_tokens": f["gen_n"], "total_tokens": f["prompt_n"] + f["gen_n"]}
            if not req.get("stream"):
                fin = None
                for kind, v in gen:
                    if kind == "end":
                        fin = v
                msg = {"role": "assistant", "content": fin["text"] if fin["text"] or not fin["tool_calls"] else None}
                if fin["reasoning"]:
                    msg["reasoning_content"] = fin["reasoning"]
                if fin["tool_calls"]:
                    msg["tool_calls"] = calls_of(fin)
                return self._json({"id": cid, "object": "chat.completion", "created": created, "model": app.name,
                                   "choices": [{"index": 0, "message": msg, "finish_reason": fr[fin["finish"]]}],
                                   "usage": usage(fin)})
            self._start_stream("text/event-stream")
            chunk = lambda delta, finish=None: {"id": cid, "object": "chat.completion.chunk", "created": created, "model": app.name,
                                                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self._sse(chunk({"role": "assistant", "content": ""}))
            for kind, v in gen:
                if kind == "text":
                    self._sse(chunk({"content": v}))
                elif kind == "reasoning":
                    self._sse(chunk({"reasoning_content": v}))
                else:
                    if v["tool_calls"]:
                        self._sse(chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls_of(v))]}))
                    self._sse(chunk({}, fr[v["finish"]]))
                    if (req.get("stream_options") or {}).get("include_usage"):
                        self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": app.name,
                                   "choices": [], "usage": usage(v)})
            self._sse("[DONE]")

        def openai_completion(self, req):
            prompt = req.get("prompt", "")
            if isinstance(prompt, list):
                if prompt and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in prompt):
                    ids = prompt
                elif len(prompt) == 1 and isinstance(prompt[0], str):
                    ids = app.tk.encode(prompt[0], add_special_tokens=False)
                else:
                    raise ValueError("поддерживается один prompt: строка или список ID токенов")
            elif isinstance(prompt, str):
                ids = app.tk.encode(prompt, add_special_tokens=False)
            else:
                raise ValueError("неверная форма prompt")
            stop = req.get("stop")
            stop = [stop] if isinstance(stop, str) else (stop or [])
            gen = app.run(ids, int(req.get("max_tokens", 256)), float(req.get("temperature", 1.0)),
                          float(req.get("top_p", 1.0)), req.get("seed"), stop)
            if req.get("stream"):
                cid = "cmpl-" + uuid.uuid4().hex[:24]
                self._start_stream("text/event-stream")
                for kind, value in gen:
                    if kind == "reasoning":
                        continue
                    finish = None
                    text = value if kind == "text" else ""
                    if kind == "end":
                        finish = "length" if value["finish"] == "length" else "stop"
                    self._sse({"id": cid, "object": "text_completion", "created": int(time.time()), "model": app.name,
                               "choices": [{"index": 0, "text": text, "finish_reason": finish}]})
                self._sse("[DONE]")
                return
            fin = None
            for kind, value in gen:
                if kind == "end":
                    fin = value
            return self._json({"id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion", "created": int(time.time()),
                               "model": app.name, "choices": [{"index": 0, "text": fin["text"],
                                                               "finish_reason": "length" if fin["finish"] == "length" else "stop"}],
                               "usage": {"prompt_tokens": fin["prompt_n"], "completion_tokens": fin["gen_n"],
                                         "total_tokens": fin["prompt_n"] + fin["gen_n"]}})

        # ---------------- Anthropic ----------------
        def anthropic(self, req):
            choice = (req.get("tool_choice") or {}).get("type", "auto")
            if choice not in ("auto", "none"):
                raise NotImplementedError("принудительный/конкретный tool_choice пока не поддерживается")
            tools = anthropic_tools(req.get("tools")) if choice != "none" else None
            think = (req.get("thinking") or {}).get("type") == "enabled"
            ids = app.build_ids(anthropic_messages(req), tools, think)
            gen = app.run(ids, int(req.get("max_tokens", 1024)), float(req.get("temperature", 1.0) if req.get("temperature") is not None else 1.0),
                          float(req.get("top_p", 1.0) or 1.0), None, req.get("stop_sequences"), think)
            mid = "msg_" + uuid.uuid4().hex[:24]
            sr = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "stop_sequence": "stop_sequence"}

            def blocks(fin):
                b = []
                if fin["reasoning"]:
                    b.append({"type": "thinking", "thinking": fin["reasoning"], "signature": ""})
                if fin["text"]:
                    b.append({"type": "text", "text": fin["text"]})
                for c in fin["tool_calls"]:
                    b.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:24], "name": c["name"],
                              "input": _maybe_json(c["arguments"]) if isinstance(c["arguments"], str) else c["arguments"]})
                return b

            if not req.get("stream"):
                fin = None
                for kind, v in gen:
                    if kind == "end":
                        fin = v
                return self._json({"id": mid, "type": "message", "role": "assistant", "model": app.name, "content": blocks(fin),
                                   "stop_reason": sr[fin["finish"]], "stop_sequence": fin.get("stop_sequence"),
                                   "usage": {"input_tokens": fin["prompt_n"], "output_tokens": fin["gen_n"]}})
            self._start_stream("text/event-stream", "anthropic")
            self._sse({"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": app.name,
                                                           "content": [], "stop_reason": None, "stop_sequence": None,
                                                           "usage": {"input_tokens": len(ids), "output_tokens": 0}}}, "message_start")
            idx, cur = 0, None

            def open_block(kind):
                nonlocal cur
                if cur == kind:
                    return
                close_block()
                cur = kind
                b = {"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking" else {"type": "text", "text": ""}
                self._sse({"type": "content_block_start", "index": idx, "content_block": b}, "content_block_start")

            def close_block():
                nonlocal cur, idx
                if cur is not None:
                    self._sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
                    idx += 1
                    cur = None

            for kind, v in gen:
                if kind == "reasoning":
                    open_block("thinking")
                    self._sse({"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": v}}, "content_block_delta")
                elif kind == "text":
                    open_block("text")
                    self._sse({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": v}}, "content_block_delta")
                else:
                    close_block()
                    for c in v["tool_calls"]:
                        args = c["arguments"] if not isinstance(c["arguments"], str) else _maybe_json(c["arguments"])
                        self._sse({"type": "content_block_start", "index": idx, "content_block": {
                            "type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:24], "name": c["name"], "input": {}}}, "content_block_start")
                        self._sse({"type": "content_block_delta", "index": idx, "delta": {
                            "type": "input_json_delta", "partial_json": json.dumps(args, ensure_ascii=False)}}, "content_block_delta")
                        self._sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
                        idx += 1
                    self._sse({"type": "message_delta", "delta": {"stop_reason": sr[v["finish"]], "stop_sequence": v.get("stop_sequence")},
                               "usage": {"output_tokens": v["gen_n"]}}, "message_delta")
                    self._sse({"type": "message_stop"}, "message_stop")

        # ---------------- Ollama ----------------
        def ollama(self, req, is_chat):
            o = dict(req.get("options") or {})
            think = bool(req.get("think", False))
            schema = _schema_from_ollama(req.get("format"))
            if schema is not None:
                think = False
            if is_chat:
                ids = app.build_ids(openai_messages(req.get("messages", [])), req.get("tools"), think)
            elif req.get("raw"):
                ids = app.tk.encode(req["prompt"], add_special_tokens=False)
            else:
                messages = []
                if req.get("system"):
                    messages.append({"role": "system", "content": req["system"]})
                messages.append({"role": "user", "content": req.get("prompt", "")})
                ids = app.build_ids(messages, None, think)
            gen = app.run(ids, int(o.get("num_predict", 512)), float(o.get("temperature", 0.8)), float(o.get("top_p", 1.0)),
                          o.get("seed"), o.get("stop"), think, app.constraint(schema))

            def fin_fields(f):
                return {"done": True, "done_reason": "stop" if f["finish"] != "length" else "length",
                        "total_duration": int(f["total_s"] * 1e9), "prompt_eval_count": f["prompt_n"],
                        "prompt_eval_duration": int(f["prompt_ms"] * 1e6), "eval_count": f["decode_n"],
                        "total_output_tokens": f["gen_n"], "first_token_count": min(f["gen_n"], 1),
                        "eval_duration": int(f["gen_ms"] * 1e6)}

            def msg(f):
                m = {"role": "assistant", "content": f["text"]}
                if f["reasoning"]:
                    m["thinking"] = f["reasoning"]
                if f["tool_calls"]:
                    m["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in f["tool_calls"]]
                return m

            if not req.get("stream", True):
                f = None
                for kind, v in gen:
                    if kind == "end":
                        f = v
                body = {"model": app.name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fin_fields(f)}
                body.update({"message": msg(f)} if is_chat else {"response": f["text"], "thinking": f["reasoning"]})
                return self._json(body)
            self._start_stream("application/x-ndjson")
            for kind, v in gen:
                if kind == "end":
                    body = {"model": app.name, **fin_fields(v)}
                    body.update({"message": dict(msg(v), content="", thinking="")} if is_chat else {"response": ""})
                elif kind == "text":
                    body = {"model": app.name, "done": False}
                    body.update({"message": {"role": "assistant", "content": v}} if is_chat else {"response": v})
                else:
                    body = {"model": app.name, "done": False}
                    body.update({"message": {"role": "assistant", "thinking": v, "content": ""}} if is_chat else {"thinking": v, "response": ""})
                body["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self.wfile.write((json.dumps(body, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
    return H


class LimitedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, *args, **kwargs):
        self.connections = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


def serve(app, host="127.0.0.1", port=11435):
    srv = LimitedHTTPServer((host, port), make_handler(app))
    print(f"сервер: http://{host}:{port} (OpenAI: /v1/chat/completions; Anthropic: /v1/messages; Ollama: /api/chat)"
          + ("; нужен ключ" if app.api_key else ""), flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        for engine in (app.eng, app.draft):
            if engine is not None and hasattr(engine, "close"):
                engine.close()
