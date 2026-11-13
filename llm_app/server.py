"""HTTP-сервер для приложений и агентов: OpenAI (/v1/chat/completions, /v1/completions, /v1/models), Anthropic
(/v1/messages — Claude Code и клиенты Claude), Ollama (/api/chat, /api/generate, /api/tags). DeepSeek и Codex
ходят через формат OpenAI.

Возможности: потоковая выдача (SSE / ndjson), вызов инструментов (tools / tool_calls, tool_use / tool_result),
ответ строго в JSON или по JSON Schema (response_format / format — ограничение грамматикой, совместимо с
угадыванием), стоп-слова, рассуждения модели отдельно (reasoning_content / thinking), учёт слов (usage).
Сеть: --host 0.0.0.0, ключ --api-key (Authorization: Bearer ... или x-api-key), CORS, запросы — по очереди."""
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    if think or full.lstrip().startswith("<think>") or "</think>" in full:
        a = full.find("<think>")
        b = full.find("</think>")
        if a >= 0 and b < 0:
            return full[a + 7:], "", [], True
        if b >= 0:
            reasoning = full[a + 7 if a >= 0 else 0:b].strip("\n")
            content = full[b + 8:].lstrip("\n")
    calls = []
    i = content.find("<tool_call>")
    body = content if i < 0 else content[:i]
    for m in TOOL_RE.finditer(content):
        try:
            o = json.loads(m.group(1))
            calls.append({"name": o.get("name", ""), "arguments": o.get("arguments", o.get("parameters", {}))})
        except json.JSONDecodeError:
            pass
    return reasoning, body, calls, False


class App:
    def __init__(self, engine, tokenizer, name, draft=None, k=None, api_key=None, force_greedy=False):
        self.eng, self.tk, self.name, self.draft, self.k = engine, tokenizer, name, draft, k
        self.api_key, self.force_greedy = api_key, force_greedy
        self.lock = threading.Lock()

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
    def run(self, ids, max_tokens=512, temperature=0.0, top_p=1.0, seed=None, stop=None, think=False, cons=None):
        """Генератор событий: ("reasoning", кусок) | ("text", кусок) | ("end", итог)."""
        if self.force_greedy:
            temperature = 0.0
        stop = [s for s in (stop or []) if s]
        hold = max([len("<tool_call>")] + [len(s) for s in stop]) - 1
        st, gen = {}, []
        sent_r, sent_c, finish = 0, 0, "length"
        with self.lock:
            t0 = time.perf_counter()
            for t in self.eng.generate(ids, max_tokens, temperature, top_p, seed,
                                       draft=self.draft if temperature <= 0 else None, k=self.k, stats=st, cons=cons):
                if t in self.eng.ck.eos:
                    finish = "stop"
                    break
                gen.append(t)
                full = self.tk.decode(gen, skip_special_tokens=False)
                if full.endswith("�"):
                    continue
                r, c, calls, open_think = _split(full, think)
                hit = next((s for s in stop if s in c), None)
                if hit is not None:
                    c = c[:c.index(hit)]
                if len(r) > sent_r:
                    yield "reasoning", r[sent_r:]
                    sent_r = len(r)
                safe = len(c) if hit is not None else max(sent_c, len(c) - hold)
                if not open_think and safe > sent_c:
                    yield "text", c[sent_c:safe]
                    sent_c = safe
                if hit is not None:
                    finish = "stop_sequence"
                    break
                # cons.done() здесь не спрашиваем: проверка черновика принимает пачку слов сразу, и ограничение
                # может быть уже «завершено» концом пачки, пока мы выдаём её середину; конец придёт словом конца
            total = time.perf_counter() - t0
        full = self.tk.decode(gen, skip_special_tokens=False)
        r, c, calls, _ = _split(full, think)
        if finish == "stop_sequence":
            hit = next(s for s in stop if s in c)
            c = c[:c.index(hit)]
        if len(c) > sent_c:
            yield "text", c[sent_c:]
        if calls:
            finish = "tool_calls"
        yield "end", {"text": c, "reasoning": r, "tool_calls": calls, "finish": finish, "prompt_n": len(ids),
                      "gen_n": len(gen), "prompt_ms": st.get("prompt_ms", 0), "gen_ms": st.get("gen_ms", 0),
                      "total_s": total}


# ---------------- преобразование форматов ----------------
def openai_messages(msgs):
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "developer":
            role = "system"
        o = {"role": role, "content": _text(m.get("content"))}
        if m.get("tool_calls"):
            o["tool_calls"] = [{"type": "function", "function": {"name": tc["function"]["name"],
                                "arguments": _maybe_json(tc["function"].get("arguments"))}} for tc in m["tool_calls"]]
        if role == "tool":
            o["name"] = m.get("name", "")
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
    out = []
    sysm = req.get("system")
    if sysm:
        out.append({"role": "system", "content": _text(sysm)})
    for m in req.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        texts, calls, results = [], [], []
        for b in c:
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "tool_use":
                calls.append({"type": "function", "function": {"name": b["name"], "arguments": b.get("input", {})}})
            elif t == "tool_result":
                rc = b.get("content")
                results.append({"role": "tool", "content": _text(rc) if not isinstance(rc, str) else rc})
        if m["role"] == "assistant":
            o = {"role": "assistant", "content": "".join(texts)}
            if calls:
                o["tool_calls"] = calls
            out.append(o)
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
        return "json"
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

        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self._cors()
            self.end_headers()
            self.wfile.write(b)

        def _start_stream(self, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True

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
                return self._json({"status": "ok", "model": app.name})
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
            if not self._auth():
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": {"message": "invalid JSON body"}}, 400)
            p = self.path.split("?")[0]
            try:
                if p == "/v1/chat/completions":
                    return self.openai_chat(req)
                if p == "/v1/completions":
                    return self.openai_completion(req)
                if p == "/v1/messages":
                    return self.anthropic(req)
                if p in ("/api/chat", "/api/generate"):
                    return self.ollama(req, p == "/api/chat")
            except (ValueError, NotImplementedError, RuntimeError) as e:
                return self._json({"error": {"message": str(e), "type": "invalid_request_error"}}, 400)
            self._json({"error": "not found"}, 404)

        # ---------------- OpenAI ----------------
        def openai_chat(self, req):
            tools = req.get("tools") if req.get("tool_choice") != "none" else None
            think = bool(req.get("reasoning_effort")) or bool(req.get("think"))
            schema = _schema_from_openai(req.get("response_format"))
            if schema is not None:
                think = False
            ids = app.build_ids(openai_messages(req.get("messages", [])), tools, think)
            stop = req.get("stop")
            stop = [stop] if isinstance(stop, str) else (stop or [])
            gen = app.run(ids, int(req.get("max_completion_tokens") or req.get("max_tokens") or 1024),
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
            prompt = prompt[0] if isinstance(prompt, list) else prompt
            ids = app.tk.encode(prompt, add_special_tokens=False)
            stop = req.get("stop")
            stop = [stop] if isinstance(stop, str) else (stop or [])
            fin = None
            for kind, v in app.run(ids, int(req.get("max_tokens") or 256), float(req.get("temperature", 1.0) or 0.0),
                                   float(req.get("top_p", 1.0) or 1.0), req.get("seed"), stop):
                if kind == "end":
                    fin = v
            return self._json({"id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion", "created": int(time.time()),
                               "model": app.name, "choices": [{"index": 0, "text": fin["text"],
                                                               "finish_reason": "length" if fin["finish"] == "length" else "stop"}],
                               "usage": {"prompt_tokens": fin["prompt_n"], "completion_tokens": fin["gen_n"],
                                         "total_tokens": fin["prompt_n"] + fin["gen_n"]}})

        # ---------------- Anthropic ----------------
        def anthropic(self, req):
            tools = anthropic_tools(req.get("tools")) if (req.get("tool_choice") or {}).get("type") != "none" else None
            think = (req.get("thinking") or {}).get("type") == "enabled"
            ids = app.build_ids(anthropic_messages(req), tools, think)
            gen = app.run(ids, int(req.get("max_tokens") or 1024), float(req.get("temperature", 1.0) if req.get("temperature") is not None else 1.0),
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
                                   "stop_reason": sr[fin["finish"]], "stop_sequence": None,
                                   "usage": {"input_tokens": fin["prompt_n"], "output_tokens": fin["gen_n"]}})
            self._start_stream("text/event-stream")
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
                    self._sse({"type": "message_delta", "delta": {"stop_reason": sr[v["finish"]], "stop_sequence": None},
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
                ids = app.build_ids([{"role": "user", "content": req.get("prompt", "")}], None, think)
            gen = app.run(ids, int(o.get("num_predict", 512) or 512), float(o.get("temperature", 0.8)), float(o.get("top_p", 1.0)),
                          o.get("seed"), o.get("stop"), think, app.constraint(schema))

            def fin_fields(f):
                return {"done": True, "done_reason": "stop" if f["finish"] != "length" else "length",
                        "total_duration": int(f["total_s"] * 1e9), "prompt_eval_count": f["prompt_n"],
                        "prompt_eval_duration": int(f["prompt_ms"] * 1e6), "eval_count": f["gen_n"],
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
                body.update({"message": msg(f)} if is_chat else {"response": f["text"]})
                return self._json(body)
            self._start_stream("application/x-ndjson")
            for kind, v in gen:
                if kind == "end":
                    body = {"model": app.name, **fin_fields(v)}
                    body.update({"message": dict(msg(v), content="")} if is_chat else {"response": ""})
                elif kind == "text":
                    body = {"model": app.name, "done": False}
                    body.update({"message": {"role": "assistant", "content": v}} if is_chat else {"response": v})
                else:
                    continue
                self.wfile.write((json.dumps(body, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
    return H


def serve(app, host="127.0.0.1", port=11435):
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"сервер: http://{host}:{port} (OpenAI: /v1/chat/completions; Anthropic: /v1/messages; Ollama: /api/chat)"
          + ("; нужен ключ" if app.api_key else ""), flush=True)
    srv.serve_forever()
