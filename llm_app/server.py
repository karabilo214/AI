"""HTTP-сервер, совместимый с Ollama (/api/generate, /api/chat, /api/tags) и OpenAI (/v1/chat/completions,
/v1/models). Одна генерация за раз. Поля времени — как у Ollama (наносекунды): prompt_eval_count/duration,
eval_count/duration — чтобы одни и те же клиенты и замеры работали с обоими."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class App:
    def __init__(self, engine, tokenizer, name, draft=None, k=7):
        self.eng, self.tk, self.name, self.draft, self.k = engine, tokenizer, name, draft, k
        self.lock = threading.Lock()

    def chat_ids(self, messages, think=False):
        text = self.tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        return self.tk.encode(text, add_special_tokens=False)

    def run(self, ids, opts):
        """Генератор (кусок текста, итог или None)."""
        temp = float(opts.get("temperature", 0.8 if "temperature" not in opts else opts["temperature"]))
        st = {}
        gen = []
        sent = ""
        with self.lock:
            t0 = time.perf_counter()
            for t in self.eng.generate(ids, int(opts.get("num_predict", opts.get("max_tokens", 512)) or 512), temp,
                                       float(opts.get("top_p", 1.0)), opts.get("seed"),
                                       draft=self.draft if temp <= 0 else None, k=self.k, stats=st):
                if t in self.eng.ck.eos:
                    break
                gen.append(t)
                txt = self.tk.decode(gen, skip_special_tokens=True)
                if not txt.endswith("�"):
                    yield txt[len(sent):], None
                    sent = txt
            total = time.perf_counter() - t0
        txt = self.tk.decode(gen, skip_special_tokens=True)
        yield txt[len(sent):], {"total_duration": int(total * 1e9), "prompt_eval_count": st.get("prompt_n", 0),
                                "prompt_eval_duration": int(st.get("prompt_ms", 0) * 1e6), "eval_count": len(gen),
                                "eval_duration": int(st.get("gen_ms", 0) * 1e6), "done_reason": "stop"}


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/api/tags", "/v1/models"):
                self._json({"models": [{"name": app.name, "model": app.name}], "data": [{"id": app.name, "object": "model"}]})
            else:
                self._json({"status": "ok"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            opts = dict(req.get("options") or {})
            for k in ("temperature", "top_p", "max_tokens", "seed"):
                if k in req:
                    opts[k] = req[k]
            if self.path == "/api/generate":
                ids = app.tk.encode(req["prompt"], add_special_tokens=False) if req.get("raw") else \
                    app.chat_ids([{"role": "user", "content": req["prompt"]}], bool(req.get("think", False)))
                kind = "generate"
            elif self.path in ("/api/chat", "/v1/chat/completions"):
                ids = app.chat_ids(req["messages"], bool(req.get("think", False)))
                kind = "chat" if self.path == "/api/chat" else "openai"
            else:
                return self._json({"error": "unknown path"}, 404)
            stream = req.get("stream", kind != "openai")
            if not stream:
                text, fin = "", None
                for piece, f in app.run(ids, opts):
                    text += piece
                    fin = f or fin
                if kind == "generate":
                    return self._json({"model": app.name, "response": text, "done": True, **fin})
                if kind == "chat":
                    return self._json({"model": app.name, "message": {"role": "assistant", "content": text}, "done": True, **fin})
                return self._json({"id": "c", "object": "chat.completion", "model": app.name,
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": fin["prompt_eval_count"], "completion_tokens": fin["eval_count"]}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if kind == "openai" else "application/x-ndjson")
            self.end_headers()
            for piece, fin in app.run(ids, opts):
                if kind == "openai":
                    ch = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": "stop" if fin else None}]}
                    self.wfile.write(b"data: " + json.dumps(ch, ensure_ascii=False).encode() + b"\n\n")
                    if fin:
                        self.wfile.write(b"data: [DONE]\n\n")
                else:
                    obj = {"model": app.name, "done": fin is not None, **(fin or {})}
                    obj.update({"response": piece} if kind == "generate" else {"message": {"role": "assistant", "content": piece}})
                    self.wfile.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()
    return H


def serve(app, host="127.0.0.1", port=11435):
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"сервер: http://{host}:{port} (Ollama: /api/chat, /api/generate; OpenAI: /v1/chat/completions)", flush=True)
    srv.serve_forever()
