"""Обработчики ollama: формат запросов и ответов, включая поток."""

import json
import time
from typing import Any

from ramvi_engine.server.validation import _schema_from_ollama, openai_messages


def ollama_routes(app: Any) -> Any:
    """Связать обработчик Ollama с одним приложением."""

    def ollama(self: Any, req: Any, is_chat: Any) -> Any:
        """Обслужить Ollama chat или generate с общей генерацией."""
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
        gen = app.run(
            ids,
            int(o.get("num_predict", 512)),
            float(o.get("temperature", 0.8)),
            float(o.get("top_p", 1.0)),
            o.get("seed"),
            o.get("stop"),
            think,
            app.constraint(schema),
        )

        def fin_fields(f: Any) -> Any:
            return {
                "done": True,
                "done_reason": "stop" if f["finish"] != "length" else "length",
                "total_duration": int(f["total_s"] * 1000000000.0),
                "prompt_eval_count": f["prompt_n"],
                "prompt_eval_duration": int(f["prompt_ms"] * 1000000.0),
                "eval_count": f["decode_n"],
                "total_output_tokens": f["gen_n"],
                "first_token_count": min(f["gen_n"], 1),
                "eval_duration": int(f["gen_ms"] * 1000000.0),
            }

        def msg(f: Any) -> Any:
            m = {"role": "assistant", "content": f["text"]}
            if f["reasoning"]:
                m["thinking"] = f["reasoning"]
            if f["tool_calls"]:
                m["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in f["tool_calls"]
                ]
            return m

        if not req.get("stream", True):
            f: Any = None
            for kind, v in gen:
                if kind == "end":
                    f = v
            body = {
                "model": app.name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **fin_fields(f),
            }
            body.update(
                {"message": msg(f)}
                if is_chat
                else {"response": f["text"], "thinking": f["reasoning"]}
            )
            return self._json(body)
        self._start_stream("application/x-ndjson")
        for kind, v in gen:
            if kind == "end":
                body = {"model": app.name, **fin_fields(v)}
                body.update(
                    {"message": dict(msg(v), content="", thinking="")}
                    if is_chat
                    else {"response": ""}
                )
            elif kind == "text":
                body = {"model": app.name, "done": False}
                body.update(
                    {"message": {"role": "assistant", "content": v}} if is_chat else {"response": v}
                )
            else:
                body = {"model": app.name, "done": False}
                body.update(
                    {"message": {"role": "assistant", "thinking": v, "content": ""}}
                    if is_chat
                    else {"thinking": v, "response": ""}
                )
            body["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.wfile.write((json.dumps(body, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

    return ollama
