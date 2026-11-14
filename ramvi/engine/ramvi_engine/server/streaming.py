"""Поток генерации, разбор служебных меток и транспорт SSE/NDJSON."""

import json
import queue
import re
import threading
import time
from typing import Any

TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def _split(full: Any, think: Any) -> Any:
    """Полный текст модели -> (рассуждение, ответ без вызовов инструментов, вызовы инструментов, хвост ответа
    может быть началом служебной метки)."""
    reasoning, content = ("", full)
    stripped = full.lstrip("\n")
    if stripped and "<think>".startswith(stripped):
        return ("", "", [], True)
    explicit = stripped.startswith("<think>")
    if think or explicit:
        start = full.find("<think>") + 7 if explicit else 0
        end = full.find("</think>", start)
        if end < 0:
            return (full[start:].lstrip("\n"), "", [], True)
        reasoning = full[start:end].lstrip("\n")
        content = full[end + 8 :].lstrip("\n")
    else:
        content = content.replace("</think>", "")
    calls, pieces, last = ([], [], 0)
    for match in TOOL_RE.finditer(content):
        pieces.append(content[last : match.start()])
        try:
            obj = json.loads(match.group(1))
            if (
                not isinstance(obj, dict)
                or not isinstance(obj.get("name"), str)
                or (not obj["name"])
            ):
                raise ValueError("неверная форма вызова")
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("аргументы вызова должны быть объектом")
            calls.append({"name": obj["name"], "arguments": args})
        except (ValueError, TypeError):
            pieces.append(match.group(0))
        last = match.end()
    tail = content[last:]
    opening = tail.find("<tool_call>")
    pieces.append(tail if opening < 0 else tail[:opening])
    return (reasoning, "".join(pieces), calls, False)


def _held(text: Any, markers: Any) -> Any:
    """Длина хвоста text, который может оказаться началом одной из меток (его пока не отдаём клиенту).
    Раньше держали фиксированные len(метки)-1 символов: первый кусок ответа уходил только после ~11 символов,
    то есть после первого прохода угадывания (+0.5-1 с к первому слову)."""
    best = 0
    for m in markers:
        for n in range(min(len(m) - 1, len(text)), best, -1):
            if text.endswith(m[:n]):
                best = n
                break
    return best


def _first_stop(content: Any, stops: Any) -> Any:
    hits = [(content.find(stop), i, stop) for i, stop in enumerate(stops) if stop in content]
    return min(hits)[2] if hits else None


class GenerationStream:
    """Выдача событий генерации с ограниченной очередью и отменой."""

    QUEUE_SIZE = 8
    QUEUE_TIMEOUT = 10.0
    ENGINE_TIMEOUT = 60.0

    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Сетевая выдача отделена от вычислений ограниченной очередью."""
        events: queue.Queue[Any] = queue.Queue(maxsize=self.QUEUE_SIZE)
        cancel, finished = (threading.Event(), threading.Event())
        failures: list[Exception] = []
        end = object()

        def put(value: Any) -> Any:
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

        def produce() -> Any:
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
                        raise failures[0]  # noqa: B904 - прежняя цепочка исключений
                    break
                try:
                    item = events.get(timeout=0.05)
                except queue.Empty:
                    if finished.is_set():
                        if failures:
                            raise failures[0]  # noqa: B904 - прежняя цепочка исключений
                        break
                    if time.monotonic() - last_event >= self.ENGINE_TIMEOUT:
                        raise TimeoutError("модель не ответила вовремя")  # noqa: B904 - прежняя цепочка исключений
                    continue
                if item is end:
                    if failures:
                        raise failures[0]  # noqa: B904 - прежняя цепочка исключений
                    break
                yield item
                last_event = time.monotonic()
        finally:
            cancel.set()
            worker.join(timeout=5)

    def _run(
        self: Any,
        ids: Any,
        max_tokens: Any = 512,
        temperature: Any = 0.0,
        top_p: Any = 1.0,
        seed: Any = None,
        stop: Any = None,
        think: Any = False,
        cons: Any = None,
    ) -> Any:
        """Генератор событий: ("reasoning", кусок) | ("text", кусок) | ("end", итог)."""
        if self.force_greedy:
            temperature = 0.0
        stop = [s for s in stop or [] if s]
        markers = ["<tool_call>", "</think>"] + stop
        st: dict[str, Any] = {}
        gen, seen = [], 0
        matched_stop = None
        sent_r, sent_c, finish = (0, 0, "length")
        with self.lock:
            t0 = time.perf_counter()
            for t in self.eng.generate(
                ids,
                max_tokens,
                temperature,
                top_p,
                seed,
                draft=self.draft if temperature <= 0 else None,
                k=self.k,
                stats=st,
                cons=cons,
            ):
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
                    c = c[: c.index(hit)]
                safe_r = max(sent_r, len(r) - _held(r, ["</think>"])) if open_think else len(r)
                if safe_r > sent_r:
                    yield ("reasoning", r[sent_r:safe_r])
                    sent_r = safe_r
                safe = len(c) if hit is not None else max(sent_c, len(c) - _held(c, markers))
                if not open_think and safe > sent_c:
                    yield ("text", c[sent_c:safe])
                    sent_c = safe
                if hit is not None:
                    matched_stop = hit
                    finish = "stop_sequence"
                    break
            total = time.perf_counter() - t0
        full = self.tk.decode(gen, skip_special_tokens=False)
        r, c, calls, _ = _split(full, think)
        if finish == "stop_sequence":
            hit = _first_stop(c, stop)
            c = c[: c.index(hit)]
        if len(r) > sent_r:
            yield ("reasoning", r[sent_r:])
        if len(c) > sent_c:
            yield ("text", c[sent_c:])
        if calls:
            finish = "tool_calls"
        if cons is not None and finish == "stop":
            cons.validate(c)
        yield (
            "end",
            {
                "text": c,
                "reasoning": r,
                "tool_calls": calls,
                "finish": finish,
                "prompt_n": len(ids),
                "gen_n": seen,
                "decode_n": max(0, seen - 1),
                "stop_sequence": matched_stop,
                "first_token_ms": st.get("first_token_ms", st.get("prompt_ms", 0)),
                "prompt_ms": st.get("prompt_ms", 0),
                "gen_ms": st.get("gen_ms", 0),
                "total_s": total,
            },
        )


class StreamingResponse:
    """Общий HTTP-транспорт для JSON, SSE и NDJSON."""

    def _cors(self: Any) -> Any:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self: Any, obj: Any, code: Any = 200) -> Any:
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

    def _start_stream(self: Any, ctype: Any, protocol: Any = "openai") -> Any:
        self._stream_started = True
        self._stream_protocol = "ollama" if ctype == "application/x-ndjson" else protocol
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        self.close_connection = True

    def _stream_error(self: Any, obj: Any) -> Any:
        self.close_connection = True
        if self._stream_protocol == "ollama":
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()
        elif self._stream_protocol == "anthropic":
            self._sse({"type": "error", **obj}, "error")
        else:
            self._sse(obj)
            self._sse("[DONE]")

    def _sse(self: Any, obj: Any, event: Any = None) -> Any:
        s = (
            (f"event: {event}\n" if event else "")
            + "data: "
            + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))
            + "\n\n"
        )
        self.wfile.write(s.encode())
        self.wfile.flush()
