"""Проверка запросов и преобразование сообщений внешних протоколов."""

import json
import math
from typing import Any


def validate_request(req: Any) -> Any:
    if not isinstance(req, dict):
        raise ValueError("тело запроса должно быть JSON-объектом")
    if "messages" in req:
        if not isinstance(req["messages"], list) or not req["messages"]:
            raise ValueError("messages должен быть непустым списком")
        for msg in req["messages"]:
            if not isinstance(msg, dict) or msg.get("role") not in (
                "system",
                "developer",
                "user",
                "assistant",
                "tool",
            ):
                raise ValueError("неверная форма сообщения или role")
            content = msg.get("content")
            if content is not None and (not isinstance(content, (str, list))):
                raise ValueError("content должен быть строкой или списком частей")
            if isinstance(content, list) and any(not isinstance(p, (str, dict)) for p in content):
                raise ValueError("неверная часть content")
    if "options" in req and (not isinstance(req["options"], dict)):
        raise ValueError("options должен быть объектом")
    for obj in (req, req.get("options", {})):
        for key, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
            if obj.get(key) is not None:
                value = obj[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or (not math.isfinite(value))
                ):
                    raise ValueError(f"{key} должен быть конечным числом")
                if value < lower or value > upper or (key == "top_p" and value == 0):
                    raise ValueError(f"неверный диапазон {key}")
        for key in ("max_tokens", "max_completion_tokens", "num_predict"):
            if key in obj and (
                isinstance(obj[key], bool) or not isinstance(obj[key], int) or obj[key] < 0
            ):
                raise ValueError(f"{key} должен быть неотрицательным целым")
        if (
            "seed" in obj
            and obj["seed"] is not None
            and (
                isinstance(obj["seed"], bool) or not isinstance(obj["seed"], int) or obj["seed"] < 0
            )
        ):
            raise ValueError("seed должен быть неотрицательным целым")
        for key in ("stop", "stop_sequences"):
            value = obj.get(key)
            if value is not None and (
                not (
                    isinstance(value, str)
                    or (isinstance(value, list) and all(isinstance(s, str) for s in value))
                )
            ):
                raise ValueError(f"{key} должен быть строкой или списком строк")
    for key in ("tools",):
        if key in req and (
            not isinstance(req[key], list) or any(not isinstance(t, dict) for t in req[key])
        ):
            raise ValueError(f"{key} должен быть списком объектов")
    return req


def validate_capabilities(req: Any, model: Any, maxt: Any, endpoint: Any) -> Any:
    if req.get("model") not in (None, model):
        raise ValueError(f"загружена только модель {model}")
    if req.get("n", 1) != 1:
        raise NotImplementedError("поддерживается только n=1")
    for obj in (req, req.get("options", {})):
        for key in ("frequency_penalty", "presence_penalty"):
            if obj.get(key, 0) != 0:
                raise NotImplementedError(f"{key} пока не поддерживается")
        if obj.get("repeat_penalty", 1) != 1:
            raise NotImplementedError("repeat_penalty пока не поддерживается")
        if "top_k" in obj and obj["top_k"] not in (0, -1):
            if not (obj.get("temperature", req.get("temperature", 1)) == 0 and obj["top_k"] == 1):
                raise NotImplementedError("top_k пока не поддерживается при sampling")
        if "num_ctx" in obj and obj["num_ctx"] != maxt:
            raise ValueError(f"контекст сервера фиксирован: {maxt}")
    for key in ("logprobs", "logit_bias", "suffix", "echo"):
        if req.get(key):
            raise NotImplementedError(f"{key} пока не поддерживается")
    for msg in req.get("messages", []):
        for part in msg.get("content") or [] if isinstance(msg.get("content"), list) else []:
            if isinstance(part, dict) and part.get("type") not in (
                "text",
                "input_text",
                "output_text",
                "tool_use",
                "tool_result",
                "thinking",
            ):
                raise NotImplementedError("поддерживается только текст и инструменты")


def _text(content: Any) -> Any:
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


def _maybe_json(a: Any) -> Any:
    """Разобрать строку JSON, оставив обычный текст без изменения."""
    if isinstance(a, str):
        try:
            return json.loads(a)
        except json.JSONDecodeError:
            return a
    return a


def _tool_result(content: Any, call_id: Any = None, name: Any = None, is_error: Any = False) -> Any:
    if not call_id:
        return _text(content)
    return json.dumps(
        {
            "tool_call_id": call_id,
            "name": name or "",
            "is_error": bool(is_error),
            "result": _maybe_json(_text(content)),
        },
        ensure_ascii=False,
    )


def openai_messages(msgs: Any) -> Any:
    """Преобразовать сообщения OpenAI во внутренний шаблон модели."""
    out, names = ([], {})
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
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fn["name"],
                            "arguments": _maybe_json(fn.get("arguments")),
                        },
                    }
                )
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


def anthropic_messages(req: Any) -> Any:
    """Преобразовать сообщения и результаты инструментов Anthropic."""
    out, names = ([], {})
    if req.get("system"):
        out.append({"role": "system", "content": _text(req["system"])})
    for m in req.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        texts, calls, results = ([], [], [])
        for block in content:
            typ = block.get("type")
            if typ == "text":
                texts.append(block.get("text", ""))
            elif typ == "tool_use":
                call_id = block["id"]
                names[call_id] = block["name"]
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": block["name"], "arguments": block.get("input", {})},
                    }
                )
            elif typ == "tool_result":
                call_id = block["tool_use_id"]
                if call_id not in names:
                    raise ValueError("результат неизвестного вызова инструмента")
                results.append(
                    {
                        "role": "tool",
                        "name": names[call_id],
                        "tool_call_id": call_id,
                        "content": _tool_result(
                            block.get("content"),
                            call_id,
                            names[call_id],
                            block.get("is_error", False),
                        ),
                    }
                )
        if m["role"] == "assistant":
            obj: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if calls:
                obj["tool_calls"] = calls
            out.append(obj)
        else:
            out.extend(results)
            if texts:
                out.append({"role": "user", "content": "".join(texts)})
    return out


def anthropic_tools(tools: Any) -> Any:
    """Преобразовать схемы инструментов Anthropic в шаблон модели."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tools or []
    ]


def _schema_from_openai(rf: Any) -> Any:
    if not rf:
        return None
    if rf.get("type") == "json_object":
        return {"type": "object"}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema", {})
        return js.get("schema", js)
    return None


def _schema_from_ollama(fmt: Any) -> Any:
    if not fmt:
        return None
    return "json" if fmt == "json" else fmt
