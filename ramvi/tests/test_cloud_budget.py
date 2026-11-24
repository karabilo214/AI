"""7.10: стоимость облачных ответов, лимиты расходов, повтор только там, где он безопасен."""

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.chat import ChatError
from ramvi_service.cloud import CloudConnections, CloudProvider, model_id
from ramvi_service.cloud_budget import Budget, cost_of
from ramvi_storage.repositories import Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client
from test_cloud import KEY, Cloud, Secrets

HI = {"messages": [{"role": "user", "content": "hi"}]}
PRICE = {"input": 3.0, "output": 15.0, "source": "https://example.com/pricing"}
ONE = (7 * 3.0 + 3 * 15.0) / 1_000_000  # тестовое облако отвечает 7 входных и 3 выходных токена


@pytest.fixture
def cloud() -> Iterator[Cloud]:
    server = Cloud()
    yield server
    server.close()


@pytest.fixture
def connections(tmp_path: Path, cloud: Cloud) -> Iterator[CloudConnections]:
    db = open_core(tmp_path)
    store = CloudConnections(tmp_path, Secrets(), Models(db))
    store.save({"kind": "openai", "name": "c", "url": cloud.url("openai"), "key": KEY})
    store.enable("c", "gpt-a", True)
    yield store
    store.budget.close()
    db.close()


def provider(store: CloudConnections, source: str = "app") -> CloudProvider:
    remote = store.provider(store.models.get(model_id("c", "gpt-a")), source)
    assert isinstance(remote, CloudProvider)
    remote.retry_delays = (0.01, 0.01)
    return remote


def posts(cloud: Cloud) -> int:
    return sum(r["method"] == "POST" for r in cloud.requests)


def test_each_answer_is_recorded_with_tokens_and_cost(
    connections: CloudConnections, cloud: Cloud
) -> None:
    saved = connections.set_price("c", {"model": "gpt-a", **PRICE})
    assert saved["price"] == {
        **PRICE,
        "checked_on": time.strftime("%Y-%m-%d"),
    }  # цена с источником и датой
    list(provider(connections).stream(HI, threading.Event()))
    provider(connections, "api:ide").complete(HI, threading.Event())
    spending = connections.spending()
    assert spending["today"] == spending["month"] == pytest.approx(2 * ONE)
    assert [
        (r["source"], r["prompt_tokens"], r["completion_tokens"], r["state"])
        for r in spending["recent"]
    ] == [
        ("api:ide", 7, 3, "done"),
        ("app", 7, 3, "done"),
    ]
    assert spending["recent"][0]["cost"] == pytest.approx(ONE)
    assert cost_of({"prompt_tokens": 1_000_000, "completion_tokens": 0}, PRICE) == 3.0


def test_without_a_price_the_cost_is_unknown_not_zero(connections: CloudConnections) -> None:
    list(provider(connections).stream(HI, threading.Event()))
    spending = connections.spending()
    assert (
        spending["recent"][0]["cost"] is None
        and spending["unknown_today"] == 1
        and spending["today"] == 0
    )
    with pytest.raises(ChatError) as error:
        connections.set_price("c", {"model": "gpt-a", "input": 1, "output": 2})
    assert error.value.code == "cloud.price_source_required"


def test_daily_limit_stops_the_next_request_before_it_is_sent(
    connections: CloudConnections, cloud: Cloud
) -> None:
    connections.set_price("c", {"model": "gpt-a", **PRICE})
    connections.set_limits({"daily": ONE * 1.5, "monthly": None})
    provider(connections).complete(HI, threading.Event())  # 1·ONE < лимита
    provider(connections).complete(HI, threading.Event())  # перед ним ещё было меньше лимита
    sent = posts(cloud)
    with pytest.raises(ChatError) as error:
        provider(connections).complete(HI, threading.Event())
    assert error.value.code == "cloud.limit_reached" and error.value.params["period"] == "day"
    assert error.value.params["spent"] == pytest.approx(2 * ONE)
    assert posts(cloud) == sent  # запрос не ушёл
    connections.set_limits({"daily": None, "monthly": ONE})
    with pytest.raises(ChatError) as error:
        provider(connections).complete(HI, threading.Event())
    assert error.value.params["period"] == "month"
    connections.set_limits({"daily": None, "monthly": None})
    provider(connections).complete(HI, threading.Event())  # лимиты сняты


def test_a_running_answer_is_not_cut_by_the_limit(
    connections: CloudConnections, cloud: Cloud
) -> None:
    connections.set_price("c", {"model": "gpt-a", **PRICE})
    connections.set_limits({"daily": 1.0, "monthly": None})
    chunks = provider(connections).stream(HI, threading.Event())
    first = next(chunks)
    connections.set_limits({"daily": 0.0, "monthly": None})  # лимит упал посреди ответа
    rest = list(chunks)
    assert first["type"] in ("reasoning", "text") and rest[-1]["type"] == "end"
    with pytest.raises(ChatError) as error:
        list(provider(connections).stream(HI, threading.Event()))
    assert error.value.code == "cloud.limit_reached"


def test_with_a_limit_an_unpriced_model_is_refused(
    connections: CloudConnections, cloud: Cloud
) -> None:
    connections.set_limits({"daily": 10.0, "monthly": None})
    with pytest.raises(ChatError) as error:
        provider(connections).complete(HI, threading.Event())
    assert error.value.code == "cloud.price_missing" and error.value.params["model"] == "gpt-a"
    assert posts(cloud) == 0


@pytest.mark.parametrize("code", [429, 503, 529])
def test_busy_service_is_retried(connections: CloudConnections, cloud: Cloud, code: int) -> None:
    cloud.failures = [code]
    reply = provider(connections).complete(HI, threading.Event())
    assert reply["choices"][0]["message"]["content"] == "Привет" and posts(cloud) == 2


def test_retry_gives_up_after_three_attempts_and_honours_retry_after(
    connections: CloudConnections, cloud: Cloud
) -> None:
    cloud.failures = [429, 429, 429]
    with pytest.raises(ChatError) as error:
        provider(connections).complete(HI, threading.Event())
    assert error.value.code == "cloud.rate_limited" and posts(cloud) == 3
    cloud.failures, cloud.retry_after = [503], "0.3"
    started = time.monotonic()
    provider(connections).complete(HI, threading.Event())
    assert time.monotonic() - started >= 0.3  # пауза — сколько попросил сервис


@pytest.mark.parametrize("code", [400, 401, 500, 502, 504])
def test_requests_that_may_have_run_are_not_repeated(
    connections: CloudConnections, cloud: Cloud, code: int
) -> None:
    cloud.failures = [code]
    with pytest.raises(ChatError):
        provider(connections).complete(HI, threading.Event())
    assert posts(cloud) == 1  # повтор мог бы списать деньги дважды


def test_cancel_during_the_retry_pause_stops_at_once(
    connections: CloudConnections, cloud: Cloud
) -> None:
    cloud.failures, cloud.retry_after = [529], "5"
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(InterruptedError):
        provider(connections).complete(HI, cancel)
    assert time.monotonic() - started < 2


def test_refused_connection_is_retried_but_a_timeout_is_not() -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    remote = CloudProvider("openai", f"http://127.0.0.1:{port}/v1", "m", lambda: KEY)
    remote.retry_delays = (0.01, 0.01)
    calls: list[int] = []
    original = remote._send

    def counted(*args: Any) -> Any:
        calls.append(1)
        return original(*args)

    remote._send = counted  # type: ignore[method-assign,assignment]
    with pytest.raises(ChatError) as error:
        remote.complete(HI, threading.Event())
    assert (
        error.value.code == "cloud.unreachable"
        and error.value.params["connect"]
        and len(calls) == 3
    )

    # Сервер принял соединение и молчит: запрос мог дойти — повтора нет.
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        quiet = CloudProvider(
            "openai",
            f"http://127.0.0.1:{silent.getsockname()[1]}/v1",
            "m",
            lambda: KEY,
            timeout=0.3,
        )
        quiet.retry_delays = (0.01, 0.01)
        tries: list[int] = []
        send = quiet._send

        def counted_quiet(*args: Any) -> Any:
            tries.append(1)
            return send(*args)

        quiet._send = counted_quiet  # type: ignore[method-assign,assignment]
        with pytest.raises(ChatError) as error:
            quiet.complete(HI, threading.Event())
        assert (
            error.value.code == "cloud.unreachable"
            and not error.value.params["connect"]
            and len(tries) == 1
        )


def test_a_cut_stream_is_recorded_as_cut(connections: CloudConnections, cloud: Cloud) -> None:
    cloud.slow = True
    cancel = threading.Event()
    chunks = provider(connections).stream(HI, cancel)
    next(chunks)
    cancel.set()
    with pytest.raises(InterruptedError):
        list(chunks)
    assert connections.spending()["recent"][0]["state"] == "cut"


def test_day_and_month_follow_the_local_calendar(tmp_path: Path) -> None:
    now = [time.mktime((2026, 11, 29, 12, 0, 0, 0, 0, -1))]
    budget = Budget(tmp_path / "u.sqlite3", clock=lambda: now[0])
    budget.record("m", "app", {"prompt_tokens": 1_000_000, "completion_tokens": 0}, PRICE, "done")
    now[0] = time.mktime((2026, 11, 30, 23, 0, 0, 0, 0, -1))
    assert budget.spent("day") == (0.0, 0) and budget.spent("month") == (
        3.0,
        0,
    )  # другой день того же месяца
    now[0] = time.mktime((2026, 11, 29, 23, 0, 0, 0, 0, -1))
    budget.record("m", "app", {"prompt_tokens": 0, "completion_tokens": 0}, PRICE, "done")
    now[0] = time.mktime((2026, 11, 30, 23, 0, 0, 0, 0, -1))
    budget._db.execute("DELETE FROM usage")
    budget.record("m", "app", {"prompt_tokens": 1_000_000, "completion_tokens": 0}, PRICE, "done")
    now[0] += 2 * 3600  # 1 декабря, 01:00
    budget.record("m", "app", {"prompt_tokens": 1_000_000, "completion_tokens": 0}, PRICE, "done")
    assert budget.spent("day") == (3.0, 0) and budget.spent("month") == (3.0, 0)
    now[0] -= 2 * 3600
    assert budget.spent("day") == (3.0, 0) and budget.spent("month") == (3.0, 0)
    budget.close()


def test_routes_show_spending_and_set_limits_and_prices(tmp_path: Path, cloud: Cloud) -> None:
    from ramvi_service.app import build

    server, engines = build(
        tmp_path, TOKEN, engine_command=ENGINE, voice_options={"secrets": Secrets()}
    )
    server.start()
    try:
        client = Client(server.port)
        client.ok(
            "POST",
            "/cloud/connections",
            {"kind": "openai", "name": "c", "url": cloud.url("openai"), "key": KEY},
        )
        client.ok("POST", "/cloud/connections/c/models", {"model": "gpt-a", "enabled": True})
        price = client.ok("POST", "/cloud/connections/c/prices", {"model": "gpt-a", **PRICE})
        assert price["price"]["input"] == 3.0
        spending = client.ok("POST", "/cloud/limits", {"daily": 5, "monthly": 50})["spending"]
        assert (
            spending["limits"] == {"daily": 5.0, "monthly": 50.0} and spending["currency"] == "USD"
        )
        chat = client.ok("POST", "/chats", {"model_id": "cloud:c:gpt-a"})["chat"]
        client.events(f"/chats/{chat['id']}/send", {"text": "hi"})
        view = client.ok("GET", "/cloud")
        assert (
            view["spending"]["today"] == pytest.approx(ONE)
            and view["spending"]["recent"][0]["source"] == "app"
        )
        status, error = client.request("POST", "/cloud/limits", {"daily": -1})
        assert status != 200
        client.ok(
            "POST", "/cloud/connections/c/prices", {"model": "gpt-a", "input": None, "output": None}
        )
        assert client.ok("GET", "/cloud")["spending"]["prices"] == {}
        client.ok("POST", "/cloud/connections/c/prices", {"model": "gpt-a", **PRICE})
        client.ok("DELETE", "/cloud/connections/c")
        assert client.ok("GET", "/cloud")["spending"]["prices"] == {}  # цены уходят с подключением
    finally:
        server.close()
        engines.stop()
