"""Capture RPC failure is observable and never leaves its owned native process alive."""

import base64
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.chat import ChatError
from ramvi_service.voice.capture import ProcessMicrophone
from ramvi_service.voice.session import VoiceSession
from test_voice import FakeTranscriber, wait
from test_voice_session import Detector


def test_failed_native_open_stops_process_without_capture() -> None:
    class Worker:
        stopped = False

        def start(self) -> Any:
            pass

        def request(self, *args: Any, **kwargs: Any) -> Any:
            raise TimeoutError("private device name")

        def stop(self, **kwargs: Any) -> Any:
            self.stopped = True

    worker = Worker()
    with pytest.raises(ChatError, match="voice.device_error"):
        ProcessMicrophone(lambda: worker).open(None, lambda _: pytest.fail("late capture"))
    assert worker.stopped


def test_native_reader_failure_is_propagated_to_session_and_closed() -> None:
    class Worker:
        stopped = threading.Event()
        reads = 0

        def start(self) -> Any:
            pass

        def request(self, method: Any, payload: Any, timeout: Any) -> Any:
            if method == "start":
                return {"rate": 16000}
            self.reads += 1
            if self.reads == 1:
                return {
                    "blocks": [base64.b64encode(bytes(3200)).decode()],
                    "captured_at": [time.monotonic()],
                }
            raise OSError("native capture lost")

        def stop(self, **kwargs: Any) -> Any:
            self.stopped.set()

    worker = Worker()
    released = threading.Event()
    recognizer = FakeTranscriber(Path("test"))
    session = VoiceSession(
        ProcessMicrophone(lambda: worker), Detector(), recognizer, on_close=released.set
    )
    try:
        session.start()
        wait(lambda: session.state == "stopped")
        session.join()
        assert (
            session.error == "voice.device_error" and released.is_set() and worker.stopped.is_set()
        )
        assert recognizer.closed
        assert not session.stream and not any(t.is_alive() for t in session.threads)
    finally:
        session.stop()
        session.join()


def test_buffered_audio_before_gate_cutoff_is_never_delivered_afterward() -> None:
    entered = threading.Event()
    release = threading.Event()
    captured: list[Any] = []
    old = time.monotonic() - 1

    class Worker:
        stopped = threading.Event()
        reads = 0

        def start(self) -> Any:
            pass

        def request(self, method: Any, payload: Any, timeout: Any) -> Any:
            if method == "start":
                return {"rate": 16000}
            self.reads += 1
            if self.reads == 1:
                entered.set()
                release.wait(2)
                return {"blocks": [base64.b64encode(b"\1\0" * 1600).decode()], "captured_at": [old]}
            if self.reads == 2:
                return {
                    "blocks": [base64.b64encode(b"\2\0" * 1600).decode()],
                    "captured_at": [time.monotonic()],
                }
            self.stopped.wait(2)
            raise OSError("closed")

        def stop(self, **kwargs: Any) -> Any:
            self.stopped.set()

    worker = Worker()
    stream = ProcessMicrophone(lambda: worker).open(None, captured.append)
    try:
        assert entered.wait(1)
        stream.flush(time.monotonic())
        release.set()
        wait(lambda: bool(captured))
        assert captured == [b"\2\0" * 1600]
    finally:
        release.set()
        stream.close()


def test_native_capture_clock_is_comparable_across_owned_processes() -> None:
    import sys

    from ramvi_engine.processes import ProcessSupervisor

    source = """
import sys,json,time,base64
for line in sys.stdin:
 request=json.loads(line)
 if request['method']=='start':result={'rate':16000,'clock':time.monotonic()}
 else:
  time.sleep(.02)
  result={'blocks':[base64.b64encode(bytes(3200)).decode()],'captured_at':[time.monotonic()]}
 print(json.dumps({'id':request['id'],'result':result}),flush=True)
"""
    process = ProcessSupervisor([sys.executable, "-u", "-c", source], timeout=2)
    clock = []
    received: list[Any] = []

    class Worker:
        def start(self) -> Any:
            return process.start()

        def request(self, method: Any, payload: Any, timeout: Any) -> Any:
            result = process.request(method, payload, timeout=timeout)
            if method == "start":
                clock.append(result["clock"])
            return result

        def stop(self, **kwargs: Any) -> Any:
            return process.stop(**kwargs)

    before = time.monotonic()
    stream = ProcessMicrophone(Worker).open(None, received.append)
    after = time.monotonic()
    try:
        assert before <= clock[0] <= after
        wait(lambda: bool(received))
    finally:
        stream.close()
    assert stream.thread is not None
    assert process.status()["state"] == "stopped" and not stream.thread.is_alive()
