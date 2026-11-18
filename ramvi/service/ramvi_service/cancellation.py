"""Cancellation view: пользователь и отзыв гранта останавливают одну операцию."""

import threading
import time


class LinkedCancellation(threading.Event):
    def __init__(self, *parents: threading.Event) -> None:
        super().__init__()
        self.parents = parents

    def is_set(self) -> bool:
        return super().is_set() or any(parent.is_set() for parent in self.parents)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                return False
            super().wait(0.05 if left is None else min(left, 0.05))
        return True
