
from __future__ import annotations

import queue
import threading

from libero_sim.libero_env import SimEnvService


class _EnvWorker(threading.Thread):

    _STOP_SENTINEL = object()

    def __init__(self, service: SimEnvService):
        super().__init__(daemon=True)
        self.service = service
        self._requests: queue.Queue = queue.Queue()

    def run(self):
        while True:
            item = self._requests.get()
            if item is self._STOP_SENTINEL:
                try:
                    self.service.close()
                except Exception:
                    pass
                return
            method, payload, reply = item
            try:
                if method == "reset":
                    reply.put(("ok", self.service.reset(payload)))
                elif method == "step":
                    reply.put(("ok", self.service.step(payload)))
                elif method == "close":
                    reply.put(("ok", self.service.close()))
                else:
                    reply.put(("err", f"unknown method {method!r}"))
            except Exception as e:
                reply.put(("err", f"{type(e).__name__}: {e}"))

    def submit(self, method: str, payload) -> queue.Queue:
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._requests.put((method, payload, reply))
        return reply

    def stop(self):
        self._requests.put(self._STOP_SENTINEL)


class InProcSimPool:
    def __init__(
        self,
        num_envs: int,
        max_steps: int | None = None,
        num_steps_wait: int = 10,
        image_size: int = 256,
        step_timeout: float = 300.0,
        reset_timeout: float = 600.0,
        action_space: str = "osc",
    ):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.num_steps_wait = int(num_steps_wait)
        self.image_size = int(image_size)
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout
        self.action_space = action_space
        self.workers: list[_EnvWorker] = []
        self.alive: list[bool] = []

    def start(self):
        self.workers = [
            _EnvWorker(
                SimEnvService(
                    max_steps=self.max_steps,
                    num_steps_wait=self.num_steps_wait,
                    image_size=self.image_size,
                    action_space=self.action_space,
                )
            )
            for _ in range(self.num_envs)
        ]
        for w in self.workers:
            w.start()
        self.alive = [True] * self.num_envs

    def _collect(self, replies: dict[int, queue.Queue], timeout: float, kind: str) -> list:
        results: list = [None] * self.num_envs
        for i, reply in replies.items():
            try:
                status, value = reply.get(timeout=timeout)
            except queue.Empty:
                self.alive[i] = False
                results[i] = (
                    {"ok": False, "reason": f"{kind}_timeout"}
                    if kind == "reset"
                    else {"error": f"{kind}_timeout"}
                )
                continue
            if status != "ok":
                self.alive[i] = False
                results[i] = (
                    {"ok": False, "reason": f"{kind}_exc: {value}"}
                    if kind == "reset"
                    else {"error": f"{kind}_exc: {value}"}
                )
            else:
                results[i] = value
                if kind == "reset" and not (isinstance(value, dict) and value.get("ok")):
                    self.alive[i] = False
        return results

    def reset(self, specs: list[dict]) -> list[dict]:
        self.alive = [True] * self.num_envs
        n = min(len(specs), self.num_envs)
        replies = {i: self.workers[i].submit("reset", specs[i]) for i in range(n)}
        return self._collect(replies, self.reset_timeout, "reset")

    def step(self, action_chunks: list, active: list[int]) -> list[dict]:
        replies = {
            i: self.workers[i].submit("step", {"action_chunk": action_chunks[i]})
            for i in active
            if self.alive[i]
        }
        return self._collect(replies, self.step_timeout, "step")

    def close(self):
        for w in self.workers:
            try:
                w.stop()
            except Exception:
                pass
        self.workers = []
