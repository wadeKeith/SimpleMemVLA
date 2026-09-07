
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from robomme_sim.robomme_env import SimEnvService

_ENV_INIT_LOCK = threading.Lock()


class InProcSimPool:
    def __init__(
        self,
        num_envs: int,
        dataset_split: str = "test",
        max_steps: int = 1300,
        step_timeout: float = 300.0,
        reset_timeout: float = 3600.0,
    ):
        self.num_envs = num_envs
        self.dataset_split = dataset_split
        self.max_steps = int(max_steps)
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout
        self.services: list[SimEnvService] = []
        self.alive: list[bool] = []
        self._pool = ThreadPoolExecutor(max_workers=max(2, num_envs))

    def start(self):
        self.services = [
            SimEnvService(dataset_split=self.dataset_split, max_steps=self.max_steps)
            for _ in range(self.num_envs)
        ]
        self.alive = [True] * self.num_envs

    def _reset_one(self, i: int, spec: dict):
        with _ENV_INIT_LOCK:
            return self.services[i].reset(spec)

    def reset(self, specs: list[dict]) -> list[dict]:
        self.alive = [True] * self.num_envs
        n = min(len(specs), self.num_envs)
        futs = {self._pool.submit(self._reset_one, i, specs[i]): i for i in range(n)}
        results: list = [None] * self.num_envs
        for fut, i in futs.items():
            try:
                results[i] = fut.result(timeout=self.reset_timeout)
                if not (isinstance(results[i], dict) and results[i].get("ok")):
                    self.alive[i] = False
            except Exception as e:
                self.alive[i] = False
                results[i] = {"ok": False, "reason": f"reset_exc: {e}"}
        return results

    def step(self, action_chunks: list, active: list[int]) -> list[dict]:
        results: list = [None] * self.num_envs
        futs = {}
        for i in active:
            if not self.alive[i]:
                continue
            futs[self._pool.submit(self.services[i].step, {"action_chunk": action_chunks[i]})] = i
        for fut, i in futs.items():
            try:
                results[i] = fut.result(timeout=self.step_timeout)
            except Exception as e:
                self.alive[i] = False
                results[i] = {"error": f"step_exc: {e}"}
        return results

    def close(self):
        for svc in self.services:
            try:
                svc.close()
            except Exception:
                pass
        self.services = []
        self._pool.shutdown(wait=False)
