from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "evaluation_benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import task2_26_reference_stage as stages  # noqa: E402


class _Model:
    body_names = ["tomato_sauce_1", "cookies_1"]

    @staticmethod
    def body_name2id(name: str) -> int:
        aliases = {
            "tomato_sauce_1": 0,
            "tomato_sauce_1_main": 0,
            "cookies_1": 1,
            "cookies_1_main": 1,
        }
        if name not in aliases:
            raise KeyError(name)
        return aliases[name]


class _Data:
    def __init__(self) -> None:
        self.body_xpos = np.zeros((2, 3), dtype=np.float64)
        self.body_xmat = np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (2, 1))


class _Sim:
    def __init__(self) -> None:
        self.model = _Model()
        self.data = _Data()


class _Env:
    def __init__(self) -> None:
        self.sim = _Sim()

    def set_source(self, *, z: float, tilt: float) -> None:
        self.sim.data.body_xpos[0] = [0.0, 0.0, z]
        c, s = np.cos(tilt), np.sin(tilt)
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        self.sim.data.body_xmat[0] = rotation.reshape(9)


class SharedPourCounterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _Env()
        self.state = {
            "step_idx": 0,
            "initial_body_pos": {"tomato_sauce_1": np.zeros(3, dtype=np.float64)},
            "shared_pour_counters": {},
        }
        self.specs = stages._task_specs(6)

    def _check(self, predicate, angle: float) -> bool:
        self.state["step_idx"] += 1
        self.env.set_source(z=0.04, tilt=angle)
        return predicate(self.env, self.state, 0)

    def _rearm(self, predicate) -> None:
        for _ in range(5):
            predicate(self.env, self.state, 0)
            self._check(predicate, 0.0)

    def test_one_long_tilt_counts_once(self) -> None:
        pour_one = self.specs[1].check_fn
        self.assertFalse(self._check(pour_one, 0.0))
        for _ in range(3):
            detected = self._check(pour_one, 0.8)
        self.assertTrue(detected)
        for _ in range(20):
            self.assertTrue(self._check(pour_one, 0.8))
        counter = next(iter(self.state["shared_pour_counters"].values()))
        self.assertEqual(counter.event_count, 1)

    def test_second_pour_requires_return_and_third_is_detected(self) -> None:
        lift, pour_one, pour_two = (spec.check_fn for spec in self.specs)
        self.assertTrue(self._check(lift, 0.0))
        self.assertFalse(self._check(pour_one, 0.0))
        for _ in range(3):
            first_done = self._check(pour_one, 0.8)
        self.assertTrue(first_done)

        # Without an upright return, another long tilt cannot become Pour Two.
        for _ in range(10):
            self.assertFalse(self._check(pour_two, 0.8))

        for _ in range(5):
            self._check(pour_two, 0.0)
        for _ in range(3):
            second_done = self._check(pour_two, 0.8)
        self.assertTrue(second_done)

        extra = stages._extra_pour_check(6)
        self.assertIsNotNone(extra)
        for _ in range(5):
            self._check(extra, 0.0)
        for _ in range(3):
            third_detected = self._check(extra, 0.8)
        self.assertTrue(third_detected)

    def test_tilt_without_physical_lift_never_counts(self) -> None:
        pour_one = self.specs[1].check_fn
        for _ in range(20):
            self.state["step_idx"] += 1
            self.env.set_source(z=0.0, tilt=0.8)
            self.assertFalse(pour_one(self.env, self.state, 0))

    def test_all_entrypoints_import_the_canonical_scorer(self) -> None:
        paths = {
            "adapter": ROOT / "evaluation_benchmark" / "scripts" / "eval_tasks2_26.py",
            "sync": ROOT / "evaluation_benchmark" / "reference_evaluation" / "tasks2_26_vlm5_reference" / "eval_tasks2_26_vlm_vla.py",
            "async": ROOT / "evaluation_benchmark" / "async_vlm26_reference" / "eval_fullvlm26_async_vlm_vla.py",
        }
        for name, path in paths.items():
            self.assertIn("task2_26_reference_stage", path.read_text(encoding="utf-8"), name)


if __name__ == "__main__":
    unittest.main()
