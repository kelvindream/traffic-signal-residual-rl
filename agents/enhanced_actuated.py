"""Deterministic enhanced actuated controller for AsyncMultiIntersectionEnv."""

import numpy as np

from environment.async_mult_intersection_env import AsyncMultiIntersectionEnv


class EnhancedActuatedController:
    """Pure rule controller that emits only the environment's two actions."""

    def __init__(self, env, tls_id, min_green=10, max_green=40, passage=3,
                 starvation_limit=30, switch_hysteresis=1.0):
        self.env = env
        self.tls_id = tls_id
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.passage = float(passage)
        self.starvation_limit = float(starvation_limit)
        self.switch_hysteresis = float(switch_hysteresis)

    def _now(self):
        return float(self.env.sumo.simulation.getTime())

    def _runtime(self):
        return self.env.runtime[self.tls_id]

    def _demands(self):
        return np.asarray(self.env._phase_demands(self.tls_id), dtype=float)

    def select_action(self):
        runtime = self._runtime()
        now = self._now()
        if runtime.yellow_until is not None:
            return AsyncMultiIntersectionEnv.HOLD
        elapsed = max(0.0, now - runtime.current_phase_started_at)
        if elapsed < self.min_green:
            return AsyncMultiIntersectionEnv.HOLD
        if elapsed >= self.max_green:
            return AsyncMultiIntersectionEnv.TERMINATE

        demands = self._demands()
        current = int(runtime.phase_index)
        current_demand = float(demands[current])
        waiting = np.asarray(runtime.phase_wait_seconds, dtype=float)
        if waiting.size and float(np.max(np.delete(waiting, current))) > self.starvation_limit:
            return AsyncMultiIntersectionEnv.TERMINATE

        candidate = self._next_demand_phase(demands, current)
        candidate_demand = float(demands[candidate])
        if current_demand <= 0.0:
            gap_out = now - float(runtime.last_current_demand_at) >= self.passage
            return (AsyncMultiIntersectionEnv.TERMINATE if gap_out
                    else AsyncMultiIntersectionEnv.HOLD)
        if candidate != current and candidate_demand > current_demand + self.switch_hysteresis:
            return AsyncMultiIntersectionEnv.TERMINATE
        return AsyncMultiIntersectionEnv.HOLD

    def _next_demand_phase(self, demands, current):
        for offset in range(1, len(demands) + 1):
            phase = (current + offset) % len(demands)
            if demands[phase] > 0:
                return phase
        return (current + 1) % len(demands)
