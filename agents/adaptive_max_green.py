"""Demand-responsive max-green control on the demand-aware Fixed baseline."""

import math

import numpy as np


class AdaptiveMaxGreenController:
    """Keep Fixed phase selection, varying only the current green target."""

    def __init__(self, env, base_green=40.0, min_target=30.0, max_target=50.0,
                 demand_gain=5.0, wait_gain=0.05):
        self.env = env
        self.base_green = float(base_green)
        self.min_target = float(min_target)
        self.max_target = float(max_target)
        self.demand_gain = float(demand_gain)
        self.wait_gain = float(wait_gain)
        values = (self.base_green, self.min_target, self.max_target,
                  self.demand_gain, self.wait_gain)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("adaptive green parameters must be finite")
        if not self.min_target <= self.base_green <= self.max_target:
            raise ValueError("base green must lie inside target bounds")
        if self.min_target < env.min_green or self.max_target > env.max_green:
            raise ValueError("target bounds must respect environment limits")
        self.stats = {
            "fixed_hold": 0,
            "adaptive_hold": 0,
            "terminate": 0,
            "mask_veto": 0,
        }

    def target_green(self, tls_id):
        runtime = self.env.runtime[tls_id]
        demand = np.asarray(self.env._phase_demands(tls_id), dtype=np.float32)
        waits = np.asarray(runtime.phase_wait_seconds, dtype=np.float32)
        current = float(demand[runtime.phase_index])
        mean = float(demand.mean())
        current_wait = float(waits[runtime.phase_index])
        relative_demand = (current - mean) / max(mean, 1.0)
        target = self.base_green + self.demand_gain * relative_demand
        target += self.wait_gain * min(current_wait, 60.0)
        return float(np.clip(target, self.min_target, self.max_target))

    def select_action(self, tls_id):
        runtime = self.env.runtime[tls_id]
        now = float(self.env.sumo.simulation.getTime())
        elapsed = max(0.0, now - runtime.current_phase_started_at)
        desired = self.env.TERMINATE if elapsed + 1e-9 >= self.target_green(tls_id) else self.env.HOLD
        mask = self.env.action_mask(tls_id, now)
        if not bool(mask[desired]):
            self.stats["mask_veto"] += 1
            desired = self.env.TERMINATE if bool(mask[self.env.TERMINATE]) else self.env.HOLD
        if desired == self.env.HOLD:
            if elapsed < self.base_green:
                self.stats["fixed_hold"] += 1
            else:
                self.stats["adaptive_hold"] += 1
        else:
            self.stats["terminate"] += 1
        return desired
