"""Safety-preserving platoon green-window coordination over real SUMO links."""

import math

import numpy as np

from agents.enhanced_actuated import EnhancedActuatedController


class PlatoonGreenWindowCoordinator:
    """Preserve an active green only for vehicles on a real TLS-to-TLS edge."""

    def __init__(self, env, max_eta_s=4.0, max_extension_s=2.0):
        self.env = env
        self.max_eta_s = float(max_eta_s)
        self.max_extension_s = float(max_extension_s)
        if (not math.isfinite(self.max_eta_s) or self.max_eta_s < 0 or
                not math.isfinite(self.max_extension_s) or self.max_extension_s < 0):
            raise ValueError("coordination horizons must be finite and non-negative")
        self.base = {tls: EnhancedActuatedController(env, tls)
                     for tls in env.tls_ids}
        self.directed_edges = self._discover_edges()
        self.stats = {"base_terminate": 0, "coordinated_hold": 0,
                      "no_connected_vehicle": 0, "masked_veto": 0}

    def _discover_edges(self):
        net = self.env.sumolib.net.readNet(self.env.net_file)
        controlled = set(self.env.tls_ids)
        edges = {}
        for edge in net.getEdges():
            source = edge.getFromNode().getTLSID()
            target = edge.getToNode().getTLSID()
            if source in controlled and target in controlled and source != target:
                edges[edge.getID()] = (source, target)
        return edges

    def _approaching_on_real_edge(self, tls_id, now):
        vehicle_api = self.env.sumo.vehicle
        for vehicle_id in vehicle_api.getIDList():
            road_id = vehicle_api.getRoadID(vehicle_id)
            speed = float(vehicle_api.getSpeed(vehicle_id))
            if speed <= 0 or road_id.startswith(":"):
                continue
            next_tls = vehicle_api.getNextTLS(vehicle_id)
            for target, _, tls_distance, state in next_tls:
                if target != tls_id or state not in "Gg":
                    continue
                tls_distance = float(tls_distance)
                if not math.isfinite(tls_distance) or tls_distance < 0:
                    continue
                road_relation = self.directed_edges.get(road_id)
                if road_relation is None or road_relation[1] != tls_id:
                    continue
                eta = tls_distance / speed
                lane_id = vehicle_api.getLaneID(vehicle_id)
                lane_length = float(self.env.sumo.lane.getLength(lane_id))
                position = float(vehicle_api.getLanePosition(vehicle_id))
                remaining = max(0.0, lane_length - position)
                if (eta <= self.max_eta_s and eta <= self.max_extension_s and
                        remaining <= max(tls_distance, 1.0)):
                    return vehicle_id, eta
        return None

    def select_action(self, tls_id):
        action = self.base[tls_id].select_action()
        if action != self.env.TERMINATE:
            return action
        self.stats["base_terminate"] += 1
        vehicle = self._approaching_on_real_edge(
            tls_id, float(self.env.sumo.simulation.getTime()))
        if vehicle is None:
            self.stats["no_connected_vehicle"] += 1
            return action
        mask = self.env.action_mask(tls_id)
        if len(mask) <= self.env.HOLD or not bool(mask[self.env.HOLD]):
            self.stats["masked_veto"] += 1
            return action
        self.stats["coordinated_hold"] += 1
        return self.env.HOLD


class FixedCycleResidualCoordinator(PlatoonGreenWindowCoordinator):
    """Add one conservative platoon extension to a fixed-duration base."""

    def __init__(
        self,
        env,
        nominal_green_s=40.0,
        max_eta_s=2.0,
        max_extension_s=2.0,
        min_network_queue=12.0,
    ):
        self.env = env
        self.nominal_green_s = float(nominal_green_s)
        self.max_eta_s = float(max_eta_s)
        self.max_extension_s = float(max_extension_s)
        self.min_network_queue = float(min_network_queue)
        values = (
            self.nominal_green_s,
            self.max_eta_s,
            self.max_extension_s,
            self.min_network_queue,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("residual coordination parameters must be finite and non-negative")
        if self.nominal_green_s + self.max_extension_s > float(env.max_green):
            raise ValueError("nominal green plus extension cannot exceed the safety cap")
        self.directed_edges = self._discover_edges()
        self._extended_phase_start = {}
        self._reserved_phase_start = {}
        self.stats = {
            "base_terminate": 0,
            "residual_reserved": 0,
            "residual_opportunity": 0,
            "residual_hold": 0,
            "low_network_queue": 0,
            "already_extended": 0,
            "no_connected_vehicle": 0,
            "masked_veto": 0,
        }

    def _fixed_cycle_action(self, tls_id, now):
        runtime = self.env.runtime[tls_id]
        elapsed = max(0.0, now - runtime.current_phase_started_at)
        return (
            self.env.TERMINATE
            if elapsed + 1e-9 >= self.nominal_green_s
            else self.env.HOLD
        )

    def _network_queue(self):
        return float(sum(
            float(self.env._phase_demands(tls_id).sum())
            for tls_id in self.env.tls_ids
        ))

    def select_action(self, tls_id):
        now = float(self.env.sumo.simulation.getTime())
        runtime = self.env.runtime[tls_id]
        phase_start = float(runtime.current_phase_started_at)
        elapsed = max(0.0, now - phase_start)
        action = self._fixed_cycle_action(tls_id, now)
        mask = self.env.action_mask(tls_id, now)
        if not bool(mask[action]):
            return self.env.TERMINATE if bool(mask[self.env.TERMINATE]) else self.env.HOLD

        if action != self.env.TERMINATE:
            reservation_window = self.nominal_green_s - self.max_extension_s
            if (
                elapsed + 1e-9 >= reservation_window
                and self._reserved_phase_start.get(tls_id) != phase_start
                and self._network_queue() >= self.min_network_queue
                and self._approaching_on_real_edge(tls_id, now) is not None
            ):
                self._reserved_phase_start[tls_id] = phase_start
                self.stats["residual_reserved"] += 1
            return action

        self.stats["base_terminate"] += 1
        if self._extended_phase_start.get(tls_id) == phase_start:
            self.stats["already_extended"] += 1
            return action
        if self._reserved_phase_start.get(tls_id) != phase_start:
            if self._network_queue() < self.min_network_queue:
                self.stats["low_network_queue"] += 1
            else:
                self.stats["no_connected_vehicle"] += 1
            return action

        self.stats["residual_opportunity"] += 1
        if not bool(mask[self.env.HOLD]):
            self.stats["masked_veto"] += 1
            return action

        self._extended_phase_start[tls_id] = phase_start
        self.stats["residual_hold"] += 1
        return self.env.HOLD


class SplitOffsetGreenWindowCoordinator(PlatoonGreenWindowCoordinator):
    """Demand-split control with platoon-aware phase targeting and Fixed fallback."""

    def __init__(
        self,
        env,
        nominal_green_s=40.0,
        min_split_s=24.0,
        max_split_s=50.0,
        split_gain_s=16.0,
        min_network_queue=12.0,
        max_eta_s=8.0,
    ):
        self.env = env
        self.nominal_green_s = float(nominal_green_s)
        self.min_split_s = float(min_split_s)
        self.max_split_s = float(max_split_s)
        self.split_gain_s = float(split_gain_s)
        self.min_network_queue = float(min_network_queue)
        self.max_eta_s = float(max_eta_s)
        values = (
            self.nominal_green_s,
            self.min_split_s,
            self.max_split_s,
            self.split_gain_s,
            self.min_network_queue,
            self.max_eta_s,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("split-offset parameters must be finite and non-negative")
        if not self.min_split_s <= self.nominal_green_s <= self.max_split_s:
            raise ValueError("nominal green must lie inside split bounds")
        if self.min_split_s < float(env.min_green) or self.max_split_s > float(env.max_green):
            raise ValueError("split bounds must respect environment safety limits")
        self.directed_edges = self._discover_edges()
        self.stats = {
            "fixed_fallback": 0,
            "split_hold": 0,
            "split_terminate": 0,
            "platoon_phase": 0,
            "pressure_phase": 0,
            "masked_veto": 0,
        }

    def _network_queue(self):
        return float(sum(
            float(self.env._phase_demands(tls_id).sum())
            for tls_id in self.env.tls_ids
        ))

    def _fixed_action(self, tls_id, now):
        runtime = self.env.runtime[tls_id]
        elapsed = max(0.0, now - runtime.current_phase_started_at)
        return self.env.TERMINATE if elapsed + 1e-9 >= self.nominal_green_s else self.env.HOLD

    def _target_green(self, tls_id):
        runtime = self.env.runtime[tls_id]
        demands = np.asarray(self.env._phase_demands(tls_id), dtype=np.float32)
        mean = float(demands.mean())
        current = float(demands[runtime.phase_index])
        relative = (current - mean) / max(mean, 1.0)
        target = self.nominal_green_s + self.split_gain_s * relative
        return float(np.clip(target, self.min_split_s, self.max_split_s))

    def _phase_for_lane(self, tls_id, lane_id):
        for phase_index, lanes in enumerate(self.env.phase_lanes[tls_id]):
            if lane_id in lanes:
                return phase_index
        return None

    def _approaching_phase(self, tls_id):
        best = None
        vehicle_api = self.env.sumo.vehicle
        for vehicle_id in vehicle_api.getIDList():
            road_id = vehicle_api.getRoadID(vehicle_id)
            relation = self.directed_edges.get(road_id)
            if relation is None or relation[1] != tls_id:
                continue
            speed = float(vehicle_api.getSpeed(vehicle_id))
            if speed <= 0:
                continue
            lane_id = vehicle_api.getLaneID(vehicle_id)
            phase_index = self._phase_for_lane(tls_id, lane_id)
            if phase_index is None:
                continue
            for target, _, distance, _ in vehicle_api.getNextTLS(vehicle_id):
                if target != tls_id:
                    continue
                eta = float(distance) / speed
                if 0 <= eta <= self.max_eta_s and (best is None or eta < best[0]):
                    best = (eta, phase_index)
                break
        return None if best is None else best[1]

    def _pressure_phase(self, tls_id):
        runtime = self.env.runtime[tls_id]
        demands = np.asarray(self.env._phase_demands(tls_id), dtype=np.float32)
        waits = np.asarray(runtime.phase_wait_seconds, dtype=np.float32)
        scores = demands + waits / max(self.nominal_green_s, 1.0)
        scores[runtime.phase_index] = -np.inf
        return int(np.argmax(scores))

    def select_action(self, tls_id):
        now = float(self.env.sumo.simulation.getTime())
        runtime = self.env.runtime[tls_id]
        elapsed = max(0.0, now - runtime.current_phase_started_at)
        mask = self.env.action_mask(tls_id, now)
        if self._network_queue() < self.min_network_queue:
            self.stats["fixed_fallback"] += 1
            action = self._fixed_action(tls_id, now)
            if bool(mask[action]):
                return action
            self.stats["masked_veto"] += 1
            return self.env.TERMINATE if bool(mask[self.env.TERMINATE]) else self.env.HOLD

        target_green = self._target_green(tls_id)
        if elapsed + 1e-9 < target_green and bool(mask[self.env.HOLD]):
            self.stats["split_hold"] += 1
            return self.env.HOLD
        if not bool(mask[self.env.TERMINATE]):
            self.stats["masked_veto"] += 1
            return self.env.HOLD

        next_phase = self._approaching_phase(tls_id)
        if next_phase is not None and next_phase != runtime.phase_index:
            self.stats["platoon_phase"] += 1
        else:
            next_phase = self._pressure_phase(tls_id)
            self.stats["pressure_phase"] += 1
        self.stats["split_terminate"] += 1
        return {"action": self.env.TERMINATE, "next_phase": next_phase}


class NetworkConstrainedCoordinator(SplitOffsetGreenWindowCoordinator):
    """Budget-conserving split control with platoon and spillback constraints."""

    def __init__(
        self,
        env,
        nominal_green_s=40.0,
        min_split_s=32.0,
        max_split_s=48.0,
        min_network_queue=12.0,
        max_eta_s=8.0,
        min_platoon_size=2,
        spillback_threshold=0.75,
    ):
        self.env = env
        self.nominal_green_s = float(nominal_green_s)
        self.min_split_s = float(min_split_s)
        self.max_split_s = float(max_split_s)
        self.split_gain_s = 0.0
        self.min_network_queue = float(min_network_queue)
        self.max_eta_s = float(max_eta_s)
        self.min_platoon_size = int(min_platoon_size)
        self.spillback_threshold = float(spillback_threshold)
        values = (
            self.nominal_green_s,
            self.min_split_s,
            self.max_split_s,
            self.min_network_queue,
            self.max_eta_s,
            self.spillback_threshold,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("network constraints must be finite and non-negative")
        if self.min_platoon_size < 1:
            raise ValueError("minimum platoon size must be positive")
        if not 0 < self.spillback_threshold <= 1:
            raise ValueError("spillback threshold must lie in (0, 1]")
        if self.min_split_s < float(env.min_green) or self.max_split_s > float(env.max_green):
            raise ValueError("split bounds must respect environment safety limits")
        self.directed_edges = self._discover_edges()
        self.stats = {
            "fixed_fallback": 0,
            "budget_hold": 0,
            "budget_terminate": 0,
            "platoon_phase": 0,
            "pressure_phase": 0,
            "spillback_veto": 0,
            "small_platoon_veto": 0,
            "masked_veto": 0,
        }

    def _phase_targets(self, tls_id):
        demands = np.asarray(self.env._phase_demands(tls_id), dtype=np.float64)
        budget = self.nominal_green_s * len(demands)
        weights = demands + 1.0
        raw = budget * weights / weights.sum()
        targets = np.clip(raw, self.min_split_s, self.max_split_s)
        for _ in range(8):
            difference = budget - float(targets.sum())
            if abs(difference) < 1e-9:
                break
            if difference > 0:
                adjustable = targets < self.max_split_s - 1e-9
            else:
                adjustable = targets > self.min_split_s + 1e-9
            count = int(adjustable.sum())
            if count == 0:
                break
            targets[adjustable] += difference / count
            targets = np.clip(targets, self.min_split_s, self.max_split_s)
        return targets.astype(np.float32)

    def _target_green(self, tls_id):
        runtime = self.env.runtime[tls_id]
        return float(self._phase_targets(tls_id)[runtime.phase_index])

    def _out_density(self, tls_id, phase_index):
        densities = []
        for lane_id in self.env.phase_out_lanes[tls_id][phase_index]:
            length = float(self.env.sumo.lane.getLength(lane_id))
            capacity = max(length / 7.5, 1.0)
            vehicles = float(self.env.sumo.lane.getLastStepVehicleNumber(lane_id))
            densities.append(vehicles / capacity)
        return min(1.0, max(densities, default=0.0))

    def _platoon_counts(self, tls_id):
        counts = np.zeros(self.env.NUM_PHASES, dtype=np.int32)
        vehicle_api = self.env.sumo.vehicle
        for vehicle_id in vehicle_api.getIDList():
            road_id = vehicle_api.getRoadID(vehicle_id)
            relation = self.directed_edges.get(road_id)
            if relation is None or relation[1] != tls_id:
                continue
            speed = float(vehicle_api.getSpeed(vehicle_id))
            if speed <= 0:
                continue
            phase_index = self._phase_for_lane(tls_id, vehicle_api.getLaneID(vehicle_id))
            if phase_index is None:
                continue
            for target, _, distance, _ in vehicle_api.getNextTLS(vehicle_id):
                if target == tls_id and 0 <= float(distance) / speed <= self.max_eta_s:
                    counts[phase_index] += 1
                break
        return counts

    def _safe_next_phase(self, tls_id):
        runtime = self.env.runtime[tls_id]
        counts = self._platoon_counts(tls_id)
        eligible = []
        for phase_index, count in enumerate(counts):
            if phase_index == runtime.phase_index or count < self.min_platoon_size:
                continue
            if self._out_density(tls_id, phase_index) >= self.spillback_threshold:
                self.stats["spillback_veto"] += 1
                continue
            eligible.append((int(count), phase_index))
        if eligible:
            self.stats["platoon_phase"] += 1
            return max(eligible)[1]
        if counts.max(initial=0) < self.min_platoon_size:
            self.stats["small_platoon_veto"] += 1

        demands = np.asarray(self.env._phase_demands(tls_id), dtype=np.float32)
        waits = np.asarray(runtime.phase_wait_seconds, dtype=np.float32)
        scores = demands + waits / max(self.nominal_green_s, 1.0)
        scores[runtime.phase_index] = -np.inf
        for phase_index in range(self.env.NUM_PHASES):
            if self._out_density(tls_id, phase_index) >= self.spillback_threshold:
                scores[phase_index] = -np.inf
                self.stats["spillback_veto"] += 1
        if np.isfinite(scores).any():
            self.stats["pressure_phase"] += 1
            return int(np.argmax(scores))
        return None

    def select_action(self, tls_id):
        now = float(self.env.sumo.simulation.getTime())
        runtime = self.env.runtime[tls_id]
        elapsed = max(0.0, now - runtime.current_phase_started_at)
        mask = self.env.action_mask(tls_id, now)
        if self._network_queue() < self.min_network_queue:
            self.stats["fixed_fallback"] += 1
            action = self._fixed_action(tls_id, now)
            if bool(mask[action]):
                return action
            self.stats["masked_veto"] += 1
            return self.env.TERMINATE if bool(mask[self.env.TERMINATE]) else self.env.HOLD

        if elapsed + 1e-9 < self._target_green(tls_id) and bool(mask[self.env.HOLD]):
            self.stats["budget_hold"] += 1
            return self.env.HOLD
        if not bool(mask[self.env.TERMINATE]):
            self.stats["masked_veto"] += 1
            return self.env.HOLD

        next_phase = self._safe_next_phase(tls_id)
        self.stats["budget_terminate"] += 1
        if next_phase is None:
            return self.env.TERMINATE
        return {"action": self.env.TERMINATE, "next_phase": next_phase}
