import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from environment.traffic_signal import TrafficSignal
from tools.e6_flow_profiles import (
    cleanup_scaled_route_file,
    make_scaled_route_file,
    validate_e6_profile,
)


@dataclass
class _TLSRuntime:
    phase_index: int = 0
    current_phase_started_at: float = 0.0
    green_elapsed: float = 0.0
    yellow_until: Optional[float] = None
    next_decision_time: float = 0.0
    last_switch_time: float = 0.0
    phase_wait_steps: Optional[np.ndarray] = None
    phase_wait_seconds: Optional[np.ndarray] = None
    pending_phase_index: Optional[int] = None
    last_current_demand_at: float = 0.0


class AsyncMultiIntersectionEnv:
    HOLD = 0
    TERMINATE = 1
    ACTION_NAMES = {HOLD: "HOLD", TERMINATE: "TERMINATE"}
    ORIGINAL_STATE_DIM = 32
    NUM_PHASES = 4
    ENHANCED_STATE_DIM = 4 + 1 + 1 + 1 + 1 + 4 + 4
    observation_dim = ORIGINAL_STATE_DIM + ENHANCED_STATE_DIM

    def __init__(
        self,
        net_file=None,
        route_file=None,
        min_green=10,
        max_green=40,
        passage=3,
        yellow=2,
        extension_step=2,
        gamma=0.99,
        co2_reward_scale=1e-6,
        use_gui=False,
        max_step_seconds=120,
        sumo_module=None,
        sumolib_module=None,
        intensity=1.0,
        direction_profile=None,
        flow_groups=None,
    ):
        root = Path(__file__).resolve().parents[1]
        self.net_file = str(net_file or root / "nets/2x2-grid/2x2.net.xml")
        self.route_file = str(route_file or root / "nets/2x2-grid/2x2-medium.rou.xml")
        self.intensity = float(intensity)
        self.direction_profile = direction_profile
        self.flow_groups = flow_groups
        validate_e6_profile(self.intensity, self.direction_profile, self.flow_groups)
        self.scaled_route_path = self.route_file
        self._scaled_route_path_is_temporary = False
        self._prepare_scaled_route_file()
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.passage = float(passage)
        self.yellow = float(yellow)
        self.extension_step = float(extension_step)
        self.gamma = float(gamma)
        self.co2_reward_scale = float(co2_reward_scale)
        if self.co2_reward_scale < 0.0:
            raise ValueError("co2_reward_scale must be non-negative")
        self.use_gui = bool(use_gui)
        self.max_step_seconds = int(max_step_seconds)
        self.sumo = sumo_module
        self.sumolib = sumolib_module
        self.tls_ids = []
        self.signals = {}
        self.runtime: Dict[str, _TLSRuntime] = {}
        self.phase_lanes = {}
        self.controlled_incoming_lanes = []
        self.simulation_time = 600.0
        self.episode_start_time = 0.0
        self._closed = True
        self._last_obs = None
        self._last_global_obs = None
        self._vehicle_waiting = {}
        self._phase_trace = []
        self.departure_times = {}
        self.travel_times = []
        self.throughput = 0
        self.co2_total_mg = 0.0
        self.queue_integral_vehicle_seconds = 0.0
        self.waiting_increment_s = 0.0
        self.departed_count = 0
        self.completed_count = 0

    def _prepare_scaled_route_file(self):
        if self._scaled_route_path_is_temporary:
            cleanup_scaled_route_file(self.scaled_route_path)
        self.scaled_route_path = self.route_file
        self._scaled_route_path_is_temporary = False
        if (self.intensity != 1.0 or self.direction_profile is not None
                or self.flow_groups is not None):
            self.scaled_route_path = str(make_scaled_route_file(
                self.route_file,
                intensity=self.intensity,
                direction_profile=self.direction_profile,
                flow_groups=self.flow_groups,
            ))
            self._scaled_route_path_is_temporary = True

    def _load_sumo(self):
        if self.sumo is not None and self.sumolib is not None:
            return
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise RuntimeError("SUMO_HOME must be set before reset()")
        tools = str(Path(sumo_home) / "tools")
        if tools not in sys.path:
            sys.path.append(tools)
        import traci
        import sumolib
        self.sumo = traci
        self.sumolib = sumolib

    @staticmethod
    def duration_discount(gamma, duration_seconds):
        return float(gamma) ** float(duration_seconds)

    @staticmethod
    def waiting_increment_from_snapshots(previous, current):
        return float(sum(max(0.0, float(wait) - float(previous.get(veh_id, 0.0)))
                         for veh_id, wait in current.items()))

    def reset(self, seed=None, simulation_time=600):
        self.close()
        self._prepare_scaled_route_file()
        self._load_sumo()
        self.simulation_time = float(simulation_time)
        binary = self.sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")
        command = [binary, "-n", self.net_file, "-r", self.scaled_route_path,
                   "--time-to-teleport", "1000", "--no-step-log", "true"]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        self.sumo.start(command)
        self._closed = False
        self.tls_ids = sorted(self.sumo.trafficlight.getIDList())
        if len(self.tls_ids) != 4:
            self.close()
            raise ValueError(f"expected 4 TLS, found {len(self.tls_ids)}: {self.tls_ids}")

        self.signals = {}
        self.runtime = {}
        self.phase_lanes = {}
        self.phase_out_lanes = {}
        controlled_incoming_lanes = set()
        now = float(self.sumo.simulation.getTime())
        for tls_id in self.tls_ids:
            signal = TrafficSignal(tls_id, int(self.yellow), self.simulation_time, 10, self.sumo)
            if len(signal.all_green_phases) != self.NUM_PHASES:
                self.close()
                raise ValueError(f"{tls_id} expected 4 green phases, found {len(signal.all_green_phases)}")
            signal.green_phase = signal.all_green_phases[0]
            signal.yellow_phase = None
            self.signals[tls_id] = signal
            self.phase_lanes[tls_id] = self._build_phase_lanes(tls_id, signal)
            self.phase_out_lanes[tls_id] = self._build_phase_out_lanes(tls_id, signal)
            for link_group in self.sumo.trafficlight.getControlledLinks(tls_id):
                controlled_incoming_lanes.update(
                    link[0] for link in link_group if link and link[0])
            self.runtime[tls_id] = _TLSRuntime(
                phase_index=0,
                current_phase_started_at=now,
                green_elapsed=0.0,
                yellow_until=None,
                next_decision_time=now + self.min_green,
                last_switch_time=now,
                phase_wait_steps=np.zeros(self.NUM_PHASES, dtype=np.float32),
                phase_wait_seconds=np.zeros(self.NUM_PHASES, dtype=np.float32),
                pending_phase_index=None,
                last_current_demand_at=now,
            )
            self.sumo.trafficlight.setRedYellowGreenState(tls_id, signal.all_green_phases[0].state)

        self.controlled_incoming_lanes = sorted(controlled_incoming_lanes)
        self.episode_start_time = now
        self._vehicle_waiting = self._waiting_snapshot()
        self._phase_trace = []
        self.departure_times = {}
        self.travel_times = []
        self.throughput = 0
        self.co2_total_mg = 0.0
        self.queue_integral_vehicle_seconds = 0.0
        self.waiting_increment_s = 0.0
        self.departed_count = 0
        self.completed_count = 0
        observations, global_observation = self._compute_observations()
        self._last_obs = observations
        self._last_global_obs = global_observation
        return observations, global_observation

    def _build_phase_lanes(self, tls_id, signal):
        links = self.sumo.trafficlight.getControlledLinks(tls_id)
        result = []
        for phase in signal.all_green_phases:
            lanes = []
            for index, state in enumerate(phase.state):
                if state not in "Gg" or index >= len(links):
                    continue
                for link in links[index]:
                    if link and link[0] not in lanes:
                        lanes.append(link[0])
            result.append(lanes)
        return result

    def _build_phase_out_lanes(self, tls_id, signal):
        links = self.sumo.trafficlight.getControlledLinks(tls_id)
        result = []
        for phase in signal.all_green_phases:
            lanes = []
            for index, state in enumerate(phase.state):
                if state not in "Gg" or index >= len(links):
                    continue
                for link in links[index]:
                    if link and len(link) > 1 and link[1] not in lanes:
                        lanes.append(link[1])
            result.append(lanes)
        return result

    def _phase_demands(self, tls_id):
        demands = []
        for lanes in self.phase_lanes[tls_id]:
            demands.append(sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes))
        return np.asarray(demands, dtype=np.float32)

    def _compute_observations(self):
        observations = {tls_id: self._local_observation(tls_id) for tls_id in self.tls_ids}
        global_observation = np.concatenate([observations[tls_id] for tls_id in self.tls_ids]).astype(np.float32)
        return observations, global_observation

    def _local_observation(self, tls_id):
        original = np.asarray(self.signals[tls_id].compute_state(), dtype=np.float32)
        if original.shape != (self.ORIGINAL_STATE_DIM,):
            raise ValueError(f"{tls_id} original state must be 32-D, got {original.shape}")
        runtime = self.runtime[tls_id]
        now = float(self.sumo.simulation.getTime())
        runtime.green_elapsed = 0.0 if runtime.yellow_until is not None else max(0.0, now - runtime.current_phase_started_at)
        demands = self._phase_demands(tls_id)
        one_hot = np.zeros(self.NUM_PHASES, dtype=np.float32)
        one_hot[runtime.phase_index] = 1.0
        enhanced = np.concatenate([
            one_hot,
            np.asarray([min(1.0, runtime.green_elapsed / max(self.max_green, 1.0))], dtype=np.float32),
            np.asarray([float(runtime.yellow_until is not None)], dtype=np.float32),
            np.asarray([max(0.0, self.min_green - runtime.green_elapsed) / max(self.min_green, 1.0)], dtype=np.float32),
            np.asarray([max(0.0, self.max_green - runtime.green_elapsed) / max(self.max_green, 1.0)], dtype=np.float32),
            np.clip(demands / 100.0, 0.0, 1.0),
            np.clip(runtime.phase_wait_seconds / max(self.max_green, 1.0), 0.0, 1.0),
        ]).astype(np.float32)
        state = np.concatenate([original, enhanced]).astype(np.float32)
        if state.shape != (self.observation_dim,):
            raise AssertionError(f"unexpected observation shape {state.shape}")
        return state

    def is_ready(self, tls_id, now=None):
        runtime = self.runtime[tls_id]
        current = float(self.sumo.simulation.getTime()) if now is None else float(now)
        return runtime.yellow_until is None and current + 1e-9 >= runtime.next_decision_time

    def ready_tls(self):
        return [tls_id for tls_id in self.tls_ids if self.is_ready(tls_id)]

    def action_mask(self, tls_id, now=None):
        runtime = self.runtime[tls_id]
        current = float(self.sumo.simulation.getTime()) if now is None else float(now)
        elapsed = 0.0 if runtime.yellow_until is not None else max(0.0, current - runtime.current_phase_started_at)
        if runtime.yellow_until is not None:
            return np.asarray([False, False], dtype=bool)
        if elapsed < self.min_green:
            return np.asarray([True, False], dtype=bool)
        if elapsed >= self.max_green:
            return np.asarray([False, True], dtype=bool)
        demands = self._phase_demands(tls_id)
        gap_out = (demands[runtime.phase_index] <= 0 and
                   demands.sum() > 0 and
                   current - runtime.last_current_demand_at >= self.passage)
        return np.asarray([not gap_out, True], dtype=bool)

    def _choose_next_phase(self, tls_id):
        runtime = self.runtime[tls_id]
        demands = self._phase_demands(tls_id)
        for offset in range(1, self.NUM_PHASES + 1):
            candidate = (runtime.phase_index + offset) % self.NUM_PHASES
            if demands[candidate] > 0:
                return candidate
        return (runtime.phase_index + 1) % self.NUM_PHASES

    def _yellow_state(self, old_state, new_state):
        return "".join("y" if old in "Gg" and new == "r" else old
                       for old, new in zip(old_state, new_state))

    def _parse_action_command(self, command):
        if isinstance(command, dict):
            action = command.get("action", self.HOLD)
            next_phase = command.get("next_phase")
        else:
            action = command
            next_phase = None
        try:
            action = int(action)
        except (TypeError, ValueError):
            action = -1
        if next_phase is not None:
            try:
                next_phase = int(next_phase)
            except (TypeError, ValueError):
                next_phase = None
        return action, next_phase

    def _execute_action(self, tls_id, requested_action, now):
        runtime = self.runtime[tls_id]
        requested_action, requested_phase = self._parse_action_command(requested_action)
        mask = self.action_mask(tls_id, now)
        valid_action = (
            requested_action in (self.HOLD, self.TERMINATE)
            and bool(mask[requested_action])
        )
        if not valid_action:
            requested_action = self.TERMINATE if mask[self.TERMINATE] else self.HOLD
        if requested_action == self.HOLD:
            runtime.next_decision_time = min(now + self.extension_step,
                                             runtime.current_phase_started_at + self.max_green)
            return self.HOLD, False, valid_action

        valid_phase = (
            requested_phase is None
            or (
                0 <= requested_phase < self.NUM_PHASES
                and requested_phase != runtime.phase_index
            )
        )
        next_phase = (
            requested_phase
            if valid_phase and requested_phase is not None
            else self._choose_next_phase(tls_id)
        )
        valid = valid_action and valid_phase
        old_state = self.signals[tls_id].all_green_phases[runtime.phase_index].state
        new_state = self.signals[tls_id].all_green_phases[next_phase].state
        self.sumo.trafficlight.setRedYellowGreenState(tls_id, self._yellow_state(old_state, new_state))
        runtime.yellow_until = now + self.yellow
        runtime.pending_phase_index = next_phase
        runtime.next_decision_time = float("inf")
        runtime.last_switch_time = now
        return self.TERMINATE, True, valid

    def _finish_yellows(self, now):
        for tls_id in self.tls_ids:
            runtime = self.runtime[tls_id]
            if runtime.yellow_until is None or now + 1e-9 < runtime.yellow_until:
                continue
            runtime.phase_index = int(runtime.pending_phase_index)
            runtime.pending_phase_index = None
            runtime.yellow_until = None
            runtime.current_phase_started_at = now
            runtime.green_elapsed = 0.0
            runtime.next_decision_time = now + self.min_green
            runtime.last_switch_time = now
            runtime.last_current_demand_at = now
            self.sumo.trafficlight.setRedYellowGreenState(
                tls_id, self.signals[tls_id].all_green_phases[runtime.phase_index].state)

    def _update_phase_wait(self, dt):
        for tls_id in self.tls_ids:
            runtime = self.runtime[tls_id]
            demands = self._phase_demands(tls_id)
            for phase_index, demand in enumerate(demands):
                served = runtime.yellow_until is None and phase_index == runtime.phase_index
                if demand > 0 and not served:
                    runtime.phase_wait_steps[phase_index] += 1
                    runtime.phase_wait_seconds[phase_index] += dt
                elif served:
                    runtime.phase_wait_steps[phase_index] = 0
                    runtime.phase_wait_seconds[phase_index] = 0
            if runtime.yellow_until is None and demands[runtime.phase_index] > 0:
                runtime.last_current_demand_at = float(self.sumo.simulation.getTime())

    def _waiting_snapshot(self):
        return {veh_id: float(self.sumo.vehicle.getWaitingTime(veh_id))
                for veh_id in self.sumo.vehicle.getIDList()}

    def _record_vehicle_events(self, now):
        departed_ids = list(self.sumo.simulation.getDepartedIDList())
        arrived_ids = list(self.sumo.simulation.getArrivedIDList())
        for veh_id in departed_ids:
            self.departure_times[veh_id] = float(now)
            self.departed_count += 1
        for veh_id in arrived_ids:
            departed_at = self.departure_times.pop(veh_id, None)
            if departed_at is not None:
                self.travel_times.append(float(now) - departed_at)
            self.throughput += 1
            self.completed_count += 1
        return len(arrived_ids)

    def _terminal_halting_vehicles(self):
        return float(sum(self.sumo.lane.getLastStepHaltingNumber(lane)
                         for lane in self.controlled_incoming_lanes))

    def get_episode_metrics(self):
        now = float(self.sumo.simulation.getTime())
        duration = max(0.0, now - self.episode_start_time)
        departed = int(self.departed_count)
        att_mean = float(np.mean(self.travel_times)) if self.travel_times else None
        return {
            "throughput": int(self.throughput),
            "departed": departed,
            "completed": int(self.completed_count),
            "unfinished": int(len(self.departure_times)),
            "att_mean_s": att_mean,
            "att_sample_n": int(len(self.travel_times)),
            "co2_kg": float(self.co2_total_mg / 1_000_000.0),
            "time_average_queue_vehicles": (
                float(self.queue_integral_vehicle_seconds / duration) if duration > 0 else 0.0),
            "total_waiting_increment_s": float(self.waiting_increment_s),
            "waiting_per_departed_vehicle_s": (
                float(self.waiting_increment_s / departed) if departed > 0 else None),
            "terminal_halting_vehicles": self._terminal_halting_vehicles(),
            "episode_duration_s": float(duration),
        }

    def step(self, actions):
        if self._closed:
            raise RuntimeError("reset() must be called before step()")
        actions = actions or {}
        start_time = float(self.sumo.simulation.getTime())
        previous_obs = self._last_obs
        ready_at_start = self.ready_tls()
        action_masks = {tls_id: self.action_mask(tls_id, start_time).copy()
                        for tls_id in ready_at_start}
        executed_actions = {}
        valid_transition = True
        switch_count = 0
        phase_trace = []

        for tls_id in ready_at_start:
            requested = actions.get(tls_id, self.HOLD)
            executed, switched, valid = self._execute_action(tls_id, requested, start_time)
            executed_actions[tls_id] = executed
            switch_count += int(switched)
            valid_transition = valid_transition and valid
            phase_trace.append({"time": start_time, "tls_id": tls_id,
                                "event": self.ACTION_NAMES[executed],
                                "phase": self.runtime[tls_id].phase_index})

        arrived_delta = 0
        halting_integral = 0.0
        waiting_increment = 0.0
        co2_mg = 0.0
        elapsed = 0
        while elapsed < self.max_step_seconds:
            self.sumo.simulationStep()
            elapsed += 1
            now = float(self.sumo.simulation.getTime())
            self._finish_yellows(now)
            self._update_phase_wait(1.0)
            arrived_delta += self._record_vehicle_events(now)
            second_halting = self._terminal_halting_vehicles()
            halting_integral += second_halting
            self.queue_integral_vehicle_seconds += second_halting
            current_waiting = self._waiting_snapshot()
            second_waiting_increment = self.waiting_increment_from_snapshots(
                self._vehicle_waiting, current_waiting)
            waiting_increment += second_waiting_increment
            self.waiting_increment_s += second_waiting_increment
            self._vehicle_waiting = current_waiting
            second_co2_mg = sum(float(self.sumo.vehicle.getCO2Emission(veh_id))
                                for veh_id in self.sumo.vehicle.getIDList())
            co2_mg += second_co2_mg
            self.co2_total_mg += second_co2_mg
            if self.ready_tls() or self._terminated(now):
                break

        end_time = float(self.sumo.simulation.getTime())
        duration = end_time - start_time
        terminated = self._terminated(end_time)
        next_obs, global_obs = self._compute_observations()
        reward_components = {
            "arrived": float(arrived_delta),
            "halting": -0.05 * float(halting_integral),
            "waiting": -0.001 * float(waiting_increment),
            "co2": -self.co2_reward_scale * float(co2_mg),
            "switch": -0.1 * float(switch_count),
        }
        reward = float(sum(reward_components.values()))
        ready_next = self.ready_tls()
        ready_mask = {tls_id: tls_id in ready_next for tls_id in self.tls_ids}
        transition = {
            "obs": previous_obs,
            "action": dict(executed_actions),
            "next_obs": next_obs,
            "reward": reward,
            "reward_components": dict(reward_components),
            "terminated": terminated,
            "truncated": False,
            "duration_seconds": duration,
            "duration_discount": self.duration_discount(self.gamma, duration),
            "next_goal": None,
            "goal_boundary": True,
            "ready_mask": ready_mask,
            "acted_tls": list(ready_at_start),
            "action_mask": {tls_id: mask.tolist()
                            for tls_id, mask in action_masks.items()},
            "neighbor_message_timestamps": [],
        }
        phase_trace.extend(self._current_phase_trace(end_time))
        self._phase_trace.extend(phase_trace)
        info = {
            "decision_dt": duration,
            "ready_tls": ready_next,
            "executed_actions": executed_actions,
            "switch_count": switch_count,
            "arrived_delta": arrived_delta,
            "halting_integral": halting_integral,
            "waiting_increment": waiting_increment,
            "co2_mg": co2_mg,
            "duration_discount_exponent": duration,
            "duration_discount": self.duration_discount(self.gamma, duration),
            "valid_transition": valid_transition,
            "phase_trace": phase_trace,
            "reward_components": reward_components,
            "transition": transition,
        }
        self._last_obs = next_obs
        self._last_global_obs = global_obs
        return next_obs, global_obs, reward, terminated, info

    def _current_phase_trace(self, now):
        return [{"time": now, "tls_id": tls_id,
                 "event": "YELLOW" if self.runtime[tls_id].yellow_until is not None else "GREEN",
                 "phase": self.runtime[tls_id].phase_index}
                for tls_id in self.tls_ids]

    def _terminated(self, now=None):
        current = float(self.sumo.simulation.getTime()) if now is None else float(now)
        return current - self.episode_start_time >= self.simulation_time

    def close(self):
        if self.sumo is not None and not self._closed:
            try:
                self.sumo.close()
            except Exception:
                pass
        if self._scaled_route_path_is_temporary:
            cleanup_scaled_route_file(self.scaled_route_path)
            self.scaled_route_path = self.route_file
            self._scaled_route_path_is_temporary = False
        self._closed = True
