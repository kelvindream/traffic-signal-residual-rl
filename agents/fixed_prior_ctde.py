"""Gate 0 fixed-prior CTDE agent for asynchronous binary signal control."""

import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent_config import AgentConfig, legacy_config_from_agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    "Transition",
    ("global_state", "goals", "joint_action_index", "next_global_state", "reward", "done", "discount", "co2_cost", "throughput_cost", "episode_cost_scale", "final_nonmask_residual_override"),
    defaults=(0.0, 0.0, 1.0, False),
)


class GoalNetwork(nn.Module):
    def __init__(self, global_state_dim, total_goal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_goal_dim),
            nn.Sigmoid(),
        )

    def forward(self, state):
        return self.net(state)


class FixedPriorResidualWorker(nn.Module):
    def __init__(self, local_state_dim, local_goal_dim, action_dim,
                 residual_logit_limit=6.0):
        super().__init__()
        self.residual_logit_limit = float(residual_logit_limit)
        self.backbone = nn.Sequential(
            nn.Linear(local_state_dim + local_goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.residual_head = nn.Linear(128, action_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, local_states, local_goals):
        features = torch.cat((local_states, local_goals), dim=-1)
        raw_logits = self.residual_head(self.backbone(features))
        limit = self.residual_logit_limit
        return limit * torch.tanh(raw_logits / limit)


class PressureFeatureProjector(nn.Module):
    """Parameter-free projector that exposes explicit pressure features.

    Gate 16A verified that pressure information is recoverable from the local
    state but is not efficiently exploited by the worker MLP. This projector
    makes the per-intersection and cross-intersection pressure signals explicit
    so the residual policy can use them without relying on the goal network to
    propagate them downstream.

    The projector is parameter-free to isolate the effect of information
    availability from capacity changes. When enabled, the worker receives the
    original local state concatenated with the 5-D pressure features per
    intersection; goal network and critic inputs are unchanged.

    Output features (per intersection, 5-D):
        1. intersection_total_demand: sum of the 4 phase demands
        2. max_phase_pressure: max of the 4 phase demands
        3. current_phase_pressure: demand of the active phase
        4. unserved_current_pressure: total - current (pressure on non-active phases)
        5. global_demand_imbalance: max_total - min_total across intersections
           (broadcast to every intersection so the worker sees the global signal)
    """

    PRESSURE_DIM = 5
    ENHANCED_SUFFIX_DIM = 16
    PHASE_DEMAND_OFFSET_IN_SUFFIX = 8
    PHASE_ONEHOT_OFFSET_IN_SUFFIX = 0
    PHASE_DEMAND_DIM = 4

    def __init__(self, num_intersections=4, local_state_dim=48):
        super().__init__()
        self.num_intersections = int(num_intersections)
        self.local_state_dim = int(local_state_dim)
        if self.local_state_dim < self.ENHANCED_SUFFIX_DIM:
            raise ValueError(
                f"local_state_dim must include the {self.ENHANCED_SUFFIX_DIM}-D "
                "enhanced suffix to extract pressure features")
        self.enhanced_start = self.local_state_dim - self.ENHANCED_SUFFIX_DIM
        self.phase_onehot_offset = self.enhanced_start + self.PHASE_ONEHOT_OFFSET_IN_SUFFIX
        self.phase_demand_offset = self.enhanced_start + self.PHASE_DEMAND_OFFSET_IN_SUFFIX
        if self.phase_demand_offset + self.PHASE_DEMAND_DIM > self.local_state_dim:
            raise ValueError("phase demand slice exceeds local_state_dim")

    def forward(self, local_states):
        if local_states.shape[-1] != self.local_state_dim:
            raise ValueError(
                f"expected local_states last dim = {self.local_state_dim}, "
                f"got {local_states.shape[-1]}")
        phase_onehot = local_states[
            ..., self.phase_onehot_offset:self.phase_onehot_offset + self.PHASE_DEMAND_DIM]
        phase_demands = local_states[
            ..., self.phase_demand_offset:self.phase_demand_offset + self.PHASE_DEMAND_DIM]
        phase_index = phase_onehot.argmax(dim=-1, keepdim=True)
        current = phase_demands.gather(-1, phase_index).squeeze(-1)
        total = phase_demands.sum(dim=-1)
        max_p = phase_demands.max(dim=-1).values
        unserved = total - current
        imbalance = total.max(dim=-1).values - total.min(dim=-1).values
        imbalance_broadcast = imbalance.unsqueeze(-1).expand_as(total)
        features = torch.stack(
            [total, max_p, current, unserved, imbalance_broadcast], dim=-1)
        return features


class CentralTwinCritic(nn.Module):
    def __init__(self, global_state_dim, total_goal_dim, joint_action_count):
        super().__init__()
        input_dim = global_state_dim + total_goal_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, joint_action_count),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, joint_action_count),
        )

    def forward(self, global_state, goals):
        features = torch.cat((global_state, goals), dim=-1)
        return self.q1(features), self.q2(features)


class FixedPriorCTDEAgent:
    HOLD = 0
    TERMINATE = 1

    def __init__(self, config, device_override=None):
        if not isinstance(config, AgentConfig):
            raise TypeError(
                "FixedPriorCTDEAgent expects an AgentConfig instance; "
                "construct it via agents.agent_config.AgentConfig")
        self.config = config
        self.num_intersections = config.num_intersections
        self.local_state_dim = config.local_state_dim
        self.local_goal_dim = config.local_goal_dim
        self.action_dim = config.action_dim
        self.global_state_dim = self.num_intersections * self.local_state_dim
        self.total_goal_dim = self.num_intersections * self.local_goal_dim
        self.joint_action_count = self.action_dim ** self.num_intersections
        self.gamma = config.gamma
        self.lr = config.lr
        self.constraint_co2_budget = config.constraint_co2_episode_budget
        self.constraint_throughput_target = config.constraint_throughput_episode_target
        self.constraint_scope = "episode"
        self.lagrange_lr = config.lagrange_lr
        self.max_lambda = config.max_lambda
        self.co2_cost_scale = config.co2_cost_scale
        self.lambda_co2 = config.lambda_co2
        self.lambda_throughput = config.lambda_throughput
        self._episode_co2_cost = 0.0
        self._episode_throughput_cost = 0.0
        self._episode_transition_count = 0
        self.calibrated_decisions = None
        self.last_episode_constraint = {
            "co2_violation": 0.0,
            "throughput_violation": 0.0,
            "episode_co2_cost": 0.0,
            "episode_throughput_cost": 0.0,
        }
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.tau = config.tau
        self.alpha = config.alpha
        self.prior_logit = config.prior_logit
        self.prior_mode = config.prior_mode
        self.intervention_penalty = config.intervention_penalty
        self.residual_logit_limit = config.residual_logit_limit
        self.target_intervention_rate = config.target_intervention_rate
        self.gate_min_global_demand = config.gate_min_global_demand
        self.gate_min_active_intersections = config.gate_min_active_intersections
        self.gate_asymmetry_threshold = config.gate_asymmetry_threshold
        self.safety_gate_enabled = config.safety_gate_enabled
        self.override_min_probability = config.override_min_probability
        self.shield_min_elapsed_ratio = config.shield_min_elapsed_ratio
        self.shield_max_current_to_competing_ratio = (
            config.shield_max_current_to_competing_ratio)
        self.device = torch.device(device_override) if device_override is not None else device
        self.enable_residual = config.enable_residual
        self.enable_goal = config.enable_goal
        self.enable_pressure_features = config.enable_pressure_features
        self.enable_advantage_residual = config.enable_advantage_residual
        self.enable_override_prioritized_replay = (
            config.enable_override_prioritized_replay)
        self._last_action_metadata = None
        if self.enable_pressure_features:
            self.pressure_projector = PressureFeatureProjector(
                num_intersections=self.num_intersections,
                local_state_dim=self.local_state_dim,
            ).to(self.device)
            self.worker_local_state_dim = (
                self.local_state_dim + PressureFeatureProjector.PRESSURE_DIM)
        else:
            self.pressure_projector = None
            self.worker_local_state_dim = self.local_state_dim
        self.learn_steps = 0
        self.episode_count = 0
        self.episode_horizon = 1
        self.reset_action_stats()

        self.goal_network = GoalNetwork(self.global_state_dim, self.total_goal_dim).to(self.device)
        self.worker_network = FixedPriorResidualWorker(
            self.worker_local_state_dim, self.local_goal_dim, self.action_dim,
            residual_logit_limit=self.residual_logit_limit).to(self.device)
        self.critic_network = CentralTwinCritic(
            self.global_state_dim, self.total_goal_dim, self.joint_action_count).to(self.device)
        self.target_critic_network = CentralTwinCritic(
            self.global_state_dim, self.total_goal_dim, self.joint_action_count).to(self.device)
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        self.goal_optimizer = optim.Adam(self.goal_network.parameters(), lr=self.lr)
        self.worker_optimizer = optim.Adam(self.worker_network.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.lr)
        self.replay_buffer = []
        self.buffer_ptr = 0
        self._joint_actions = torch.tensor(
            [self.decode_joint_action(i) for i in range(self.joint_action_count)],
            dtype=torch.long, device=self.device)

    def reset_action_stats(self):
        self._action_stats = {
            "prior_action": [0] * self.action_dim,
            "selected_action": [0] * self.action_dim,
            "residual_override": 0,
            "policy_residual_override": 0,
            "policy_nonmask_override": 0,
            "confidence_veto": 0,
            "shield_veto": 0,
            "shield_eligible": 0,
            "override_hold_to_terminate": 0,
            "override_terminate_to_hold": 0,
            "nonmask_residual_override": 0,
            "final_nonmask_residual_override": 0,
            "mask_veto": 0,
            "gate_open_actions": 0,
            "gate_closed_actions": 0,
            "gate_low_demand": 0,
            "gate_low_activity": 0,
            "total_actions": 0,
            "expected_intervention_sum": 0.0,
        }

    def get_action_stats(self):
        stats = dict(self._action_stats)
        stats["prior_action"] = list(stats["prior_action"])
        stats["selected_action"] = list(stats["selected_action"])
        total = stats["total_actions"]
        stats["residual_intervention_rate"] = (
            float(stats["residual_override"]) / total if total else 0.0)
        stats["nonmask_residual_override_rate"] = (
            float(stats["nonmask_residual_override"]) / total if total else 0.0)
        stats["policy_nonmask_override_rate"] = (
            float(stats["policy_nonmask_override"]) / total if total else 0.0)
        stats["confidence_veto_rate"] = (
            float(stats["confidence_veto"]) / total if total else 0.0)
        stats["final_nonmask_residual_override_rate"] = (
            float(stats["final_nonmask_residual_override"]) / total if total else 0.0)
        stats["final_nonmask_rate"] = stats["final_nonmask_residual_override_rate"]
        stats["shield_veto_rate"] = (
            float(stats["shield_veto"]) / total if total else 0.0)
        stats["shield_eligible_rate"] = (
            float(stats["shield_eligible"]) / total if total else 0.0)
        stats["gate_open_rate"] = (
            float(stats["gate_open_actions"]) / total if total else 0.0)
        stats["expected_intervention_rate"] = (
            float(stats.pop("expected_intervention_sum")) / total if total else 0.0)
        excess = max(0.0, stats["expected_intervention_rate"] - self.target_intervention_rate)
        stats["intervention_loss"] = self.intervention_penalty * excess * excess
        return stats

    def get_last_action_metadata(self):
        if self._last_action_metadata is None:
            return None
        return dict(self._last_action_metadata)

    def consume_last_action_metadata(self):
        metadata = self.get_last_action_metadata()
        self._last_action_metadata = None
        return metadata

    def save(self, path):
        torch.save({
            "config": legacy_config_from_agent(self),
            "goal_network": self.goal_network.state_dict(),
            "worker_network": self.worker_network.state_dict(),
            "critic_network": self.critic_network.state_dict(),
            "target_critic_network": self.target_critic_network.state_dict(),
            "goal_optimizer": self.goal_optimizer.state_dict(),
            "worker_optimizer": self.worker_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "learn_steps": self.learn_steps,
            "episode_count": self.episode_count,
        }, path)

    def load(self, path, map_location=None):
        location = self.device if map_location is None else map_location
        checkpoint = torch.load(path, map_location=location, weights_only=False)
        config = checkpoint.get("config", {})
        self.prior_logit = float(config.get("prior_logit", self.prior_logit))
        checkpoint_prior_mode = config.get("prior_mode", "fixed")
        if checkpoint_prior_mode != self.prior_mode:
            raise ValueError(
                f"checkpoint prior_mode={checkpoint_prior_mode} does not match "
                f"agent prior_mode={self.prior_mode}")
        self.constraint_co2_budget = config.get(
            "constraint_co2_episode_budget",
            config.get("constraint_co2_budget", self.constraint_co2_budget),
        )
        self.constraint_throughput_target = config.get(
            "constraint_throughput_episode_target",
            config.get("constraint_throughput_target", self.constraint_throughput_target),
        )
        self.constraint_scope = config.get("constraint_scope", self.constraint_scope)
        self.lagrange_lr = float(config.get("lagrange_lr", self.lagrange_lr))
        self.lambda_co2 = float(config.get("lambda_co2", self.lambda_co2))
        self.lambda_throughput = float(config.get("lambda_throughput", self.lambda_throughput))
        self.max_lambda = float(config.get("max_lambda", self.max_lambda))
        self.co2_cost_scale = float(config.get("co2_cost_scale", self.co2_cost_scale))
        self.calibrated_decisions = config.get("calibrated_decisions", None)
        self.intervention_penalty = float(
            config.get("intervention_penalty", self.intervention_penalty))
        self.residual_logit_limit = float(
            config.get("residual_logit_limit", self.residual_logit_limit))
        self.target_intervention_rate = float(
            config.get("target_intervention_rate", self.target_intervention_rate))
        self.gate_min_global_demand = float(
            config.get("gate_min_global_demand", self.gate_min_global_demand))
        self.gate_min_active_intersections = int(config.get(
            "gate_min_active_intersections", self.gate_min_active_intersections))
        self.gate_asymmetry_threshold = float(config.get(
            "gate_asymmetry_threshold", self.gate_asymmetry_threshold))
        self.safety_gate_enabled = bool(
            config.get("safety_gate_enabled", self.safety_gate_enabled))
        self.override_min_probability = float(
            config.get("override_min_probability", self.override_min_probability))
        self.shield_min_elapsed_ratio = config.get(
            "shield_min_elapsed_ratio", self.shield_min_elapsed_ratio)
        self.shield_max_current_to_competing_ratio = config.get(
            "shield_max_current_to_competing_ratio",
            self.shield_max_current_to_competing_ratio)
        if self.shield_min_elapsed_ratio is not None:
            self.shield_min_elapsed_ratio = float(self.shield_min_elapsed_ratio)
        if self.shield_max_current_to_competing_ratio is not None:
            self.shield_max_current_to_competing_ratio = float(
                self.shield_max_current_to_competing_ratio)
        for name, default in (
                ("enable_residual", True),
                ("enable_goal", True),
                ("enable_advantage_residual", False),
                ("enable_override_prioritized_replay", False)):
            checkpoint_value = bool(config.get(name, default))
            current_value = getattr(self, name)
            if checkpoint_value != current_value:
                raise ValueError(
                    f"checkpoint {name}={checkpoint_value} does not match "
                    f"agent {name}={current_value}")
        checkpoint_pressure_features = bool(
            config.get("enable_pressure_features", self.enable_pressure_features))
        if checkpoint_pressure_features != self.enable_pressure_features:
            raise ValueError(
                "checkpoint enable_pressure_features="
                f"{checkpoint_pressure_features} does not match agent "
                f"enable_pressure_features={self.enable_pressure_features}; "
                "construct the agent with the matching flag before loading")
        self.worker_network.residual_logit_limit = self.residual_logit_limit
        self.goal_network.load_state_dict(checkpoint["goal_network"])
        self.worker_network.load_state_dict(checkpoint["worker_network"])
        self.critic_network.load_state_dict(checkpoint["critic_network"])
        self.target_critic_network.load_state_dict(checkpoint["target_critic_network"])
        self.goal_optimizer.load_state_dict(checkpoint["goal_optimizer"])
        self.worker_optimizer.load_state_dict(checkpoint["worker_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
        self.episode_count = int(checkpoint.get("episode_count", 0))
        for optimizer in (
                self.goal_optimizer, self.worker_optimizer, self.critic_optimizer):
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)
        return self

    def encode_joint_action(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.size != self.num_intersections:
            raise ValueError("joint action has wrong number of intersections")
        index = 0
        for action in actions:
            if action < 0 or action >= self.action_dim:
                raise ValueError("joint action contains an invalid action")
            index = index * self.action_dim + int(action)
        return index

    def decode_joint_action(self, index):
        index = int(index)
        if index < 0 or index >= self.joint_action_count:
            raise ValueError("joint action index out of range")
        actions = [0] * self.num_intersections
        for position in range(self.num_intersections - 1, -1, -1):
            actions[position] = index % self.action_dim
            index //= self.action_dim
        return actions

    def _split(self, global_states, goals):
        return (
            global_states.reshape(-1, self.num_intersections, self.local_state_dim),
            goals.reshape(-1, self.num_intersections, self.local_goal_dim),
        )

    def _augment_local_states(self, local_states):
        if not self.enable_pressure_features:
            return local_states
        pressure_features = self.pressure_projector(local_states)
        return torch.cat([local_states, pressure_features], dim=-1)

    def _elapsed_ratio(self, local_states):
        if self.local_state_dim < 16:
            raise ValueError("local state does not contain the 16-D enhanced suffix")
        enhanced_start = self.local_state_dim - 16
        return local_states[..., enhanced_start + 4]

    @property
    def action_shield_enabled(self):
        return (self.shield_min_elapsed_ratio is not None
                and self.shield_max_current_to_competing_ratio is not None)

    def _shield_eligible_tensor(self, local_states):
        if not self.action_shield_enabled:
            return torch.ones(local_states.shape[:-1], dtype=torch.bool,
                              device=local_states.device)
        enhanced_start = self.local_state_dim - 16
        phases = local_states[..., enhanced_start:enhanced_start + 4]
        demands = local_states[..., enhanced_start + 8:enhanced_start + 12]
        current_phase = phases.argmax(dim=-1)
        current_demand = demands.gather(-1, current_phase.unsqueeze(-1)).squeeze(-1)
        competing_demands = demands.clone()
        competing_demands.scatter_(-1, current_phase.unsqueeze(-1), -1.0)
        max_competing_demand = competing_demands.max(dim=-1).values
        return (
            (self._elapsed_ratio(local_states) >= self.shield_min_elapsed_ratio)
            & (max_competing_demand > 0.0)
            & (current_demand <= (
                self.shield_max_current_to_competing_ratio * max_competing_demand))
        )

    def _gate_metrics_tensor(self, local_states):
        if self.local_state_dim < 16:
            raise ValueError("local state does not contain the 16-D enhanced suffix")
        demand_start = self.local_state_dim - 8
        intersection_demands = local_states[..., demand_start:demand_start + 4].sum(dim=-1)
        global_demand = intersection_demands.sum(dim=-1)
        active_intersections = (intersection_demands > 1e-8).sum(dim=-1)
        max_demand = intersection_demands.max(dim=-1).values
        min_demand = intersection_demands.min(dim=-1).values
        asymmetry = torch.where(
            max_demand > 1e-8,
            (max_demand - min_demand) / max_demand.clamp_min(1e-8),
            torch.zeros_like(max_demand),
        )
        low_demand = global_demand < self.gate_min_global_demand
        low_activity = active_intersections < self.gate_min_active_intersections
        asymmetric = asymmetry >= self.gate_asymmetry_threshold
        gate_open = (~low_demand) & ((~low_activity) | asymmetric)
        return gate_open, low_demand, low_activity, global_demand, active_intersections, asymmetry

    def _gate_open_tensor(self, states):
        local_states = states.reshape(-1, self.num_intersections, self.local_state_dim)
        return self._gate_metrics_tensor(local_states)[0]

    def safety_gate_open(self, global_state):
        state = torch.as_tensor(
            np.asarray(global_state, dtype=np.float32), device=self.device).reshape(1, -1)
        if state.shape[1] != self.global_state_dim:
            raise ValueError("global_state has wrong dimension")
        return bool(self._gate_open_tensor(state).item())

    def fixed_prior_actions(self, global_state):
        """兼容旧接口：返回当前 prior_mode 对应的先验动作。"""
        state = np.asarray(global_state, dtype=np.float32).reshape(
            self.num_intersections, self.local_state_dim)
        actions = self._prior_actions_tensor(torch.as_tensor(state))
        return actions.cpu().numpy().astype(np.int64)

    def _prior_actions_tensor(self, local_states):
        if self.prior_mode == "fixed":
            elapsed_ratio = self._elapsed_ratio(local_states)
            return (elapsed_ratio >= 1.0 - 1e-6).to(dtype=torch.long)
        # 与 tools/eval_rule_baselines_formal.py 的 actuated 语义一致：
        # 当前相位之外只要存在排队需求，就请求 TERMINATE；合法性由 action mask 强制。
        enhanced_start = self.local_state_dim - 16
        phases = local_states[..., enhanced_start:enhanced_start + 4]
        demands = local_states[..., enhanced_start + 8:enhanced_start + 12]
        current_phase = phases.argmax(dim=-1)
        current_demand = demands.gather(-1, current_phase.unsqueeze(-1)).squeeze(-1)
        competing_demand = demands.sum(dim=-1) - current_demand
        return (competing_demand > 1e-8).to(dtype=torch.long)

    def _prior_logits(self, local_states):
        preferred = self._prior_actions_tensor(local_states)
        logits = torch.full(
            (*local_states.shape[:-1], self.action_dim),
            -self.prior_logit, dtype=local_states.dtype, device=local_states.device)
        return logits.scatter(-1, preferred.unsqueeze(-1), self.prior_logit)

    def _expected_intervention(self, local_probs, local_states):
        prior_actions = self._prior_actions_tensor(local_states)
        prior_probabilities = local_probs.gather(
            -1, prior_actions.unsqueeze(-1)).squeeze(-1)
        return 1.0 - prior_probabilities

    @staticmethod
    def _normalize_masks(action_masks, num_intersections, action_dim):
        if action_masks is None:
            return np.ones((num_intersections, action_dim), dtype=bool)
        if isinstance(action_masks, dict):
            values = list(action_masks.values())
            action_masks = values
        masks = np.asarray(action_masks, dtype=bool)
        if masks.shape != (num_intersections, action_dim):
            raise ValueError("action_masks must have shape (num_intersections, action_dim)")
        if np.any(~masks.any(axis=1)):
            raise ValueError("each intersection must allow at least one action")
        return masks

    def select_goal(self, global_state, evaluate=False):
        del evaluate
        state = torch.as_tensor(
            np.asarray(global_state, dtype=np.float32), device=self.device).reshape(1, -1)
        if state.shape[1] != self.global_state_dim:
            raise ValueError("global_state has wrong dimension")
        if not self.enable_goal:
            return np.zeros(self.total_goal_dim, dtype=np.float32)
        with torch.no_grad():
            return self.goal_network(state).squeeze(0).cpu().numpy()

    def _local_probabilities(self, states, goals, masks=None):
        local_states, local_goals = self._split(states, goals)
        logits = self._prior_logits(local_states)
        if self.enable_residual:
            worker_inputs = self._augment_local_states(local_states)
            logits = logits + self.worker_network(worker_inputs, local_goals)
        if masks is not None:
            logits = logits.masked_fill(~masks, -1e9)
        return torch.softmax(logits, dim=-1)

    def select_action(self, global_state, goals=None, action_masks=None, evaluate=False,
                      apply_safety_gate=True):
        state_array = np.asarray(global_state, dtype=np.float32).reshape(-1)
        if state_array.size != self.global_state_dim:
            raise ValueError("global_state has wrong dimension")
        if goals is None:
            goals = self.select_goal(state_array, evaluate=evaluate)
        goal_array = np.asarray(goals, dtype=np.float32).reshape(-1)
        if goal_array.size != self.total_goal_dim:
            raise ValueError("goals has wrong dimension")
        mask_array = self._normalize_masks(
            action_masks, self.num_intersections, self.action_dim)
        prior_actions = self.fixed_prior_actions(state_array)
        states = torch.as_tensor(state_array, device=self.device).reshape(1, -1)
        goal_tensor = torch.as_tensor(goal_array, device=self.device).reshape(1, -1)
        with torch.no_grad():
            probabilities = self._local_probabilities(states, goal_tensor).squeeze(0)
            local_states, _ = self._split(states, goal_tensor)
            gate_open, low_demand, low_activity, _, _, _ = self._gate_metrics_tensor(
                local_states)
            expected_intervention = self._expected_intervention(
                probabilities.unsqueeze(0), local_states).squeeze(0)
        if not self.enable_residual:
            policy_actions = prior_actions.copy()
        elif evaluate:
            policy_actions = probabilities.argmax(dim=-1).cpu().numpy().astype(np.int64)
        else:
            policy_actions = torch.multinomial(
                probabilities, 1).squeeze(-1).cpu().numpy().astype(np.int64)
        gate_applied = bool(apply_safety_gate and self.safety_gate_enabled)
        is_open = bool(gate_open.item()) if gate_applied else True
        probability_array = probabilities.cpu().numpy()
        actions = prior_actions.copy()
        any_final_nonmask_residual_override = False
        confidence_vetoes = np.zeros(self.num_intersections, dtype=bool)
        shield_vetoes = np.zeros(self.num_intersections, dtype=bool)
        shield_eligible = np.zeros(self.num_intersections, dtype=bool)
        if self.action_shield_enabled:
            shield_eligible = self._shield_eligible_tensor(
                local_states).squeeze(0).cpu().numpy()
        if is_open:
            self._action_stats["expected_intervention_sum"] += float(
                expected_intervention.sum().item())
            self._action_stats["gate_open_actions"] += self.num_intersections
        else:
            self._action_stats["gate_closed_actions"] += self.num_intersections
            if bool(low_demand.item()):
                self._action_stats["gate_low_demand"] += self.num_intersections
            if bool(low_activity.item()):
                self._action_stats["gate_low_activity"] += self.num_intersections
        for index, (prior_action, policy_action, mask) in enumerate(zip(
                prior_actions, policy_actions, mask_array)):
            prior_action = int(prior_action)
            policy_action = int(policy_action)
            prior_is_legal = bool(mask[prior_action])
            if not prior_is_legal:
                selected_action = policy_action
                if not mask[selected_action]:
                    selected_action = int(np.flatnonzero(mask)[0])
            elif not is_open:
                selected_action = prior_action
            else:
                selected_action = policy_action
                if not mask[selected_action]:
                    selected_action = int(np.flatnonzero(mask)[0])
                if (selected_action != prior_action
                        and probability_array[index, selected_action]
                        < self.override_min_probability):
                    selected_action = prior_action
                    confidence_vetoes[index] = True
                elif selected_action != prior_action and self.action_shield_enabled:
                    if (prior_action == self.HOLD
                            and selected_action == self.TERMINATE
                            and bool(shield_eligible[index])):
                        self._action_stats["shield_eligible"] += 1
                    else:
                        selected_action = prior_action
                        shield_vetoes[index] = True
            actions[index] = selected_action
            self._action_stats["prior_action"][prior_action] += 1
            self._action_stats["selected_action"][selected_action] += 1
            self._action_stats["total_actions"] += 1
            if policy_action != prior_action:
                self._action_stats["policy_residual_override"] += 1
                if is_open and prior_is_legal and mask[policy_action]:
                    self._action_stats["policy_nonmask_override"] += 1
            if not prior_is_legal:
                self._action_stats["mask_veto"] += 1
            if confidence_vetoes[index]:
                self._action_stats["confidence_veto"] += 1
            if shield_vetoes[index]:
                self._action_stats["shield_veto"] += 1
            if selected_action != prior_action:
                self._action_stats["residual_override"] += 1
                if prior_is_legal:
                    if prior_action == self.HOLD and selected_action == self.TERMINATE:
                        self._action_stats["override_hold_to_terminate"] += 1
                    elif prior_action == self.TERMINATE and selected_action == self.HOLD:
                        self._action_stats["override_terminate_to_hold"] += 1
                    self._action_stats["nonmask_residual_override"] += 1
                    self._action_stats["final_nonmask_residual_override"] += 1
                    any_final_nonmask_residual_override = True
        joint_index = self.encode_joint_action(actions)
        self._last_action_metadata = {
            "final_joint_action": joint_index,
            "any_final_nonmask_residual_override": bool(
                any_final_nonmask_residual_override),
        }
        return actions, joint_index

    def factorized_joint_probs(self, local_probs, return_log_probs=False):
        indices = self._joint_actions.unsqueeze(0).expand(local_probs.shape[0], -1, -1)
        expanded = local_probs.unsqueeze(1).expand(-1, self.joint_action_count, -1, -1)
        selected = expanded.gather(3, indices.unsqueeze(-1)).squeeze(-1)
        joint_probs = selected.prod(dim=-1)
        if not return_log_probs:
            return joint_probs
        log_probs = torch.log(local_probs.clamp_min(1e-8))
        expanded_logs = log_probs.unsqueeze(1).expand(-1, self.joint_action_count, -1, -1)
        selected_logs = expanded_logs.gather(3, indices.unsqueeze(-1)).squeeze(-1)
        return joint_probs, selected_logs.sum(dim=-1)

    def reset_episode_constraint_costs(self, expected_decisions=1):
        expected_decisions = int(expected_decisions)
        if expected_decisions <= 0:
            raise ValueError("expected_decisions must be positive")
        if self.calibrated_decisions is not None:
            expected_decisions = max(1, int(round(self.calibrated_decisions)))
        self._episode_co2_cost = 0.0
        self._episode_throughput_cost = 0.0
        self._episode_transition_count = 0
        self.episode_horizon = expected_decisions
        self.last_episode_constraint = {
            "co2_violation": 0.0,
            "throughput_violation": 0.0,
            "episode_co2_cost": 0.0,
            "episode_throughput_cost": 0.0,
        }

    def store_transition(self, global_state, goals, joint_action_index,
                         next_global_state, reward, done, duration_seconds=None,
                         co2_cost=0.0, throughput_cost=0.0, episode_cost_scale=1.0,
                         final_nonmask_residual_override=False):
        discount = (self.gamma if duration_seconds is None
                    else self.gamma ** float(duration_seconds))
        self._episode_co2_cost += float(co2_cost) / self.co2_cost_scale
        self._episode_throughput_cost += float(throughput_cost)
        self._episode_transition_count += 1
        episode_cost_scale = float(episode_cost_scale)
        transition = Transition(
            np.asarray(global_state, dtype=np.float32).copy(),
            np.asarray(goals, dtype=np.float32).copy(),
            int(joint_action_index),
            np.asarray(next_global_state, dtype=np.float32).copy(),
            float(reward), float(done), float(discount),
            float(co2_cost) / self.co2_cost_scale, float(throughput_cost),
            float(episode_cost_scale), bool(final_nonmask_residual_override),
        )
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[self.buffer_ptr] = transition
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size

    def _sample_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        override_priority_active = False
        if self.enable_override_prioritized_replay:
            override_count = self.batch_size // 2
            override_transitions = [
                transition for transition in self.replay_buffer
                if transition.final_nonmask_residual_override
            ]
            if len(override_transitions) >= override_count:
                batch = (
                    random.sample(override_transitions, override_count)
                    + random.sample(
                        self.replay_buffer, self.batch_size - override_count)
                )
                override_priority_active = True
            else:
                batch = random.sample(self.replay_buffer, self.batch_size)
        else:
            batch = random.sample(self.replay_buffer, self.batch_size)
        return (
            torch.as_tensor(
                np.stack([x.global_state for x in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                np.stack([x.goals for x in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                [x.joint_action_index for x in batch],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(1),
            torch.as_tensor(
                np.stack([x.next_global_state for x in batch]),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                [x.reward for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            torch.as_tensor(
                [x.done for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            torch.as_tensor(
                [x.discount for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            torch.as_tensor(
                [x.co2_cost for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            torch.as_tensor(
                [x.throughput_cost for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            torch.as_tensor(
                [x.episode_cost_scale for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            torch.as_tensor(
                [x.final_nonmask_residual_override for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            override_priority_active,
        )

    def learn(self):
        batch = self._sample_batch()
        if batch is None:
            return None
        (states, goals, actions, next_states, rewards, dones, discounts,
         co2_costs, throughput_costs, episode_cost_scales, override_flags,
         override_priority_active) = batch
        override_sample_rate = override_flags.mean()
        co2_violation = rewards.new_zeros(())
        throughput_violation = rewards.new_zeros(())
        constrained_rewards = (
            rewards
            - self.lambda_co2 * co2_costs * episode_cost_scales
            - self.lambda_throughput * throughput_costs * episode_cost_scales
        )
        with torch.no_grad():
            next_goals = (
                self.goal_network(next_states) if self.enable_goal
                else next_states.new_zeros((next_states.shape[0], self.total_goal_dim)))
            next_local_probs = self._local_probabilities(next_states, next_goals)
            next_joint_probs, next_joint_logs = self.factorized_joint_probs(
                next_local_probs, return_log_probs=True)
            target_q1, target_q2 = self.target_critic_network(next_states, next_goals)
            target_value = (next_joint_probs * (
                torch.minimum(target_q1, target_q2) - self.alpha * next_joint_logs
            )).sum(dim=1, keepdim=True)
            target = constrained_rewards + (1.0 - dones) * discounts * target_value

        q1, q2 = self.critic_network(states, goals)
        critic_loss = F.mse_loss(q1.gather(1, actions), target) + F.mse_loss(
            q2.gather(1, actions), target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        local_states, _ = self._split(states, goals)
        local_probs = self._local_probabilities(states, goals)
        joint_probs, joint_logs = self.factorized_joint_probs(
            local_probs, return_log_probs=True)
        with torch.no_grad():
            q1, q2 = self.critic_network(states, goals)
            min_q = torch.minimum(q1, q2)
        gate_open_mask = self._gate_open_tensor(states)
        if not self.safety_gate_enabled:
            gate_open_mask = torch.ones_like(gate_open_mask, dtype=torch.bool)
        if self.action_shield_enabled:
            shield_opportunity = (
                self._shield_eligible_tensor(local_states)
                & (self._prior_actions_tensor(local_states) == self.HOLD)
            ).any(dim=1)
            gate_open_mask = gate_open_mask & shield_opportunity
        gate_open_batch_rate = gate_open_mask.float().mean()
        per_state_intervention_rate = self._expected_intervention(
            local_probs, local_states).mean(dim=1)
        if self.enable_advantage_residual:
            v_bar = (joint_probs * min_q).sum(dim=1, keepdim=True).detach()
            positive_advantage_mask = min_q > v_bar
            per_state_policy_loss = (
                joint_probs * (self.alpha * joint_logs - min_q)
                * positive_advantage_mask
            ).sum(dim=1)
            active_state_mask = gate_open_mask & positive_advantage_mask.any(dim=1)
            advantage_active_state_rate = active_state_mask.float().mean()
        else:
            per_state_policy_loss = (
                joint_probs * (self.alpha * joint_logs - min_q)).sum(dim=1)
            active_state_mask = gate_open_mask
            advantage_active_state_rate = gate_open_batch_rate
        if self.enable_residual and bool(active_state_mask.any().item()):
            policy_loss = per_state_policy_loss[active_state_mask].mean()
            expected_intervention_rate = per_state_intervention_rate[active_state_mask].mean()
            intervention_excess = torch.relu(
                expected_intervention_rate - self.target_intervention_rate)
            intervention_loss = self.intervention_penalty * intervention_excess.square()
            worker_loss = policy_loss + intervention_loss
            self.worker_optimizer.zero_grad()
            worker_loss.backward()
            self.worker_optimizer.step()
        else:
            worker_loss = states.new_zeros(())
            policy_loss = worker_loss
            intervention_loss = worker_loss
            expected_intervention_rate = worker_loss

        if self.enable_goal:
            for parameter in self.critic_network.parameters():
                parameter.requires_grad_(False)
            generated_goals = self.goal_network(states)
            goal_probs = self._local_probabilities(states, generated_goals)
            goal_joint_probs = self.factorized_joint_probs(goal_probs)
            goal_q1, goal_q2 = self.critic_network(states, generated_goals)
            goal_loss = -(goal_joint_probs * torch.minimum(goal_q1, goal_q2)).sum(dim=1).mean()
            self.goal_optimizer.zero_grad()
            goal_loss.backward()
            self.goal_optimizer.step()
            for parameter in self.critic_network.parameters():
                parameter.requires_grad_(True)
        else:
            goal_loss = states.new_zeros(())

        with torch.no_grad():
            for target_parameter, parameter in zip(
                    self.target_critic_network.parameters(), self.critic_network.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        self.learn_steps += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "worker_loss": float(worker_loss.item()),
            "goal_loss": float(goal_loss.item()),
            "intervention_loss": float(intervention_loss.item()),
            "expected_intervention_rate": float(expected_intervention_rate.item()),
            "gate_open_batch_rate": float(gate_open_batch_rate.item()),
            "override_sample_rate": float(override_sample_rate.item()),
            "override_priority_active": float(override_priority_active),
            "advantage_active_state_rate": float(advantage_active_state_rate.item()),
            "lambda_co2": self.lambda_co2,
            "lambda_throughput": self.lambda_throughput,
            "co2_violation": 0.0,
            "throughput_violation": 0.0,
        }

    def end_episode(self):
        self.last_episode_constraint = {
            "co2_violation": 0.0,
            "throughput_violation": 0.0,
            "episode_co2_cost": 0.0,
            "episode_throughput_cost": 0.0,
        }
        co2_violation = max(
            0.0,
            self._episode_co2_cost - float(self.constraint_co2_budget),
        ) if self.constraint_co2_budget is not None else 0.0
        throughput_violation = max(
            0.0,
            float(self.constraint_throughput_target) - self._episode_throughput_cost,
        ) if self.constraint_throughput_target is not None else 0.0
        if self.constraint_co2_budget is not None:
            self.lambda_co2 = float(np.clip(
                self.lambda_co2 + self.lagrange_lr * co2_violation,
                0.0, self.max_lambda))
        if self.constraint_throughput_target is not None:
            self.lambda_throughput = float(np.clip(
                self.lambda_throughput + self.lagrange_lr * throughput_violation,
                0.0, self.max_lambda))
        actual_decisions = self._episode_transition_count
        if actual_decisions > 0:
            if self.calibrated_decisions is None:
                self.calibrated_decisions = float(actual_decisions)
            else:
                self.calibrated_decisions = (
                    0.5 * self.calibrated_decisions + 0.5 * float(actual_decisions))
        self.last_episode_constraint = {
            "co2_violation": co2_violation,
            "throughput_violation": throughput_violation,
            "episode_co2_cost": self._episode_co2_cost,
            "episode_throughput_cost": self._episode_throughput_cost,
            "episode_horizon": self.episode_horizon,
            "actual_decisions": actual_decisions,
            "calibrated_decisions": self.calibrated_decisions,
            "lambda_co2": self.lambda_co2,
            "lambda_throughput": self.lambda_throughput,
        }
        self.episode_count += 1
