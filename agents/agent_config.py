"""AgentConfig：FixedPriorCTDEAgent 配置的单一来源。

收敛 agent 的全部构造参数（32 个字段）、取值校验与 CLI 映射；
checkpoint 内的 config 字典（33 键，含 live 状态 lambda/calibrated_decisions）
由 to_legacy_dict / legacy_config_from_agent / from_legacy_dict 编解码，
键名与历史产物 results/formal_v2/ 完全一致，不得增删改名。
"""

import dataclasses


@dataclasses.dataclass
class AgentConfig:
    num_intersections: int = 4
    local_state_dim: int = 40
    local_goal_dim: int = 8
    action_dim: int = 2
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    buffer_size: int = 50000
    tau: float = 0.005
    alpha: float = 0.2
    prior_logit: float = 4.0
    prior_mode: str = "fixed"
    intervention_penalty: float = 0.5
    residual_logit_limit: float = 6.0
    target_intervention_rate: float = 0.15
    gate_min_global_demand: float = 0.1
    gate_min_active_intersections: int = 2
    gate_asymmetry_threshold: float = 0.65
    safety_gate_enabled: bool = True
    override_min_probability: float = 0.0
    shield_min_elapsed_ratio: float = None
    shield_max_current_to_competing_ratio: float = None
    constraint_co2_episode_budget: float = None
    constraint_throughput_episode_target: float = None
    lagrange_lr: float = 1e-3
    initial_lambda_co2: float = 0.0
    initial_lambda_throughput: float = 0.0
    max_lambda: float = 100.0
    co2_cost_scale: float = 1_000_000.0
    enable_pressure_features: bool = False
    enable_residual: bool = True
    enable_goal: bool = True
    enable_advantage_residual: bool = False
    enable_override_prioritized_replay: bool = False

    def __post_init__(self):
        self.num_intersections = int(self.num_intersections)
        self.local_state_dim = int(self.local_state_dim)
        self.local_goal_dim = int(self.local_goal_dim)
        self.action_dim = int(self.action_dim)
        if self.num_intersections != 4 or self.action_dim != 2:
            raise ValueError("Gate 0 expects four intersections with binary actions")
        self.gamma = float(self.gamma)
        self.lr = float(self.lr)
        self.constraint_co2_episode_budget = (
            None if self.constraint_co2_episode_budget is None
            else float(self.constraint_co2_episode_budget))
        self.constraint_throughput_episode_target = (
            None if self.constraint_throughput_episode_target is None
            else float(self.constraint_throughput_episode_target))
        self.lagrange_lr = float(self.lagrange_lr)
        self.max_lambda = float(self.max_lambda)
        self.co2_cost_scale = float(self.co2_cost_scale)
        self.lambda_co2 = float(self.initial_lambda_co2)
        self.lambda_throughput = float(self.initial_lambda_throughput)
        for name in ("constraint_co2_episode_budget",
                     "constraint_throughput_episode_target"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.lagrange_lr < 0.0 or self.max_lambda <= 0.0 or self.co2_cost_scale <= 0.0:
            raise ValueError("lagrange_lr must be non-negative; max_lambda and co2_cost_scale must be positive")
        if not 0.0 <= self.lambda_co2 <= self.max_lambda or not 0.0 <= self.lambda_throughput <= self.max_lambda:
            raise ValueError("initial lambdas must be within [0, max_lambda]")
        self.batch_size = int(self.batch_size)
        self.buffer_size = int(self.buffer_size)
        self.tau = float(self.tau)
        self.alpha = float(self.alpha)
        self.prior_logit = float(self.prior_logit)
        self.prior_mode = str(self.prior_mode)
        if self.prior_mode not in ("fixed", "actuated"):
            raise ValueError("prior_mode must be 'fixed' or 'actuated'")
        self.intervention_penalty = float(self.intervention_penalty)
        self.residual_logit_limit = float(self.residual_logit_limit)
        self.target_intervention_rate = float(self.target_intervention_rate)
        self.gate_min_global_demand = float(self.gate_min_global_demand)
        self.gate_min_active_intersections = int(self.gate_min_active_intersections)
        self.gate_asymmetry_threshold = float(self.gate_asymmetry_threshold)
        self.safety_gate_enabled = bool(self.safety_gate_enabled)
        self.override_min_probability = float(self.override_min_probability)
        self.shield_min_elapsed_ratio = (
            None if self.shield_min_elapsed_ratio is None
            else float(self.shield_min_elapsed_ratio))
        self.shield_max_current_to_competing_ratio = (
            None if self.shield_max_current_to_competing_ratio is None
            else float(self.shield_max_current_to_competing_ratio))
        if self.prior_logit <= 0.0:
            raise ValueError("prior_logit must be positive")
        if self.intervention_penalty < 0.0:
            raise ValueError("intervention_penalty must be non-negative")
        if self.residual_logit_limit <= 0.0:
            raise ValueError("residual_logit_limit must be positive")
        if not 0.0 <= self.target_intervention_rate <= 1.0:
            raise ValueError("target_intervention_rate must be in [0, 1]")
        if self.gate_min_global_demand < 0.0:
            raise ValueError("gate_min_global_demand must be non-negative")
        if not 1 <= self.gate_min_active_intersections <= self.num_intersections:
            raise ValueError("gate_min_active_intersections is out of range")
        if self.gate_asymmetry_threshold < 0.0:
            raise ValueError("gate_asymmetry_threshold must be non-negative")
        if not 0.0 <= self.override_min_probability <= 1.0:
            raise ValueError("override_min_probability must be in [0, 1]")
        shield_values = (
            self.shield_min_elapsed_ratio,
            self.shield_max_current_to_competing_ratio,
        )
        if (shield_values[0] is None) != (shield_values[1] is None):
            raise ValueError("both action shield thresholds must be configured together")
        if shield_values[0] is not None and not 0.0 <= shield_values[0] <= 1.0:
            raise ValueError("shield_min_elapsed_ratio must be in [0, 1]")
        if shield_values[1] is not None and shield_values[1] < 0.0:
            raise ValueError("shield_max_current_to_competing_ratio must be non-negative")
        self.enable_pressure_features = bool(self.enable_pressure_features)
        self.enable_residual = bool(self.enable_residual)
        self.enable_goal = bool(self.enable_goal)
        self.enable_advantage_residual = bool(self.enable_advantage_residual)
        self.enable_override_prioritized_replay = bool(
            self.enable_override_prioritized_replay)

    @classmethod
    def from_args(cls, args, local_state_dim):
        """CLI adapter：train_eval_fixed_prior_ctde 的 argparse Namespace → config。"""
        return cls(
            local_state_dim=local_state_dim,
            batch_size=8,
            buffer_size=50000,
            prior_logit=args.prior_logit,
            prior_mode=args.prior_mode,
            intervention_penalty=args.intervention_penalty,
            residual_logit_limit=args.residual_logit_limit,
            gate_min_global_demand=args.gate_min_global_demand,
            gate_min_active_intersections=args.gate_min_active_intersections,
            gate_asymmetry_threshold=args.gate_asymmetry_threshold,
            safety_gate_enabled=not args.disable_safety_gate,
            override_min_probability=args.override_min_probability,
            shield_min_elapsed_ratio=args.shield_min_elapsed_ratio,
            shield_max_current_to_competing_ratio=(
                args.shield_max_current_to_competing_ratio),
            constraint_co2_episode_budget=args.constraint_co2_episode_budget,
            constraint_throughput_episode_target=(
                args.constraint_throughput_episode_target),
            lagrange_lr=args.lagrange_lr,
            initial_lambda_co2=args.initial_lambda_co2,
            initial_lambda_throughput=args.initial_lambda_throughput,
            max_lambda=args.max_lambda,
            co2_cost_scale=args.co2_cost_scale,
            enable_pressure_features=args.enable_pressure_features,
            enable_residual=not args.disable_residual,
            enable_goal=not args.disable_goal,
            enable_advantage_residual=args.enable_advantage_residual,
            enable_override_prioritized_replay=(
                args.enable_override_prioritized_replay),
        )

    def to_legacy_dict(self):
        """构造值视角的 checkpoint config 字典（33 键，顺序同 save()）。"""
        legacy = {
            "num_intersections": self.num_intersections,
            "local_state_dim": self.local_state_dim,
            "local_goal_dim": self.local_goal_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "constraint_co2_episode_budget": self.constraint_co2_episode_budget,
            "constraint_throughput_episode_target": (
                self.constraint_throughput_episode_target),
            "constraint_scope": "episode",
            "lagrange_lr": self.lagrange_lr,
            "lambda_co2": self.lambda_co2,
            "lambda_throughput": self.lambda_throughput,
            "max_lambda": self.max_lambda,
            "co2_cost_scale": self.co2_cost_scale,
            "calibrated_decisions": None,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "tau": self.tau,
            "alpha": self.alpha,
            "prior_logit": self.prior_logit,
            "prior_mode": self.prior_mode,
            "intervention_penalty": self.intervention_penalty,
            "residual_logit_limit": self.residual_logit_limit,
            "target_intervention_rate": self.target_intervention_rate,
            "gate_min_global_demand": self.gate_min_global_demand,
            "gate_min_active_intersections": self.gate_min_active_intersections,
            "gate_asymmetry_threshold": self.gate_asymmetry_threshold,
            "safety_gate_enabled": self.safety_gate_enabled,
            "override_min_probability": self.override_min_probability,
            "shield_min_elapsed_ratio": self.shield_min_elapsed_ratio,
            "shield_max_current_to_competing_ratio": (
                self.shield_max_current_to_competing_ratio),
            "enable_pressure_features": self.enable_pressure_features,
            "enable_residual": self.enable_residual,
            "enable_goal": self.enable_goal,
        }
        if self.enable_advantage_residual:
            legacy["enable_advantage_residual"] = True
        if self.enable_override_prioritized_replay:
            legacy["enable_override_prioritized_replay"] = True
        return legacy

    @classmethod
    def from_legacy_dict(cls, data):
        """checkpoint config 字典 → config；兼容旧键名，忽略 live 状态键。"""
        if not isinstance(data, dict):
            raise TypeError("legacy config must be a dict")
        field_names = {f.name for f in dataclasses.fields(cls)}
        merged = {k: v for k, v in data.items() if k in field_names}
        for legacy_key, field_name in (
                ("constraint_co2_budget", "constraint_co2_episode_budget"),
                ("constraint_throughput_target",
                 "constraint_throughput_episode_target"),
                ("lambda_co2", "initial_lambda_co2"),
                ("lambda_throughput", "initial_lambda_throughput")):
            if field_name not in merged and legacy_key in data:
                merged[field_name] = data[legacy_key]
        return cls(**merged)


def legacy_config_from_agent(agent):
    """live 值视角的 checkpoint config 字典（save() 使用，键序同 to_legacy_dict）。"""
    legacy = {
        "num_intersections": agent.num_intersections,
        "local_state_dim": agent.local_state_dim,
        "local_goal_dim": agent.local_goal_dim,
        "action_dim": agent.action_dim,
        "gamma": agent.gamma,
        "constraint_co2_episode_budget": agent.constraint_co2_budget,
        "constraint_throughput_episode_target": agent.constraint_throughput_target,
        "constraint_scope": agent.constraint_scope,
        "lagrange_lr": agent.lagrange_lr,
        "lambda_co2": agent.lambda_co2,
        "lambda_throughput": agent.lambda_throughput,
        "max_lambda": agent.max_lambda,
        "co2_cost_scale": agent.co2_cost_scale,
        "calibrated_decisions": agent.calibrated_decisions,
        "batch_size": agent.batch_size,
        "buffer_size": agent.buffer_size,
        "tau": agent.tau,
        "alpha": agent.alpha,
        "prior_logit": agent.prior_logit,
        "prior_mode": agent.prior_mode,
        "intervention_penalty": agent.intervention_penalty,
        "residual_logit_limit": agent.residual_logit_limit,
        "target_intervention_rate": agent.target_intervention_rate,
        "gate_min_global_demand": agent.gate_min_global_demand,
        "gate_min_active_intersections": agent.gate_min_active_intersections,
        "gate_asymmetry_threshold": agent.gate_asymmetry_threshold,
        "safety_gate_enabled": agent.safety_gate_enabled,
        "override_min_probability": agent.override_min_probability,
        "shield_min_elapsed_ratio": agent.shield_min_elapsed_ratio,
        "shield_max_current_to_competing_ratio": (
            agent.shield_max_current_to_competing_ratio),
        "enable_pressure_features": agent.enable_pressure_features,
        "enable_residual": agent.enable_residual,
        "enable_goal": agent.enable_goal,
    }
    if agent.enable_advantage_residual:
        legacy["enable_advantage_residual"] = True
    if agent.enable_override_prioritized_replay:
        legacy["enable_override_prioritized_replay"] = True
    return legacy
