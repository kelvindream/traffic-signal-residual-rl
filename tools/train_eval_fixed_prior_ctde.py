#!/usr/bin/env python3
"""Train Fixed-prior CTDE, then run deterministic paired evaluation."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent_config import AgentConfig
from agents.fixed_prior_ctde import FixedPriorCTDEAgent
from environment.async_mult_intersection_env import AsyncMultiIntersectionEnv
from tools.eval_enhanced_actuated_r1 import DEFAULT_SCENARIOS, SCENARIO_MANIFEST
from utils.metrics_schema import (
    COMMON_COLUMNS,
    EVAL_COLUMNS,
    INTERMEDIATE_EVAL_COLUMNS,
    MECHANISM_ZERO_ROW,
    PAIRED_COLUMNS,
    TRAIN_COLUMNS,
    write_csv,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_scenarios = ",".join(DEFAULT_SCENARIOS)
    parser.add_argument("--train-scenarios", default=default_scenarios)
    parser.add_argument("--eval-scenarios", default=default_scenarios)
    parser.add_argument("--train-seeds", default="2026")
    parser.add_argument("--eval-seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--simulation-time", type=int, default=600)
    parser.add_argument("--output", default="results/fixed_prior_ctde_train_eval")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--prior-mode", default="fixed", choices=("fixed", "actuated"))
    parser.add_argument("--prior-logit", type=float, default=4.0)
    parser.add_argument("--intervention-penalty", type=float, default=0.5)
    parser.add_argument("--residual-logit-limit", type=float, default=6.0)
    parser.add_argument("--gate-min-global-demand", type=float, default=0.1)
    parser.add_argument("--gate-min-active-intersections", type=int, default=2)
    parser.add_argument("--gate-asymmetry-threshold", type=float, default=0.65)
    parser.add_argument("--override-min-probability", type=float, default=0.0)
    parser.add_argument("--co2-reward-scale", type=float, default=1e-6)
    parser.add_argument(
        "--constraint-co2-episode-budget", type=float, default=None,
        help="Optional global per-episode CO2 cost budget; omit to disable the constraint.")
    parser.add_argument(
        "--constraint-throughput-episode-target", type=float, default=None,
        help="Optional global per-episode throughput target; omit to disable the constraint.")
    parser.add_argument(
        "--constraint-co2-budgets", default=None,
        help="Optional scenario-specific CO2 budgets, e.g. medium=17.2,oversaturated=25.8.")
    parser.add_argument(
        "--constraint-throughput-targets", default=None,
        help="Optional scenario-specific throughput targets, e.g. medium=76,oversaturated=110.")
    parser.add_argument("--lagrange-lr", type=float, default=1e-3)
    parser.add_argument("--initial-lambda-co2", type=float, default=0.0)
    parser.add_argument("--initial-lambda-throughput", type=float, default=0.0)
    parser.add_argument("--max-lambda", type=float, default=100.0)
    parser.add_argument("--co2-cost-scale", type=float, default=1_000_000.0)
    parser.add_argument("--shield-min-elapsed-ratio", type=float, default=None)
    parser.add_argument(
        "--shield-max-current-to-competing-ratio", type=float, default=None)
    parser.add_argument("--disable-safety-gate", action="store_true")
    parser.add_argument(
        "--disable-residual", action="store_true",
        help="Disable the learned residual and execute the legal prior action.")
    parser.add_argument(
        "--disable-goal", action="store_true",
        help="Disable goal generation and use an all-zero goal vector.")
    parser.add_argument(
        "--enable-advantage-residual", action="store_true",
        help="Enable positive-advantage filtering for residual policy updates.")
    parser.add_argument(
        "--enable-override-prioritized-replay", action="store_true",
        help="Enable replay sampling that prioritizes verified final residual overrides.")
    parser.add_argument("--safe-throughput-tolerance", type=float, default=1.0)
    parser.add_argument("--safe-unfinished-tolerance", type=float, default=1.0)
    parser.add_argument("--safe-co2-tolerance-kg", type=float, default=0.5)
    parser.add_argument("--safe-queue-tolerance", type=float, default=0.5)
    parser.add_argument("--safe-waiting-tolerance-s", type=float, default=150.0)
    parser.add_argument(
        "--enable-pressure-features", action="store_true",
        help="Enable Gate 16B pressure feature projector (adds 5 derived pressure "
             "features per intersection to the worker input).")
    parser.add_argument(
        "--unfinished-penalty-scale", type=float, default=0.0,
        help="Gate 16C reward shaping: per-step penalty = scale * num_vehicles_in_network. "
             "Applied to training reward only; eval/metrics use original reward. 0 disables.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from train_state.json and checkpoint in the output directory "
             "if they exist. Skips episodes already completed.")
    parser.add_argument(
        "--intermediate-eval-interval", type=int, default=0,
        help="E8 pre-registered intermediate evaluation every N training episodes; "
             "0 disables (default). Eval-only rollouts from a frozen checkpoint copy; "
             "training RNG/replay/learn-steps are untouched.")
    parser.add_argument(
        "--intermediate-eval-seeds", default="2031,2032,2033,2034,2035",
        help="Eval seeds for intermediate evaluation checkpoints.")
    parser.add_argument(
        "--intermediate-eval-episodes", type=int, default=3,
        help="Episodes per intermediate eval seed at each checkpoint.")
    return parser.parse_args(argv)


def algorithm_name(config):
    if (config.enable_advantage_residual
            and config.enable_override_prioritized_replay):
        return "advantage_override_ctde"
    return (
        "fixed_prior_ctde" if config.prior_mode == "fixed"
        else "actuated_prior_ctde")


def build_run_signature(config, args, train_scenarios, train_seeds):
    return {
        "prior_mode": config.prior_mode,
        "train_scenarios": train_scenarios,
        "train_seeds": train_seeds,
        "train_episodes": args.train_episodes,
        "simulation_time": args.simulation_time,
        "prior_logit": config.prior_logit,
        "intervention_penalty": config.intervention_penalty,
        "residual_logit_limit": config.residual_logit_limit,
        "gate_min_global_demand": config.gate_min_global_demand,
        "gate_min_active_intersections": config.gate_min_active_intersections,
        "gate_asymmetry_threshold": config.gate_asymmetry_threshold,
        "safety_gate_enabled": config.safety_gate_enabled,
        "override_min_probability": config.override_min_probability,
        "shield_min_elapsed_ratio": config.shield_min_elapsed_ratio,
        "shield_max_current_to_competing_ratio": (
            config.shield_max_current_to_competing_ratio),
        "unfinished_penalty_scale": args.unfinished_penalty_scale,
        "co2_reward_scale": args.co2_reward_scale,
        "enable_pressure_features": config.enable_pressure_features,
        "enable_residual": config.enable_residual,
        "enable_goal": config.enable_goal,
        "enable_advantage_residual": config.enable_advantage_residual,
        "enable_override_prioritized_replay": (
            config.enable_override_prioritized_replay),
        "intermediate_eval_interval": args.intermediate_eval_interval,
        "intermediate_eval_seeds": parse_list(
            args.intermediate_eval_seeds, int),
        "intermediate_eval_episodes": args.intermediate_eval_episodes,
    }


def parse_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_scenario_values(value, flag):
    if value is None:
        return {}
    result = {}
    for item in value.split(","):
        name, separator, raw_value = item.partition("=")
        if not separator or not name.strip() or not raw_value.strip():
            raise ValueError(f"{flag} must use scenario=value entries")
        result[name.strip()] = float(raw_value)
    return result


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_seed(seed, episode):
    return int(seed) * 1000 + int(episode)


def make_env(scenario, co2_reward_scale=1e-6, route_file=None):
    return AsyncMultiIntersectionEnv(
        net_file=ROOT / "nets/2x2-grid/2x2.net.xml",
        route_file=route_file or ROOT / SCENARIO_MANIFEST[scenario],
        min_green=10,
        max_green=40,
        passage=3,
        co2_reward_scale=co2_reward_scale,
    )


def policy_masks(env, ready):
    return np.asarray([
        env.action_mask(tls_id) if tls_id in ready
        else np.asarray([True, False], dtype=bool)
        for tls_id in env.tls_ids
    ], dtype=bool)


def base_row(env, scenario, algo, seed, episode, total_reward, decisions,
             switches, invalid_transitions):
    row = {
        "scenario": scenario,
        "algo": algo,
        "seed": int(seed),
        "episode": int(episode),
        "episode_seed": episode_seed(seed, episode),
        "total_reward": float(total_reward),
        "decisions": int(decisions),
        "switches": int(switches),
        "invalid_transitions": int(invalid_transitions),
    }
    row.update(env.get_episode_metrics())
    return row


def run_ctde_episode(agent, scenario, seed, episode, simulation_time, train,
                     algo_name="fixed_prior_ctde", co2_reward_scale=1e-6,
                     unfinished_penalty_scale=0.0, route_file=None):
    run_seed = episode_seed(seed, episode)
    seed_everything(run_seed)
    env = make_env(
        scenario, co2_reward_scale=co2_reward_scale, route_file=route_file)
    agent.reset_action_stats()
    expected_decisions = max(1, simulation_time // 5)
    agent.reset_episode_constraint_costs(expected_decisions)
    total_reward = 0.0
    decisions = 0
    switches = 0
    invalid = 0
    losses = []
    try:
        _, global_state = env.reset(seed=run_seed, simulation_time=simulation_time)
        done = False
        while not done:
            ready = set(env.ready_tls())
            goals = agent.select_goal(global_state, evaluate=not train)
            actions, _ = agent.select_action(
                global_state,
                goals=goals,
                action_masks=policy_masks(env, ready),
                evaluate=not train,
            )
            commands = {
                tls_id: int(actions[index])
                for index, tls_id in enumerate(env.tls_ids)
                if tls_id in ready
            }
            _, next_state, reward, done, info = env.step(commands)
            consume_metadata = getattr(
                agent, "consume_last_action_metadata", None)
            action_metadata = (
                consume_metadata() if consume_metadata is not None else None)
            total_reward += float(reward)
            decisions += 1
            switches += int(info["switch_count"])
            invalid += int(not info["valid_transition"])
            if train:
                shaped_reward = float(reward)
                if unfinished_penalty_scale > 0.0:
                    unfinished = len(env.departure_times)
                    shaped_reward = float(reward) - unfinished_penalty_scale * unfinished
                executed = np.zeros(agent.num_intersections, dtype=np.int64)
                for index, tls_id in enumerate(env.tls_ids):
                    if tls_id in info["executed_actions"]:
                        executed[index] = int(info["executed_actions"][tls_id])
                executed_joint_action = agent.encode_joint_action(executed)
                final_nonmask_residual_override = bool(
                    action_metadata
                    and action_metadata.get(
                        "any_final_nonmask_residual_override", False)
                    and action_metadata.get("final_joint_action")
                    == executed_joint_action)
                agent.store_transition(
                    global_state, goals, executed_joint_action,
                    next_state, shaped_reward, done,
                    duration_seconds=info["decision_dt"],
                    co2_cost=info["co2_mg"],
                    throughput_cost=info["arrived_delta"],
                    episode_cost_scale=1.0 / expected_decisions,
                    final_nonmask_residual_override=(
                        final_nonmask_residual_override))
                learned = agent.learn()
                if learned is not None:
                    losses.append(learned)
            global_state = next_state
        if train:
            agent.end_episode()
        row = base_row(
            env, scenario, algo_name, seed, episode, total_reward,
            decisions, switches, invalid)
        stats = agent.get_action_stats()
        row.update({
            key: stats[key] for key in (
                "residual_override", "policy_residual_override",
                "policy_nonmask_override", "confidence_veto", "shield_veto",
                "shield_eligible", "override_hold_to_terminate",
                "override_terminate_to_hold", "nonmask_residual_override",
                "final_nonmask_residual_override", "mask_veto", "gate_open_actions",
                "gate_closed_actions",
                "gate_open_rate", "gate_low_demand", "gate_low_activity",
                "total_actions", "residual_intervention_rate",
                "policy_nonmask_override_rate", "confidence_veto_rate",
                "shield_veto_rate", "shield_eligible_rate",
                "nonmask_residual_override_rate",
                "final_nonmask_residual_override_rate", "final_nonmask_rate",
                "expected_intervention_rate", "intervention_loss",
            )
        })
        if train:
            latest = losses[-1] if losses else {}
            finite = all(
                math.isfinite(float(value))
                for loss in losses for value in loss.values())
            row.update({
                "learn_steps": agent.learn_steps,
                "replay_size": len(agent.replay_buffer),
                "critic_loss": latest.get("critic_loss"),
                "worker_loss": latest.get("worker_loss"),
                "goal_loss": latest.get("goal_loss"),
                "intervention_loss": latest.get(
                    "intervention_loss", row["intervention_loss"]),
                "expected_intervention_rate": latest.get(
                    "expected_intervention_rate", row["expected_intervention_rate"]),
                "lambda_co2": agent.lambda_co2,
                "lambda_throughput": agent.lambda_throughput,
                "co2_violation": agent.last_episode_constraint.get("co2_violation", 0.0),
                "throughput_violation": agent.last_episode_constraint.get("throughput_violation", 0.0),
                "episode_co2_cost": agent.last_episode_constraint.get("episode_co2_cost", 0.0),
                "episode_throughput_cost": agent.last_episode_constraint.get("episode_throughput_cost", 0.0),
                "episode_horizon": agent.last_episode_constraint.get("episode_horizon", 0),
                "gate_open_batch_rate": latest.get("gate_open_batch_rate"),
                "loss_finite": bool(finite),
            })
        return row
    finally:
        env.close()


def run_fixed_cycle_episode(scenario, seed, episode, simulation_time,
                            co2_reward_scale=1e-6):
    run_seed = episode_seed(seed, episode)
    seed_everything(run_seed)
    env = make_env(scenario, co2_reward_scale=co2_reward_scale)
    total_reward = 0.0
    decisions = 0
    switches = 0
    invalid = 0
    try:
        env.reset(seed=run_seed, simulation_time=simulation_time)
        done = False
        while not done:
            commands = {}
            for tls_id in env.ready_tls():
                runtime = env.runtime[tls_id]
                now = float(env.sumo.simulation.getTime())
                action = (env.TERMINATE
                          if now - runtime.current_phase_started_at >= env.max_green
                          else env.HOLD)
                mask = env.action_mask(tls_id)
                if not mask[action]:
                    action = env.TERMINATE if mask[env.TERMINATE] else env.HOLD
                commands[tls_id] = action
            _, _, reward, done, info = env.step(commands)
            total_reward += float(reward)
            decisions += 1
            switches += int(info["switch_count"])
            invalid += int(not info["valid_transition"])
        row = base_row(
            env, scenario, "fixed_cycle", seed, episode, total_reward,
            decisions, switches, invalid)
        row.update(MECHANISM_ZERO_ROW)
        return row
    finally:
        env.close()


def pair_rows(eval_rows, algo_name):
    indexed = {
        (row["scenario"], row["seed"], row["episode"], row["algo"]): row
        for row in eval_rows
    }
    pairs = []
    keys = sorted({key[:3] for key in indexed})
    for scenario, seed, episode in keys:
        ctde = indexed[(scenario, seed, episode, algo_name)]
        fixed = indexed[(scenario, seed, episode, "fixed_cycle")]
        pairs.append({
            "scenario": scenario,
            "seed": seed,
            "episode": episode,
            "episode_seed": ctde["episode_seed"],
            "ctde_total_reward": ctde["total_reward"],
            "fixed_cycle_total_reward": fixed["total_reward"],
            "delta_total_reward": ctde["total_reward"] - fixed["total_reward"],
            "ctde_throughput": ctde["throughput"],
            "fixed_cycle_throughput": fixed["throughput"],
            "delta_throughput": ctde["throughput"] - fixed["throughput"],
            "ctde_unfinished": ctde["unfinished"],
            "fixed_cycle_unfinished": fixed["unfinished"],
            "delta_unfinished": ctde["unfinished"] - fixed["unfinished"],
            "ctde_co2_kg": ctde["co2_kg"],
            "fixed_cycle_co2_kg": fixed["co2_kg"],
            "delta_co2_kg": ctde["co2_kg"] - fixed["co2_kg"],
            "ctde_time_average_queue_vehicles": ctde["time_average_queue_vehicles"],
            "fixed_cycle_time_average_queue_vehicles": fixed["time_average_queue_vehicles"],
            "delta_time_average_queue_vehicles": (
                ctde["time_average_queue_vehicles"]
                - fixed["time_average_queue_vehicles"]),
            "ctde_total_waiting_increment_s": ctde["total_waiting_increment_s"],
            "fixed_cycle_total_waiting_increment_s": fixed["total_waiting_increment_s"],
            "delta_total_waiting_increment_s": (
                ctde["total_waiting_increment_s"]
                - fixed["total_waiting_increment_s"]),
            "ctde_invalid_transitions": ctde["invalid_transitions"],
            "fixed_cycle_invalid_transitions": fixed["invalid_transitions"],
            "residual_intervention_rate": ctde["residual_intervention_rate"],
            "policy_nonmask_override_rate": ctde["policy_nonmask_override_rate"],
            "confidence_veto_rate": ctde["confidence_veto_rate"],
            "nonmask_residual_override_rate": ctde[
                "nonmask_residual_override_rate"],
            "final_nonmask_residual_override_rate": ctde[
                "final_nonmask_residual_override_rate"],
            "gate_open_rate": ctde["gate_open_rate"],
        })
    return pairs


def validate_scenarios(values, flag):
    scenarios = parse_list(values)
    unknown = sorted(set(scenarios) - set(SCENARIO_MANIFEST))
    if not scenarios or unknown:
        raise ValueError(f"{flag} contains invalid scenarios: {unknown or scenarios}")
    return scenarios


def evaluate_safety_boundary(paired, args):
    by_scenario = {}
    for scenario in sorted({row["scenario"] for row in paired}):
        rows = [row for row in paired if row["scenario"] == scenario]
        safe_rows = []
        for row in rows:
            conditions = {
                "throughput": float(row["delta_throughput"]) >= -args.safe_throughput_tolerance,
                "unfinished": float(row["delta_unfinished"]) <= args.safe_unfinished_tolerance,
                "co2": float(row["delta_co2_kg"]) <= args.safe_co2_tolerance_kg,
                "queue": float(row["delta_time_average_queue_vehicles"]) <= args.safe_queue_tolerance,
                "waiting": float(row["delta_total_waiting_increment_s"]) <= args.safe_waiting_tolerance_s,
            }
            safe_rows.append({
                "seed": row["seed"],
                "episode": row["episode"],
                "all_five_safe": bool(all(conditions.values())),
                "conditions": conditions,
                "queue_or_waiting_improved": (
                    float(row["delta_time_average_queue_vehicles"]) < 0.0
                    or float(row["delta_total_waiting_increment_s"]) < 0.0),
                "queue_and_waiting_not_worse": (
                    float(row["delta_time_average_queue_vehicles"]) <= 0.0
                    and float(row["delta_total_waiting_increment_s"]) <= 0.0),
            })
        by_scenario[scenario] = {
            "paired_runs": len(rows),
            "all_five_safe_runs": sum(item["all_five_safe"] for item in safe_rows),
            "all_five_safe": bool(safe_rows) and all(
                item["all_five_safe"] for item in safe_rows),
            "queue_or_waiting_improved_runs": sum(
                item["queue_or_waiting_improved"] for item in safe_rows),
            "rows": safe_rows,
        }
    return {
        "tolerances": {
            "throughput": args.safe_throughput_tolerance,
            "unfinished": args.safe_unfinished_tolerance,
            "co2_kg": args.safe_co2_tolerance_kg,
            "queue": args.safe_queue_tolerance,
            "waiting_s": args.safe_waiting_tolerance_s,
        },
        "scenario_results": by_scenario,
        "all_scenarios_all_five_safe": bool(by_scenario) and all(
            item["all_five_safe"] for item in by_scenario.values()),
    }


def intermediate_checkpoint_due(completed_episodes, interval, done):
    """E8 预注册中间评估的到期判定：仅在恰好完成 interval 整数倍时到期。

    返回待评估的 checkpoint_episode（已完成训练回合数），无待办返回 None。
    只承认"磁盘 checkpoint 恰好位于该边界"的情形，禁止跨越边界补评估。
    """
    if interval <= 0 or completed_episodes <= 0:
        return None
    if completed_episodes % interval != 0:
        return None
    return None if completed_episodes in done else completed_episodes


def run_intermediate_eval(agent_factory, checkpoint, scenario, algo_name,
                          eval_seeds, eval_episodes, simulation_time,
                          co2_reward_scale, checkpoint_episode,
                          map_location=None):
    """E8 预注册中间评估：冻结 checkpoint 的纯评估 rollout。

    隔离保证：使用独立 eval agent（不触碰训练 agent 状态），且
    run_ctde_episode 内部以 episode_seed 全量重播种，训练侧 RNG、
    回放缓冲与学习步计数均不受影响；--intermediate-eval-interval=0
    （默认）路径与无钩子行为逐值等价。
    """
    eval_agent = agent_factory()
    eval_agent.load(checkpoint, map_location=map_location)
    try:
        rows = []
        for seed in eval_seeds:
            for episode in range(eval_episodes):
                row = run_ctde_episode(
                    eval_agent, scenario, seed, episode, simulation_time,
                    train=False, algo_name=algo_name,
                    co2_reward_scale=co2_reward_scale)
                rows.append({"checkpoint_episode": int(checkpoint_episode), **row})
        return rows
    finally:
        del eval_agent


def main(argv=None):
    args = parse_args(argv)
    if args.train_episodes <= 0 or args.eval_episodes <= 0 or args.simulation_time <= 0:
        raise ValueError("episode counts and simulation-time must be positive")
    train_scenarios = validate_scenarios(args.train_scenarios, "--train-scenarios")
    eval_scenarios = validate_scenarios(args.eval_scenarios, "--eval-scenarios")
    co2_budgets = parse_scenario_values(args.constraint_co2_budgets, "--constraint-co2-budgets")
    throughput_targets = parse_scenario_values(
        args.constraint_throughput_targets, "--constraint-throughput-targets")
    unknown_budget_scenarios = sorted(set(co2_budgets) - set(SCENARIO_MANIFEST))
    unknown_target_scenarios = sorted(set(throughput_targets) - set(SCENARIO_MANIFEST))
    if unknown_budget_scenarios or unknown_target_scenarios:
        raise ValueError("scenario-specific constraints contain unknown scenarios")
    train_seeds = parse_list(args.train_seeds, int)
    eval_seeds = parse_list(args.eval_seeds, int)
    if not train_seeds or not eval_seeds:
        raise ValueError("train/eval seeds must be non-empty")
    selected_device = resolve_device(args.device)
    config = AgentConfig.from_args(
        args, local_state_dim=AsyncMultiIntersectionEnv.observation_dim)
    algo_name = algorithm_name(config)
    run_signature = build_run_signature(
        config, args, train_scenarios, train_seeds)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        resolve_path(args.checkpoint) if args.checkpoint
        else output / f"{args.prior_mode}_prior_ctde.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    agent = FixedPriorCTDEAgent(config, device_override=selected_device)
    train_rows = []
    interval = args.intermediate_eval_interval
    if interval < 0 or args.intermediate_eval_episodes <= 0:
        raise ValueError(
            "intermediate eval interval must be >= 0 and episodes must be positive")
    intermediate_seeds = parse_list(args.intermediate_eval_seeds, int)
    if interval > 0 and not intermediate_seeds:
        raise ValueError("intermediate eval seeds must be non-empty when enabled")
    intermediate_rows = []
    intermediate_done = set()

    # Resume logic: load train_state.json and checkpoint if --resume is set
    train_state_path = output / "train_state.json"
    resume_from = None  # (scenario_idx, seed_idx, episode) to resume from
    training_complete = False
    if args.resume and train_state_path.exists():
        import json as _json
        with open(train_state_path) as f:
            ts = _json.load(f)
        train_rows = ts.get("train_rows", [])
        intermediate_rows = list(ts.get("intermediate_eval_rows", []))
        intermediate_done = set(ts.get("intermediate_eval_done", []))
        saved_algo_name = ts.get("algo_name", "fixed_prior_ctde")
        resume_matches = (
            saved_algo_name == algo_name
            and ts.get("run_signature") == run_signature
        )
        if resume_matches:
            resume_from = (
                ts.get("scenario_idx", 0),
                ts.get("seed_idx", 0),
                ts.get("episode", -1) + 1,
            )
            if resume_from[2] >= args.train_episodes:
                # Last episode of this scenario-seed was completed; advance
                resume_from = (resume_from[0], resume_from[1] + 1, 0)
            if resume_from[1] >= len(train_seeds):
                resume_from = (resume_from[0] + 1, 0, 0)
            if resume_from[0] < len(train_scenarios):
                agent.load(checkpoint, map_location=selected_device)
                print(
                    f"Resumed from checkpoint: scenario_idx={resume_from[0]} "
                    f"seed_idx={resume_from[1]} episode={resume_from[2]} "
                    f"learn_steps={agent.learn_steps}", flush=True)
            else:
                resume_from = None
                training_complete = True
                agent.load(checkpoint, map_location=selected_device)
                print("All training episodes already completed.", flush=True)
        else:
            raise ValueError(
                "resume signature changed; use a separate output directory "
                "or remove the existing partial run")

    def dump_train_state(scenario_idx, seed_idx, episode):
        import json as _json
        with open(train_state_path, "w") as f:
            _json.dump({
                "prior_mode": args.prior_mode,
                "algo_name": algo_name,
                "run_signature": run_signature,
                "train_scenarios": train_scenarios,
                "train_seeds": train_seeds,
                "train_episodes": args.train_episodes,
                "simulation_time": args.simulation_time,
                "scenario_idx": scenario_idx,
                "seed_idx": seed_idx,
                "episode": episode,
                "learn_steps": agent.learn_steps,
                "train_rows": train_rows,
                "intermediate_eval_rows": intermediate_rows,
                "intermediate_eval_done": sorted(intermediate_done),
            }, f)

    def maybe_intermediate_eval(completed_episodes, scenario,
                                scenario_idx, seed_idx):
        """到期则执行 E8 预注册中间评估并持久化（幂等，可安全 resume）。"""
        checkpoint_episode = intermediate_checkpoint_due(
            completed_episodes, interval, intermediate_done)
        if checkpoint_episode is None:
            return
        rows = run_intermediate_eval(
            lambda: FixedPriorCTDEAgent(config, device_override=selected_device),
            checkpoint, scenario, algo_name, intermediate_seeds,
            args.intermediate_eval_episodes, args.simulation_time,
            args.co2_reward_scale, checkpoint_episode,
            map_location=selected_device)
        intermediate_rows.extend(rows)
        intermediate_done.add(checkpoint_episode)
        write_csv(output / "intermediate_eval.csv",
                  INTERMEDIATE_EVAL_COLUMNS, intermediate_rows)
        dump_train_state(scenario_idx, seed_idx, completed_episodes - 1)
        print(
            f"intermediate_eval scenario={scenario} "
            f"checkpoint_episode={checkpoint_episode} runs={len(rows)}",
            flush=True)

    # Resume 后补跑尚未落盘的边界中间评估（磁盘 checkpoint 恰在边界上才合法）
    pending_unit = None
    if resume_from is not None:
        if intermediate_checkpoint_due(
                resume_from[2], interval, intermediate_done) is not None:
            pending_unit = (resume_from[0], resume_from[1], resume_from[2])
    elif training_complete:
        if intermediate_checkpoint_due(
                args.train_episodes, interval, intermediate_done) is not None:
            pending_unit = (len(train_scenarios) - 1, len(train_seeds) - 1,
                            args.train_episodes)
    if pending_unit is not None:
        scenario_idx, seed_idx, completed = pending_unit
        maybe_intermediate_eval(
            completed, train_scenarios[scenario_idx], scenario_idx, seed_idx)

    for scenario_idx, scenario in enumerate(train_scenarios):
        if training_complete:
            break
        if resume_from and scenario_idx < resume_from[0]:
            continue
        for seed_idx, seed in enumerate(train_seeds):
            if resume_from and scenario_idx == resume_from[0] and seed_idx < resume_from[1]:
                continue
            agent.constraint_co2_budget = co2_budgets.get(
                scenario, args.constraint_co2_episode_budget)
            agent.constraint_throughput_target = throughput_targets.get(
                scenario, args.constraint_throughput_episode_target)
            start_episode = resume_from[2] if (resume_from and scenario_idx == resume_from[0] and seed_idx == resume_from[1]) else 0
            resume_from = None  # Only apply to the first scenario-seed after resume
            for episode in range(start_episode, args.train_episodes):
                row = run_ctde_episode(
                    agent, scenario, seed, episode, args.simulation_time, train=True,
                    algo_name=algo_name, co2_reward_scale=args.co2_reward_scale,
                    unfinished_penalty_scale=args.unfinished_penalty_scale)
                train_rows.append(row)
                print(
                    f"train scenario={scenario} seed={seed} episode={episode} "
                    f"reward={row['total_reward']:.3f} invalid={row['invalid_transitions']} "
                    f"learn_steps={row['learn_steps']}", flush=True)
                # Save train_state.json and checkpoint after each episode
                dump_train_state(scenario_idx, seed_idx, episode)
                agent.save(checkpoint)
                maybe_intermediate_eval(
                    episode + 1, scenario, scenario_idx, seed_idx)
    agent.save(checkpoint)

    eval_agent = FixedPriorCTDEAgent(config, device_override=selected_device)
    eval_agent.load(checkpoint, map_location=selected_device)
    eval_rows = []
    for scenario in eval_scenarios:
        eval_agent.constraint_co2_budget = co2_budgets.get(
            scenario, args.constraint_co2_episode_budget)
        eval_agent.constraint_throughput_target = throughput_targets.get(
            scenario, args.constraint_throughput_episode_target)
        for seed in eval_seeds:
            for episode in range(args.eval_episodes):
                ctde = run_ctde_episode(
                    eval_agent, scenario, seed, episode, args.simulation_time, train=False,
                    algo_name=algo_name, co2_reward_scale=args.co2_reward_scale)
                fixed = run_fixed_cycle_episode(
                    scenario, seed, episode, args.simulation_time,
                    co2_reward_scale=args.co2_reward_scale)
                eval_rows.extend((ctde, fixed))
                print(
                    f"eval scenario={scenario} seed={seed} episode={episode} "
                    f"ctde_reward={ctde['total_reward']:.3f} "
                    f"fixed_reward={fixed['total_reward']:.3f} "
                    f"invalid={ctde['invalid_transitions'] + fixed['invalid_transitions']}",
                    flush=True)

    paired = pair_rows(eval_rows, algo_name)
    safety_boundary = evaluate_safety_boundary(paired, args)
    write_csv(output / "train_metrics.csv", TRAIN_COLUMNS, train_rows)
    write_csv(output / "eval_metrics.csv", EVAL_COLUMNS, eval_rows)
    write_csv(output / "paired_metrics.csv", PAIRED_COLUMNS, paired)
    total_invalid = sum(row["invalid_transitions"] for row in train_rows + eval_rows)
    loss_finite = all(row["loss_finite"] for row in train_rows)
    summary = {
        "device": str(selected_device),
        "algo": algo_name,
        "prior_mode": args.prior_mode,
        "checkpoint": str(checkpoint),
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "simulation_time": args.simulation_time,
        "co2_reward_scale": args.co2_reward_scale,
        "constraint_scope": agent.constraint_scope,
        "constraint_co2_episode_budget": args.constraint_co2_episode_budget,
        "constraint_throughput_episode_target": args.constraint_throughput_episode_target,
        "constraint_co2_budgets": co2_budgets,
        "constraint_throughput_targets": throughput_targets,
        "lagrange_lr": agent.lagrange_lr,
        "lambda_co2": agent.lambda_co2,
        "lambda_throughput": agent.lambda_throughput,
        "max_lambda": agent.max_lambda,
        "co2_cost_scale": agent.co2_cost_scale,
        "prior_logit": agent.prior_logit,
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
        "enable_advantage_residual": agent.enable_advantage_residual,
        "enable_override_prioritized_replay": (
            agent.enable_override_prioritized_replay),
        "worker_local_state_dim": agent.worker_local_state_dim,
        "unfinished_penalty_scale": args.unfinished_penalty_scale,
        "train_runs": len(train_rows),
        "eval_runs": len(eval_rows),
        "paired_runs": len(paired),
        "safety_boundary": safety_boundary,
        "invalid_transitions": int(total_invalid),
        "loss_finite": bool(loss_finite),
        "residual_intervention_rate": (
            float(sum(row["residual_override"] for row in eval_rows
                      if row["algo"] == algo_name))
            / max(1, sum(row["total_actions"] for row in eval_rows
                         if row["algo"] == algo_name))
        ),
        "policy_nonmask_override_rate": (
            float(sum(row["policy_nonmask_override"] for row in eval_rows
                      if row["algo"] == algo_name))
            / max(1, sum(row["total_actions"] for row in eval_rows
                         if row["algo"] == algo_name))
        ),
        "confidence_veto_rate": (
            float(sum(row["confidence_veto"] for row in eval_rows
                      if row["algo"] == algo_name))
            / max(1, sum(row["total_actions"] for row in eval_rows
                         if row["algo"] == algo_name))
        ),
        "nonmask_residual_override_rate": (
            float(sum(row["nonmask_residual_override"] for row in eval_rows
                      if row["algo"] == algo_name))
            / max(1, sum(row["total_actions"] for row in eval_rows
                         if row["algo"] == algo_name))
        ),
        "final_nonmask_residual_override_rate": (
            float(sum(row["final_nonmask_residual_override"] for row in eval_rows
                      if row["algo"] == algo_name))
            / max(1, sum(row["total_actions"] for row in eval_rows
                         if row["algo"] == algo_name))
        ),
        "gate_open_rate": (
            float(sum(row["gate_open_actions"] for row in eval_rows
                      if row["algo"] == algo_name))
            / max(1, sum(row["total_actions"] for row in eval_rows
                         if row["algo"] == algo_name))
        ),
        "gate_low_demand": int(sum(
            row["gate_low_demand"] for row in eval_rows
            if row["algo"] == algo_name)),
        "gate_low_activity": int(sum(
            row["gate_low_activity"] for row in eval_rows
            if row["algo"] == algo_name)),
        "expected_intervention_rate": float(np.mean([
            row["expected_intervention_rate"] for row in eval_rows
            if row["algo"] == algo_name
        ])),
        "intervention_loss": float(np.mean([
            row["intervention_loss"] for row in train_rows
        ])),
    }
    if interval > 0:
        summary["intermediate_eval"] = {
            "interval": interval,
            "seeds": intermediate_seeds,
            "episodes_per_checkpoint": args.intermediate_eval_episodes,
            "checkpoints": sorted(intermediate_done),
            "runs": len(intermediate_rows),
        }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    if total_invalid:
        raise RuntimeError(f"invalid transitions detected: {total_invalid}")
    if not loss_finite:
        raise RuntimeError("non-finite training loss detected")


if __name__ == "__main__":
    main()
