#!/usr/bin/env python3
"""Run the formal R1 paired evaluation for enhanced actuated control."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.adaptive_max_green import AdaptiveMaxGreenController
from agents.enhanced_actuated import EnhancedActuatedController
from agents.rule_coordination import (
    FixedCycleResidualCoordinator,
    NetworkConstrainedCoordinator,
    PlatoonGreenWindowCoordinator,
    SplitOffsetGreenWindowCoordinator,
)
from environment.async_mult_intersection_env import AsyncMultiIntersectionEnv

SCENARIO_MANIFEST = {
    "low": "nets/2x2-grid/2x2-low.rou.xml",
    "medium": "nets/2x2-grid/2x2-medium.rou.xml",
    "oversaturated": "nets/2x2-grid/2x2_oversaturated.rou.xml",
    "asymmetric": "nets/2x2-grid/2x2-asymmetric.rou.xml",
    "pulse": "nets/2x2-grid/2x2-pulse.rou.xml",
}
DEFAULT_SCENARIOS = tuple(SCENARIO_MANIFEST)
FORMAL_ALGORITHMS = ("enhanced_actuated", "fixed_cycle")
EPISODE_METRICS = (
    "throughput", "departed", "completed", "unfinished", "att_mean_s",
    "att_sample_n", "co2_kg", "time_average_queue_vehicles",
    "total_waiting_increment_s", "waiting_per_departed_vehicle_s",
    "terminal_halting_vehicles", "episode_duration_s",
)
RUN_METRICS = (
    "total_reward", "switches", "valid_transitions", "invalid_transitions",
    "coord_base_terminate", "coord_hold", "coord_no_vehicle", "coord_mask_veto",
    "coord_reserved", "coord_opportunity", "coord_low_queue", "coord_already_extended",
    "coord_fixed_fallback", "coord_split_hold", "coord_split_terminate",
    "coord_platoon_phase", "coord_pressure_phase",
    "coord_budget_hold", "coord_budget_terminate",
    "adaptive_fixed_hold", "adaptive_hold", "adaptive_terminate", "adaptive_mask_veto",
    "coord_spillback_veto", "coord_small_platoon_veto",
)
METRICS = EPISODE_METRICS + RUN_METRICS
CSV_COLUMNS = (
    "scenario", "algo", "eval_seed", "episode", "episode_seed", *METRICS,
)


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--algorithms", default="rule_coordination,enhanced_actuated,fixed_cycle")
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--simulation-time", type=int, default=600)
    parser.add_argument("--output", default="results/enhanced_actuated_r1_gate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-random", action="store_true")
    return parser.parse_args(argv)


def episode_seed(eval_seed, episode):
    return int(eval_seed) * 1000 + int(episode)


def run_episode(scenario, algo, eval_seed, episode, simulation_time):
    seed = episode_seed(eval_seed, episode)
    route_file = ROOT / SCENARIO_MANIFEST[scenario]
    env = AsyncMultiIntersectionEnv(
        net_file=ROOT / "nets/2x2-grid/2x2.net.xml",
        route_file=route_file,
        min_green=10,
        max_green=(
            50 if algo == "split_offset_coordination"
            else 48 if algo == "network_constrained_coordination"
            else 42 if algo == "fixed_residual_coordination"
            else 40
        ),
        passage=3,
    )
    rng = random.Random(seed)
    totals = {
        "total_reward": 0.0,
        "switches": 0,
        "valid_transitions": 0,
        "invalid_transitions": 0,
        "coord_base_terminate": 0,
        "coord_hold": 0,
        "coord_no_vehicle": 0,
        "coord_mask_veto": 0,
        "coord_reserved": 0,
        "coord_opportunity": 0,
        "coord_low_queue": 0,
        "coord_already_extended": 0,
        "coord_fixed_fallback": 0,
        "coord_split_hold": 0,
        "coord_split_terminate": 0,
        "coord_platoon_phase": 0,
        "coord_pressure_phase": 0,
        "coord_budget_hold": 0,
        "coord_budget_terminate": 0,
        "coord_spillback_veto": 0,
        "coord_small_platoon_veto": 0,
    }
    try:
        env.reset(seed=seed, simulation_time=simulation_time)
        if algo == "rule_coordination":
            coordinator = PlatoonGreenWindowCoordinator(env)
            controllers = coordinator.base
        elif algo == "fixed_residual_coordination":
            coordinator = FixedCycleResidualCoordinator(env, nominal_green_s=40)
            controllers = None
        elif algo == "split_offset_coordination":
            coordinator = SplitOffsetGreenWindowCoordinator(env)
            controllers = None
        elif algo == "network_constrained_coordination":
            coordinator = NetworkConstrainedCoordinator(env)
            controllers = None
        else:
            coordinator = None
            controllers = {tls: EnhancedActuatedController(env, tls) for tls in env.tls_ids}
        terminated = False
        while not terminated:
            actions = {}
            for tls_id in env.ready_tls():
                if algo == "enhanced_actuated":
                    actions[tls_id] = controllers[tls_id].select_action()
                elif algo in {
                    "rule_coordination",
                    "fixed_residual_coordination",
                    "split_offset_coordination",
                    "network_constrained_coordination",
                }:
                    actions[tls_id] = coordinator.select_action(tls_id)
                elif algo == "fixed_cycle":
                    runtime = env.runtime[tls_id]
                    now = float(env.sumo.simulation.getTime())
                    desired = (env.TERMINATE
                               if now - runtime.current_phase_started_at >= env.max_green
                               else env.HOLD)
                    mask = env.action_mask(tls_id)
                    if not mask[desired]:
                        desired = env.TERMINATE if mask[env.TERMINATE] else env.HOLD
                    actions[tls_id] = desired
                elif algo == "random":
                    mask = env.action_mask(tls_id)
                    actions[tls_id] = rng.choice(
                        [index for index, allowed in enumerate(mask) if allowed])
                else:
                    raise ValueError(f"unknown algorithm: {algo}")
            _, _, reward, terminated, info = env.step(actions)
            totals["total_reward"] += float(reward)
            totals["switches"] += int(info["switch_count"])
            totals["valid_transitions"] += int(info["valid_transition"])
            totals["invalid_transitions"] += int(not info["valid_transition"])

        row = {
            "scenario": scenario,
            "algo": algo,
            "eval_seed": int(eval_seed),
            "episode": int(episode),
            "episode_seed": seed,
        }
        if coordinator is not None:
            totals["coord_base_terminate"] = coordinator.stats.get("base_terminate", 0)
            totals["coord_hold"] = coordinator.stats.get(
                "coordinated_hold", coordinator.stats.get("residual_hold", 0))
            totals["coord_no_vehicle"] = coordinator.stats.get("no_connected_vehicle", 0)
            totals["coord_mask_veto"] = coordinator.stats.get("masked_veto", 0)
            totals["coord_reserved"] = coordinator.stats.get("residual_reserved", 0)
            totals["coord_opportunity"] = coordinator.stats.get("residual_opportunity", 0)
            totals["coord_low_queue"] = coordinator.stats.get("low_network_queue", 0)
            totals["coord_already_extended"] = coordinator.stats.get("already_extended", 0)
            totals["coord_fixed_fallback"] = coordinator.stats.get("fixed_fallback", 0)
            totals["coord_split_hold"] = coordinator.stats.get("split_hold", 0)
            totals["coord_split_terminate"] = coordinator.stats.get("split_terminate", 0)
            totals["coord_platoon_phase"] = coordinator.stats.get("platoon_phase", 0)
            totals["coord_pressure_phase"] = coordinator.stats.get("pressure_phase", 0)
            totals["coord_budget_hold"] = coordinator.stats.get("budget_hold", 0)
            totals["coord_budget_terminate"] = coordinator.stats.get("budget_terminate", 0)
            totals["coord_spillback_veto"] = coordinator.stats.get("spillback_veto", 0)
            totals["coord_small_platoon_veto"] = coordinator.stats.get("small_platoon_veto", 0)
        row.update(env.get_episode_metrics())
        row.update(totals)
        return row
    finally:
        env.close()


def paired_key(row):
    return (row["scenario"], int(row["eval_seed"]), int(row["episode"]),
            int(row["episode_seed"]))


def validate_rows(rows, algorithms):
    expected_keys = None
    for algo in algorithms:
        algo_rows = [row for row in rows if row["algo"] == algo]
        keys = {paired_key(row) for row in algo_rows}
        if len(keys) != len(algo_rows):
            raise RuntimeError(f"duplicate paired key for {algo}")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError(f"paired key mismatch for {algo}")
    if any(int(row["invalid_transitions"]) > 0 for row in rows):
        raise RuntimeError("invalid transition detected; R1 evaluation is invalid")


def write_outputs(output, rows, args, scenarios, seeds, algorithms):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "scenarios": scenarios,
        "scenario_manifest": {name: SCENARIO_MANIFEST[name] for name in scenarios},
        "network": "nets/2x2-grid/2x2.net.xml",
        "algorithms": algorithms,
        "eval_seeds": seeds,
        "episodes_per_seed": args.episodes,
        "simulation_time_s": args.simulation_time,
        "episode_seed_formula": "eval_seed*1000+episode",
        "include_random": args.include_random,
        "metrics_columns": list(CSV_COLUMNS),
    }
    with (output / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def main(argv=None):
    args = parse_args(argv)
    if args.episodes <= 0 or args.simulation_time <= 0:
        raise ValueError("episodes and simulation-time must be positive")
    scenarios = parse_csv_list(args.scenarios)
    unknown = sorted(set(scenarios) - set(SCENARIO_MANIFEST))
    if not scenarios or unknown:
        raise ValueError(f"invalid scenarios: {unknown or scenarios}")
    seeds = [int(item) for item in parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("--seeds requires at least one integer")
    algorithms = parse_csv_list(args.algorithms)
    allowed = {
        "rule_coordination",
        "fixed_residual_coordination",
        "split_offset_coordination",
        "network_constrained_coordination",
        *FORMAL_ALGORITHMS,
    }
    if args.include_random:
        allowed.add("random")
    if not algorithms or set(algorithms) - allowed:
        raise ValueError(f"invalid algorithms: {sorted(set(algorithms) - allowed)}")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    combinations = [
        (scenario, algo, seed, episode)
        for algo in algorithms
        for scenario in scenarios
        for seed in seeds
        for episode in range(args.episodes)
    ]
    if args.dry_run:
        print(json.dumps({"episodes": len(combinations), "algorithms": algorithms,
                          "scenarios": scenarios, "seeds": seeds}, indent=2))
        return 0
    rows = []
    for index, (scenario, algo, seed, episode) in enumerate(combinations, 1):
        print(f"[{index}/{len(combinations)}] {scenario} {algo} seed={seed} episode={episode}",
              flush=True)
        rows.append(run_episode(scenario, algo, seed, episode, args.simulation_time))

    write_outputs(output, rows, args, scenarios, seeds, algorithms)
    validate_rows(rows, algorithms)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
