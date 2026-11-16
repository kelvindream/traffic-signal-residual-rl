"""Canonical CSV schemas for formal training and evaluation outputs."""

import csv
from pathlib import Path

EPISODE_METRICS = (
    "throughput", "departed", "completed", "unfinished", "att_mean_s",
    "att_sample_n", "co2_kg", "time_average_queue_vehicles",
    "total_waiting_increment_s", "waiting_per_departed_vehicle_s",
    "terminal_halting_vehicles", "episode_duration_s",
)

COMMON_COLUMNS = (
    "scenario", "algo", "seed", "episode", "episode_seed", "total_reward",
    "decisions", "switches", "invalid_transitions",
) + EPISODE_METRICS

MECHANISM_COUNT_COLUMNS = (
    "residual_override", "policy_residual_override", "policy_nonmask_override",
    "confidence_veto", "shield_veto", "shield_eligible",
    "override_hold_to_terminate", "override_terminate_to_hold",
    "nonmask_residual_override", "final_nonmask_residual_override",
    "mask_veto", "gate_open_actions", "gate_closed_actions",
    "gate_low_demand", "gate_low_activity", "total_actions",
)

MECHANISM_RATE_COLUMNS = (
    "gate_open_rate", "residual_intervention_rate",
    "policy_nonmask_override_rate", "confidence_veto_rate",
    "shield_veto_rate", "shield_eligible_rate",
    "nonmask_residual_override_rate", "final_nonmask_residual_override_rate",
    "final_nonmask_rate", "expected_intervention_rate", "intervention_loss",
)

MECHANISM_COLUMNS = (
    "residual_override", "policy_residual_override", "policy_nonmask_override",
    "confidence_veto", "shield_veto", "shield_eligible",
    "override_hold_to_terminate", "override_terminate_to_hold",
    "nonmask_residual_override", "final_nonmask_residual_override",
    "mask_veto", "gate_open_actions", "gate_closed_actions", "gate_open_rate",
    "gate_low_demand", "gate_low_activity", "total_actions",
    "residual_intervention_rate", "policy_nonmask_override_rate",
    "confidence_veto_rate", "shield_veto_rate", "shield_eligible_rate",
    "nonmask_residual_override_rate", "final_nonmask_residual_override_rate",
    "final_nonmask_rate", "expected_intervention_rate", "intervention_loss",
)

MECHANISM_ZERO_ROW = {
    column: 0 if column in MECHANISM_COUNT_COLUMNS else 0.0
    for column in MECHANISM_COLUMNS
}

TRAIN_COLUMNS = COMMON_COLUMNS + (
    "learn_steps", "replay_size", "critic_loss", "worker_loss", "goal_loss",
    "intervention_loss", "expected_intervention_rate", "gate_open_batch_rate",
    "lambda_co2", "lambda_throughput", "co2_violation",
    "throughput_violation", "episode_co2_cost", "episode_throughput_cost",
    "episode_horizon", "loss_finite",
) + MECHANISM_COLUMNS[:-2]

EVAL_COLUMNS = COMMON_COLUMNS + MECHANISM_COLUMNS
INTERMEDIATE_EVAL_COLUMNS = ("checkpoint_episode",) + EVAL_COLUMNS

PAIRED_COLUMNS = (
    "scenario", "seed", "episode", "episode_seed",
    "ctde_total_reward", "fixed_cycle_total_reward", "delta_total_reward",
    "ctde_throughput", "fixed_cycle_throughput", "delta_throughput",
    "ctde_unfinished", "fixed_cycle_unfinished", "delta_unfinished",
    "ctde_co2_kg", "fixed_cycle_co2_kg", "delta_co2_kg",
    "ctde_time_average_queue_vehicles", "fixed_cycle_time_average_queue_vehicles",
    "delta_time_average_queue_vehicles", "ctde_total_waiting_increment_s",
    "fixed_cycle_total_waiting_increment_s", "delta_total_waiting_increment_s",
    "ctde_invalid_transitions", "fixed_cycle_invalid_transitions",
    "residual_intervention_rate", "policy_nonmask_override_rate",
    "confidence_veto_rate", "nonmask_residual_override_rate",
    "final_nonmask_residual_override_rate", "gate_open_rate",
)


def write_csv(path, columns, rows):
    """Write rows in canonical order and reject undeclared fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    allowed = set(columns)
    rows = list(rows)
    for row in rows:
        extra = set(row) - allowed
        if extra:
            raise ValueError(f"Row contains fields outside schema: {sorted(extra)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
