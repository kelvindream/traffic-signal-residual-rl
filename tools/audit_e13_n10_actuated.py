#!/usr/bin/env python3
"""Independent deterministic audit of E13 point estimates, signs, tests, and Holm."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/formal_v2"
OUT = RESULTS / "e13_n10_actuated_posthoc"
SCENARIOS = ("low", "medium", "oversaturated", "asymmetric", "pulse")
TRAIN_SEEDS = tuple(range(2026, 2031)) + tuple(range(2101, 2106))
KEYS = tuple((seed, episode) for seed in range(2031, 2051) for episode in range(5))
METRICS = ("throughput", "unfinished", "co2_kg", "time_average_queue_vehicles", "total_waiting_increment_s")


def load_rules():
    rows = {}
    with (RESULTS / "rule_baselines/all_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["algo"] == "actuated":
                rows[(row["scenario"], int(row["seed"]), int(row["episode"]))] = row
    return rows


def load_framework(scenario):
    units = {}
    for directory, seeds in (("ctde_300", range(2026, 2031)), ("ctde_300_n10", range(2101, 2106))):
        for seed in seeds:
            path = RESULTS / directory / f"{scenario}_seed{seed}/eval_metrics.csv"
            rows = {}
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["algo"] == "fixed_prior_ctde":
                        rows[(int(row["seed"]), int(row["episode"]))] = row
            if set(rows) != set(KEYS):
                raise AssertionError(f"CRN mismatch: {path}")
            units[seed] = rows
    return units


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        out[index] = min(running, 1.0)
    return out


def main():
    reported = json.loads((OUT / "statistics.json").read_text(encoding="utf-8"))["tests"]
    rules = load_rules()
    audited = []
    for scenario in SCENARIOS:
        framework = load_framework(scenario)
        for metric in METRICS:
            direction = 1.0 if metric == "throughput" else -1.0
            values = []
            for train_seed in TRAIN_SEEDS:
                differences = [
                    direction * (
                        float(framework[train_seed][key][metric])
                        - float(rules[(scenario, *key)][metric])
                    )
                    for key in KEYS
                ]
                values.append(float(np.mean(differences)))
            statistic, p_value = stats.wilcoxon(values, method="exact")
            audited.append({
                "scenario": scenario,
                "metric": metric,
                "mean": float(np.mean(values)),
                "positive": int(np.sum(np.asarray(values) > 0)),
                "negative": int(np.sum(np.asarray(values) < 0)),
                "zero": int(np.sum(np.asarray(values) == 0)),
                "statistic": float(statistic),
                "p_raw": float(p_value),
            })
    adjusted = holm([item["p_raw"] for item in audited])
    for item, p_holm in zip(audited, adjusted):
        item["p_holm"] = p_holm

    if len(reported) != len(audited):
        raise AssertionError("Test count mismatch")
    for expected, actual in zip(reported, audited):
        if (expected["scenario"], expected["metric"]) != (actual["scenario"], actual["metric"]):
            raise AssertionError("Test order mismatch")
        comparisons = (
            (expected["oriented_mean_positive_framework_better"], actual["mean"], "mean"),
            (expected["wilcoxon_statistic"], actual["statistic"], "statistic"),
            (expected["p_raw"], actual["p_raw"], "p_raw"),
            (expected["p_holm_25"], actual["p_holm"], "p_holm"),
        )
        for left, right, label in comparisons:
            if not np.isclose(left, right, rtol=0, atol=1e-12):
                raise AssertionError(f"{actual['scenario']}/{actual['metric']} {label}: {left} != {right}")
        for field, key in (("positive_seeds", "positive"), ("negative_seeds", "negative"), ("zero_seeds", "zero")):
            if expected[field] != actual[key]:
                raise AssertionError(f"{actual['scenario']}/{actual['metric']} {field} mismatch")
    print("E13 independent audit PASS: 25/25 point estimates, signs, Wilcoxon tests, and Holm values match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
