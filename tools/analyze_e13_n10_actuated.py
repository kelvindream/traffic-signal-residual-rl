#!/usr/bin/env python3
"""E13 post-hoc n=10 fixed-prior framework vs actuated controller analysis.

Primary inference unit: training seed (n=10 per scenario). Each seed is first
reduced over the same 100 CRN evaluation keys. The 25 scenario-metric tests form
one Holm family. Positive oriented effects always favor the framework.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/formal_v2"
OUTPUT = RESULTS / "e13_n10_actuated_posthoc"
RULE_PATH = RESULTS / "rule_baselines/all_metrics.csv"
SCENARIOS = ("low", "medium", "oversaturated", "asymmetric", "pulse")
OLD_SEEDS = tuple(range(2026, 2031))
NEW_SEEDS = tuple(range(2101, 2106))
TRAIN_SEEDS = OLD_SEEDS + NEW_SEEDS
EVAL_SEEDS = tuple(range(2031, 2051))
EPISODES = tuple(range(5))
EXPECTED_KEYS = tuple((seed, episode) for seed in EVAL_SEEDS for episode in EPISODES)
METRICS = (
    "throughput",
    "unfinished",
    "co2_kg",
    "time_average_queue_vehicles",
    "total_waiting_increment_s",
)
LOWER_IS_BETTER = {metric: metric != "throughput" for metric in METRICS}
SAFE_LIMITS = {
    "throughput": (">=", -1.0),
    "unfinished": ("<=", 1.0),
    "co2_kg": ("<=", 0.5),
    "time_average_queue_vehicles": ("<=", 0.5),
    "total_waiting_increment_s": ("<=", 150.0),
}
BOOTSTRAP_N = 50_000
RNG_SEED = 20261111
ALPHA = 0.05


def require_exact(actual, expected, context):
    actual, expected = set(actual), set(expected)
    if actual != expected:
        raise ValueError(
            f"{context}: key mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oriented_effect(raw_difference, metric):
    """Convert framework-actuated raw difference to positive=framework better."""
    return -raw_difference if LOWER_IS_BETTER[metric] else raw_difference


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def percentile_ci(samples):
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def seed_bootstrap_ci(seed_effects, n_resamples=BOOTSTRAP_N, rng_seed=RNG_SEED):
    values = np.asarray(seed_effects, dtype=float)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return percentile_ci(values[indices].mean(axis=1))


def hierarchical_bootstrap_ci(
    episode_effects, n_resamples=BOOTSTRAP_N, rng_seed=RNG_SEED + 1
):
    """Resample training seeds and shared CRN episode keys with replacement."""
    matrix = np.asarray(episode_effects, dtype=float)
    if matrix.shape != (len(TRAIN_SEEDS), len(EXPECTED_KEYS)):
        raise ValueError(f"Expected 10x100 episode matrix, got {matrix.shape}")
    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_resamples, dtype=float)
    chunk = 1000
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        seed_idx = rng.integers(0, matrix.shape[0], size=(size, matrix.shape[0]))
        key_idx = rng.integers(0, matrix.shape[1], size=(size, matrix.shape[1]))
        sampled = matrix[seed_idx[:, :, None], key_idx[:, None, :]]
        means[start:start + size] = sampled.mean(axis=(1, 2))
    return percentile_ci(means)


def ci_excludes_zero(interval):
    return interval[0] > 0.0 or interval[1] < 0.0


def claim_eligible(holm_p, seed_ci, hierarchical_ci):
    return bool(
        holm_p < ALPHA
        and ci_excludes_zero(seed_ci)
        and ci_excludes_zero(hierarchical_ci)
    )


def safe(row, fixed_row):
    raw = {metric: float(row[metric]) - float(fixed_row[metric]) for metric in METRICS}
    return all(
        raw[metric] >= limit if operator == ">=" else raw[metric] <= limit
        for metric, (operator, limit) in SAFE_LIMITS.items()
    )


def load_framework():
    units, inputs = {}, []
    sources = (("ctde_300", OLD_SEEDS), ("ctde_300_n10", NEW_SEEDS))
    for scenario in SCENARIOS:
        units[scenario] = {}
        for dirname, allowed_seeds in sources:
            for path in sorted((RESULTS / dirname).glob(f"{scenario}_seed*/eval_metrics.csv")):
                train_seed = int(path.parent.name.rsplit("seed", 1)[1])
                if train_seed not in allowed_seeds:
                    continue
                rows = {}
                with path.open(newline="", encoding="utf-8") as handle:
                    for line, row in enumerate(csv.DictReader(handle), start=2):
                        if row["algo"] != "fixed_prior_ctde":
                            continue
                        key = (int(row["seed"]), int(row["episode"]))
                        if key in rows:
                            raise ValueError(f"{path}:{line}: duplicate key {key}")
                        rows[key] = row
                require_exact(rows, EXPECTED_KEYS, f"framework/{scenario}/{train_seed}")
                units[scenario][train_seed] = rows
                inputs.append(path)
        require_exact(units[scenario], TRAIN_SEEDS, f"framework/{scenario} train seeds")
    return units, inputs


def load_rules():
    rows = {}
    with RULE_PATH.open(newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            if row["algo"] not in ("actuated", "fixed_cycle"):
                continue
            key = (row["scenario"], row["algo"], int(row["seed"]), int(row["episode"]))
            if key in rows:
                raise ValueError(f"{RULE_PATH}:{line}: duplicate key {key}")
            rows[key] = row
    for scenario in SCENARIOS:
        for algo in ("actuated", "fixed_cycle"):
            keys = {(seed, episode) for s, a, seed, episode in rows if s == scenario and a == algo}
            require_exact(keys, EXPECTED_KEYS, f"rule/{scenario}/{algo}")
    return rows


def wilcoxon(values):
    values = np.asarray(values, dtype=float)
    if np.all(values == 0):
        return {"statistic": 0.0, "p_raw": 1.0, "method": "all-zero"}
    zeros = int(np.sum(values == 0))
    method = "exact" if zeros == 0 else "approx"
    statistic, p_value = stats.wilcoxon(values, method=method)
    return {"statistic": float(statistic), "p_raw": float(p_value), "method": method}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    framework, model_inputs = load_framework()
    rules = load_rules()
    seed_rows, episode_rows, tests = [], [], []

    for scenario in SCENARIOS:
        for metric in METRICS:
            episode_matrix, raw_matrix = [], []
            for train_seed in TRAIN_SEEDS:
                oriented_values, raw_values = [], []
                for eval_seed, episode in EXPECTED_KEYS:
                    fw = float(framework[scenario][train_seed][(eval_seed, episode)][metric])
                    act = float(rules[(scenario, "actuated", eval_seed, episode)][metric])
                    raw = fw - act
                    effect = oriented_effect(raw, metric)
                    raw_values.append(raw)
                    oriented_values.append(effect)
                    episode_rows.append({
                        "scenario": scenario, "metric": metric,
                        "train_seed": train_seed, "eval_seed": eval_seed,
                        "episode": episode, "framework": fw, "actuated": act,
                        "raw_framework_minus_actuated": raw,
                        "oriented_effect_positive_framework_better": effect,
                    })
                raw_matrix.append(raw_values)
                episode_matrix.append(oriented_values)
                seed_rows.append({
                    "scenario": scenario, "metric": metric, "train_seed": train_seed,
                    "raw_mean_framework_minus_actuated": float(np.mean(raw_values)),
                    "oriented_mean_positive_framework_better": float(np.mean(oriented_values)),
                    "episodes": len(oriented_values),
                })
            seed_effects = np.mean(np.asarray(episode_matrix), axis=1)
            raw_seed_effects = np.mean(np.asarray(raw_matrix), axis=1)
            test = wilcoxon(seed_effects)
            tests.append({
                "scenario": scenario,
                "metric": metric,
                "raw_mean_framework_minus_actuated": float(raw_seed_effects.mean()),
                "oriented_mean_positive_framework_better": float(seed_effects.mean()),
                "oriented_std_across_training_seeds": float(seed_effects.std(ddof=1)),
                "positive_seeds": int(np.sum(seed_effects > 0)),
                "negative_seeds": int(np.sum(seed_effects < 0)),
                "zero_seeds": int(np.sum(seed_effects == 0)),
                "seed_bootstrap_ci95": seed_bootstrap_ci(seed_effects),
                "hierarchical_bootstrap_ci95": hierarchical_bootstrap_ci(episode_matrix),
                "wilcoxon_statistic": test["statistic"],
                "p_raw": test["p_raw"],
                "wilcoxon_method": test["method"],
                "values_by_training_seed": {
                    str(seed): float(value) for seed, value in zip(TRAIN_SEEDS, seed_effects)
                },
            })

    adjusted = holm_adjust([test["p_raw"] for test in tests])
    for test, holm_p in zip(tests, adjusted):
        test["p_holm_25"] = float(holm_p)
        test["holm_corrected_difference"] = claim_eligible(
            holm_p, test["seed_bootstrap_ci95"], test["hierarchical_bootstrap_ci95"]
        )

    safety_rows = []
    for scenario in SCENARIOS:
        framework_flags = []
        per_seed = {}
        for train_seed in TRAIN_SEEDS:
            flags = []
            for key in EXPECTED_KEYS:
                fixed = rules[(scenario, "fixed_cycle", *key)]
                flag = safe(framework[scenario][train_seed][key], fixed)
                flags.append(flag)
                framework_flags.append(flag)
            per_seed[train_seed] = float(np.mean(flags))
        actuated_flags = [
            safe(rules[(scenario, "actuated", *key)], rules[(scenario, "fixed_cycle", *key)])
            for key in EXPECTED_KEYS
        ]
        safety_rows.extend((
            {
                "scenario": scenario, "controller": "fixed_prior_framework",
                "safe": int(sum(framework_flags)), "runs": len(framework_flags),
                "rate": float(np.mean(framework_flags)),
                "unit_note": "10 training seeds x 100 CRN episodes",
                "per_training_seed_rates": json.dumps(per_seed, sort_keys=True),
            },
            {
                "scenario": scenario, "controller": "actuated",
                "safe": int(sum(actuated_flags)), "runs": len(actuated_flags),
                "rate": float(np.mean(actuated_flags)),
                "unit_note": "100 unique CRN episodes; not replicated across training seeds",
                "per_training_seed_rates": "",
            },
        ))

    protocol = {
        "analysis": "post-hoc n=10 fixed-prior framework vs actuated controller",
        "inference_unit": "training seed (n=10 per scenario)",
        "within_seed_reduction": "mean of 100 exact CRN evaluation keys",
        "metrics": list(METRICS),
        "scenarios": list(SCENARIOS),
        "effect_orientation": "positive always favors framework; raw framework-actuated also retained",
        "test": "two-sided paired Wilcoxon signed-rank on 10 training-seed means",
        "holm_family": "all 5 scenarios x 5 metrics = 25 tests",
        "bootstrap": {
            "seed_level_resamples": BOOTSTRAP_N,
            "hierarchical_resamples": BOOTSTRAP_N,
            "percentile_ci": 0.95,
            "seed_rng": RNG_SEED,
            "hierarchical_rng": RNG_SEED + 1,
            "hierarchy": "resample training seeds and shared CRN episode keys",
        },
        "claim_rule": "Holm p<0.05 and both 95% CIs exclude zero",
        "safety": "descriptive engineering audit versus fixed-cycle tolerances; no proportion test",
    }
    statistics = {"protocol": protocol, "tests": tests}
    (OUTPUT / "statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(OUTPUT / "seed_level_differences.csv", seed_rows, list(seed_rows[0]))
    write_csv(OUTPUT / "episode_level_differences.csv", episode_rows, list(episode_rows[0]))
    write_csv(OUTPUT / "safety_summary.csv", safety_rows, list(safety_rows[0]))

    md = [
        "# E13 n=10 framework vs actuated post-hoc analysis", "",
        "> Positive oriented effects favor the framework. All 25 tests share one Holm family.",
        "> A Holm-corrected difference requires adjusted p<0.05 and both bootstrap CIs to exclude zero.", "",
        "| Scenario | Metric | Oriented mean | Seed 95% CI | Hierarchical 95% CI | Signs +/−/0 | Raw p | Holm p | Claim |",
        "|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for test in tests:
        md.append(
            f"| {test['scenario']} | {test['metric']} | {test['oriented_mean_positive_framework_better']:+.4f} "
            f"| [{test['seed_bootstrap_ci95'][0]:+.4f}, {test['seed_bootstrap_ci95'][1]:+.4f}] "
            f"| [{test['hierarchical_bootstrap_ci95'][0]:+.4f}, {test['hierarchical_bootstrap_ci95'][1]:+.4f}] "
            f"| {test['positive_seeds']}/{test['negative_seeds']}/{test['zero_seeds']} "
            f"| {test['p_raw']:.6f} | {test['p_holm_25']:.6f} "
            f"| {'yes' if test['holm_corrected_difference'] else 'no'} |"
        )
    md.extend(("", "## Descriptive safety audit", "", "| Scenario | Controller | Safe/runs | Rate |", "|---|---|---:|---:|"))
    for row in safety_rows:
        md.append(f"| {row['scenario']} | {row['controller']} | {row['safe']}/{row['runs']} | {row['rate']*100:.1f}% |")
    (OUTPUT / "statistics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    input_paths = sorted(set(model_inputs + [RULE_PATH]))
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(__file__),
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in input_paths],
        "protocol": protocol,
        "output_files": [
            "seed_level_differences.csv", "episode_level_differences.csv",
            "statistics.json", "statistics.md", "safety_summary.csv",
        ],
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote E13 analysis to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
