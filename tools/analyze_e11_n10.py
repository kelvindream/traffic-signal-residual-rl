#!/usr/bin/env python3
"""E11 n=10 合并统计：合并 E4 旧 seeds（2026–2030）与 E11 新 seeds（2101–2105）。

统计口径（正式实验方案 3.12 / 4.3 节）：
- 模型级重复单位 = train seed（每场景 n=10）；每 seed 先对严格相同的
  20 eval seeds（2031–2050）× 5 episodes CRN 配对轨迹求差值均值
- 检验：精确符号翻转检验（枚举 2^n 种符号配置，双侧）+ 配对 Wilcoxon
  exact（n=10 双侧最小 p=0.001953；存在零差时 exact 不可用，显式降级 approx 并注明）
- 多重比较：Holm 校正（族 = 每场景 5 指标）
- 效果量：配对 Cliff's delta（seed 层）；CI：95% bootstrap + 95% t-CI
- C1 升级判据：10/10 同号且 95% bootstrap CI 不跨 0 → 可写 p<0.05 显著声明
- 安全率为 n=10 重估的副产物观察，95% 判据维持不变（Q7a）

比较组：ctde vs fixed_cycle（C1 主比较）、hdrl vs fixed_cycle、ctde vs hdrl。
输出：results/formal_v2/e4_n10_statistics.{json,md}
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/formal_v2"
SCENARIOS = ("low", "medium", "oversaturated", "asymmetric", "pulse")
OLD_SEEDS = tuple(range(2026, 2031))
NEW_SEEDS = tuple(range(2101, 2106))
TRAIN_SEEDS_ALL = OLD_SEEDS + NEW_SEEDS
EVAL_SEEDS = tuple(range(2031, 2051))
EPISODES = tuple(range(5))
EXPECTED_KEYS = {(seed, episode) for seed in EVAL_SEEDS for episode in EPISODES}
ALGORITHMS = {
    "ctde": {
        "eval_algo": "fixed_prior_ctde",
        "sources": (("ctde_300", OLD_SEEDS), ("ctde_300_n10", NEW_SEEDS)),
    },
    "hdrl": {
        "eval_algo": "hdrl",
        "sources": (("hdrl_300", OLD_SEEDS), ("hdrl_300_n10", NEW_SEEDS)),
    },
}
RULE_ALGO = "fixed_cycle"
METRICS = (
    "throughput", "unfinished", "co2_kg",
    "time_average_queue_vehicles", "total_waiting_increment_s",
)
LOWER_IS_BETTER = {metric: metric != "throughput" for metric in METRICS}
BOOTSTRAP_N = 10000
RNG_SEED = 42
SAFE_LIMITS = {
    "throughput": (">=", -1.0),
    "unfinished": ("<=", 1.0),
    "co2_kg": ("<=", 0.5),
    "time_average_queue_vehicles": ("<=", 0.5),
    "total_waiting_increment_s": ("<=", 150.0),
}
COMPARISONS = (
    ("ctde_vs_fixed_cycle", "ctde", "rule", "C1 主比较"),
    ("hdrl_vs_fixed_cycle", "hdrl", "rule", ""),
    ("ctde_vs_hdrl", "ctde", "hdrl", ""),
)


def require_exact(actual, expected, context):
    actual = set(actual)
    expected = set(expected)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context}: 键不一致；missing={missing}, extra={extra}")


def load_model_eval(algo, scenario, allow_partial):
    """合并多目录来源，返回 train_seed -> {(eval_seed, episode): metric row}。"""
    config = ALGORITHMS[algo]
    units = {}
    for dirname, expected_seeds in config["sources"]:
        paths = sorted((RESULTS / dirname).glob(f"{scenario}_seed*/eval_metrics.csv"))
        for path in paths:
            train_seed = int(path.parent.name.rsplit("seed", 1)[1])
            if train_seed not in expected_seeds:
                continue  # 其他批次的目录不属于本实验
            if train_seed in units:
                raise ValueError(f"{algo}/{scenario}: 重复 train seed {train_seed}")
            rows = {}
            with path.open(newline="", encoding="utf-8") as f:
                for row_number, row in enumerate(csv.DictReader(f), start=2):
                    if row["algo"] != config["eval_algo"]:
                        continue
                    if row["scenario"] != scenario:
                        raise ValueError(
                            f"{path}:{row_number}: scenario={row['scenario']}，预期 {scenario}")
                    key = (int(row["seed"]), int(row["episode"]))
                    if key in rows:
                        raise ValueError(f"{path}:{row_number}: 重复键 {key}")
                    rows[key] = row
            require_exact(rows, EXPECTED_KEYS, f"{algo}/{scenario}/train_seed={train_seed}")
            units[train_seed] = rows
    if allow_partial:
        return units
    require_exact(units, TRAIN_SEEDS_ALL, f"{algo}/{scenario} train seeds（n=10 合并）")
    return units


def load_fixed_cycle():
    out = {}
    path = RESULTS / "rule_baselines/all_metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            if row["algo"] != RULE_ALGO:
                continue
            key = (row["scenario"], int(row["seed"]), int(row["episode"]))
            if key in out:
                raise ValueError(f"{path}:{row_number}: fixed_cycle 重复键 {key}")
            out[key] = row
    for scenario in SCENARIOS:
        keys = {key[1:] for key in out if key[0] == scenario}
        require_exact(keys, EXPECTED_KEYS, f"rule={RULE_ALGO}/{scenario}")
    return out


def sign_flip_exact_p(values):
    """精确符号翻转检验（双侧）：枚举全部 2^n 符号配置。

    全同号下 p = 2/2^n（n=10 时 0.001953）；全零时 p=1。
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    obs = abs(float(arr.mean()))
    signs = np.array([
        [1.0 if not ((mask >> i) & 1) else -1.0 for i in range(n)]
        for mask in range(2 ** n)
    ])
    perm_means = np.abs((signs * arr).mean(axis=1))
    tol = 1e-9 * max(1.0, obs)
    return float(np.mean(perm_means >= obs - tol))


def wilcoxon_test(values):
    """配对 Wilcoxon：无零差用 exact；有零差 exact 不可用，显式 approx 并注明。"""
    arr = np.asarray(values, dtype=np.float64)
    zeros = int(np.sum(arr == 0.0))
    if np.all(arr == 0.0):
        return {"stat": None, "p": None, "n": len(arr), "zero_diffs": zeros,
                "method": "全零"}
    if zeros:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # n-零差 过小时的近似提示，已知且可接受
            stat, p = stats.wilcoxon(arr, method="approx")
        method = "approx(存在零差，exact 不可用)"
    else:
        stat, p = stats.wilcoxon(arr, method="exact")
        method = "exact"
    return {"stat": float(stat), "p": float(p), "n": len(arr),
            "zero_diffs": zeros, "method": method}


def holm_adjust(pvals):
    """Holm 步降校正；None（全零不可检验）原样保留。"""
    m = len(pvals)
    valid = sorted(((i, p) for i, p in enumerate(pvals) if p is not None),
                   key=lambda x: x[1])
    out = [None] * m
    running = 0.0
    for rank, (i, p) in enumerate(valid):
        running = max(running, (m - rank) * p)
        out[i] = min(1.0, float(running))
    return out


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def t_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return [mean, mean]
    sem = float(stats.sem(values))
    if sem == 0.0:
        return [mean, mean]
    low, high = stats.t.interval(0.95, df=len(values) - 1, loc=mean, scale=sem)
    return [float(low), float(high)]


def paired_cliffs_delta(values):
    """配对 Cliff's delta（seed 层）：(#{d>0} - #{d<0}) / n，取值 [-1, 1]。"""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    return float((np.sum(arr > 0) - np.sum(arr < 0)) / n)


def summarize(values):
    values = list(values)
    arr = np.asarray(values, dtype=np.float64)
    positive = int(np.sum(arr > 0))
    negative = int(np.sum(arr < 0))
    zero = int(np.sum(arr == 0))
    boot_ci = bootstrap_ci(arr)
    summary = {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
        "bootstrap_ci95": boot_ci,
        "t_ci95": t_ci(arr),
        "sign_flip_exact_p": sign_flip_exact_p(arr),
        "wilcoxon": wilcoxon_test(arr),
        "cliffs_delta_paired": paired_cliffs_delta(arr),
        "positive_seeds": positive,
        "negative_seeds": negative,
        "zero_seeds": zero,
        "same_sign_strict": bool(positive == len(arr) or negative == len(arr)),
        "bootstrap_ci_excludes_zero": bool(boot_ci[0] > 0 or boot_ci[1] < 0),
        "values_by_train_seed": {},
    }
    summary["claim_eligible_p<0.05"] = bool(
        summary["same_sign_strict"] and summary["bootstrap_ci_excludes_zero"])
    return summary


def compare(model_units, reference, scenario, reference_is_rule):
    """A-B 每 train seed 均值差；正值表示 A 指标更大。"""
    result = {}
    for metric in METRICS:
        by_seed = {}
        for seed, rows in model_units.items():
            if reference_is_rule:
                values = [
                    float(rows[key][metric]) - float(rule_rows[(scenario, *key)][metric])
                    for key in sorted(EXPECTED_KEYS)
                ]
            else:
                ref_rows = reference[seed]
                values = [
                    float(rows[key][metric]) - float(ref_rows[key][metric])
                    for key in sorted(EXPECTED_KEYS)
                ]
            by_seed[seed] = float(np.mean(values))
        summary = summarize(by_seed.values())
        summary["values_by_train_seed"] = {str(k): v for k, v in sorted(by_seed.items())}
        if metric == "throughput":
            worst_seed = min(by_seed, key=by_seed.get)
        else:
            worst_seed = max(by_seed, key=by_seed.get)
        summary["worst_train_seed"] = int(worst_seed)
        summary["worst_train_seed_delta"] = float(by_seed[worst_seed])
        result[metric] = summary
    # Holm 校正族 = 本场景 5 指标
    holm = holm_adjust([result[m]["wilcoxon"]["p"] for m in METRICS])
    for metric, p_holm in zip(METRICS, holm):
        result[metric]["wilcoxon_holm_p"] = p_holm
    return result


def safe(row_algo, row_ref):
    deltas = {m: float(row_algo[m]) - float(row_ref[m]) for m in METRICS}
    return all(
        deltas[m] >= limit if op == ">=" else deltas[m] <= limit
        for m, (op, limit) in SAFE_LIMITS.items()
    )


def seed_safety_vs_fixed(algo, scenario, units):
    """每 train seed 的逐 episode 安全率（相对 fixed_cycle，CRN 配对）。"""
    rates = {}
    for seed, rows in units.items():
        flags = [safe(rows[key], rule_rows[(scenario, *key)]) for key in sorted(EXPECTED_KEYS)]
        rates[seed] = float(np.mean(flags))
    return rates


def fmt(summary, metric):
    ci = summary["bootstrap_ci95"]
    p = summary["wilcoxon"]["p"]
    ptxt = "n/a" if p is None else f"{p:.4f}"
    mean = summary["mean"]
    if mean == 0:
        outcome = "持平"
    elif (mean < 0) == LOWER_IS_BETTER[metric]:
        outcome = "A优"
    else:
        outcome = "B优"
    return f"{mean:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}], p={ptxt}, {outcome}"


def main():
    global model_units, rule_rows
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true",
                        help="允许缺失 train seed/未完成单元（仅用于进度检查，"
                             "正式统计必须用完整 n=10 严格模式）")
    args = parser.parse_args()

    rule_rows = load_fixed_cycle()
    model_units = {
        algo: {scenario: load_model_eval(algo, scenario, args.allow_partial)
               for scenario in SCENARIOS}
        for algo in ("ctde", "hdrl")
    }

    completeness = {
        algo: {scenario: {
            "seeds_present": sorted(units),
            "seeds_missing": sorted(set(TRAIN_SEEDS_ALL) - set(units)),
        } for scenario, units in per_scenario.items()}
        for algo, per_scenario in model_units.items()
    }
    partial = any(info["seeds_missing"]
                  for per in completeness.values() for info in per.values())

    output = {
        "protocol": {
            "unit": f"train seed (n={len(TRAIN_SEEDS_ALL)} per scenario, 模型级；"
                    f"旧 {list(OLD_SEEDS)} + 新 {list(NEW_SEEDS)})",
            "within_seed_pairs": "eval seeds 2031-2050 × episodes 0-4 = 100 exact keys（CRN）",
            "difference_definition": "A-B；throughput 正值有利 A，其余指标负值有利 A",
            "tests": "精确符号翻转（双侧，枚举 2^n）+ Wilcoxon exact（零差显式降级 approx）",
            "multiplicity": "Holm 校正，族 = 每（比较, 场景）5 指标",
            "effect_size": "配对 Cliff's delta（seed 层）",
            "bootstrap_resamples": BOOTSTRAP_N,
            "bootstrap_rng_seed": RNG_SEED,
            "c1_upgrade_rule": "10/10 同号且 95% bootstrap CI 不跨 0 → 可写 p<0.05 显著声明；"
                               "否则维持'方向一致 + 诚实披露'表述",
            "safety_note": "安全率为 n=10 重估副产物观察，95% 判据维持不变（Q7a）",
            "partial": bool(partial and args.allow_partial),
        },
        "data_completeness": completeness,
        "comparisons": {},
        "safety_vs_fixed_cycle": {},
    }

    md = [
        "# E11 n=10 合并统计（E4 旧 seeds 2026–2030 + E11 新 seeds 2101–2105）", "",
        "> 所有数值均为 A-B：throughput 越大越好（正值有利 A）；unfinished、co2、queue、waiting 越小越好（负值有利 A）。",
        "> 主统计单位为 train seed（每场景 n=10）；每 seed 先对严格相同的 100 条 eval-seed/episode CRN 轨迹求差值均值。",
        "> 判据（3.12）：10/10 同号且 95% bootstrap CI 不跨 0 → 可写 p<0.05 显著声明；否则维持'方向一致 + 诚实披露'。",
        "> 安全率为 n=10 重估副产物观察，95% 判据维持不变（Q7a）。",
    ]
    if partial:
        if not args.allow_partial:
            print("数据不完整且未指定 --allow-partial，中止。缺失情况：", file=sys.stderr)
            for algo, per in completeness.items():
                for scenario, info in per.items():
                    if info["seeds_missing"]:
                        print(f"  {algo}/{scenario}: missing {info['seeds_missing']}",
                              file=sys.stderr)
            return 1
        md += ["", "> ⚠️ **部分数据模式**：以下结果基于尚未完整的单元，仅作进度检查，不得写入论文。"]

    for comp_name, algo_a, ref_kind, note in COMPARISONS:
        ref_label = "fixed_cycle" if ref_kind == "rule" else "hdrl"
        title = f"{comp_name}（{algo_a} − {ref_label}）{note}".strip()
        output["comparisons"][comp_name] = {"primary_for_C1": comp_name == "ctde_vs_fixed_cycle"}
        md += ["", f"## {title}", "",
               "| 场景 | 指标 | delta均值±std | 95% bootstrap CI | 同号(+/−/0) | "
               "符号翻转p | Wilcoxon p(method) | Holm p | Cliff's δ | 显著声明资格 | 最差seed |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for scenario in SCENARIOS:
            units_a = model_units[algo_a][scenario]
            if not units_a:
                continue
            if ref_kind == "rule":
                comp = compare(units_a, None, scenario, reference_is_rule=True)
            else:
                units_b = model_units[ref_kind][scenario]
                common = sorted(set(units_a) & set(units_b))
                if not common:
                    continue
                comp = compare({s: units_a[s] for s in common},
                               {s: units_b[s] for s in common},
                               scenario, reference_is_rule=False)
            output["comparisons"][comp_name][scenario] = comp
            for metric in METRICS:
                r = comp[metric]
                sign = f"{r['positive_seeds']}/{r['negative_seeds']}/{r['zero_seeds']}"
                p = r["wilcoxon"]["p"]
                ptxt = "n/a" if p is None else f"{p:.4f}"
                holm = r["wilcoxon_holm_p"]
                holm_txt = "n/a" if holm is None else f"{holm:.4f}"
                claim = ("是" if r["claim_eligible_p<0.05"]
                         else ("否(全零)" if r["zero_seeds"] == r["n"] else "否"))
                md.append(
                    f"| {scenario} | {metric} | {r['mean']:+.3f}±{r['std']:.3f} "
                    f"| [{r['bootstrap_ci95'][0]:+.3f}, {r['bootstrap_ci95'][1]:+.3f}] "
                    f"| {sign} | {r['sign_flip_exact_p']:.4f} | {ptxt}({r['wilcoxon']['method']}) "
                    f"| {holm_txt} | {r['cliffs_delta_paired']:+.2f} | {claim} "
                    f"| {r['worst_train_seed']}({r['worst_train_seed_delta']:+.3f}) |")

    md += ["", "## 安全率 vs fixed_cycle（n=10 重估，副产物观察）", "",
           "| 算法 | 场景 | safe/runs | 安全率 | 最差 seed 率 |",
           "|---|---|---|---|---|"]
    for algo in ("ctde", "hdrl"):
        output["safety_vs_fixed_cycle"][algo] = {}
        for scenario in SCENARIOS:
            units = model_units[algo][scenario]
            if not units:
                continue
            rates = seed_safety_vs_fixed(algo, scenario, units)
            n_rows = len(units) * len(EXPECTED_KEYS)
            output["safety_vs_fixed_cycle"][algo][scenario] = {
                "runs": n_rows,
                "safe": int(round(sum(rates.values()) * len(EXPECTED_KEYS))),
                "rate": float(np.mean(list(rates.values()))),
                "per_seed_rates": {str(k): v for k, v in sorted(rates.items())},
                "worst_seed_rate": float(min(rates.values())),
            }
            s = output["safety_vs_fixed_cycle"][algo][scenario]
            md.append(f"| {algo} | {scenario} | {s['safe']}/{s['runs']} "
                      f"| {s['rate'] * 100:.1f}% | {s['worst_seed_rate'] * 100:.1f}% |")

    json_path = RESULTS / "e4_n10_statistics.json"
    md_path = RESULTS / "e4_n10_statistics.md"
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n已写入 {json_path}\n已写入 {md_path}")
    if partial:
        print("\n[提示] 当前为部分数据模式；训练完成后请以严格模式（不带 --allow-partial）重跑。",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
