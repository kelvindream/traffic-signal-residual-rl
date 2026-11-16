#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成论文 Section VI 四张图（IEEE 单栏 3.5in，Okabe-Ito 色盲安全配色）。

数据源（封版，全部只读）：
  - results/formal_v2/e4_n10_statistics.json            -> fig_forest（n=10 森林图）
  - results/formal_v2/e4_n10_statistics.json            -> fig_attribution(a)（n=10 安全率）
  - results/formal_v2/e7_tail_risk.json                 -> fig_attribution(b)(c)（失败结构/压力归因）
  - results/formal_v2/residual_improvement_full/analysis.json -> fig_ablation（P0 约束消融）
  - results/formal_v2/e5_ablation/analysis.json         -> fig_ablation（E5 结构消融）
  - results/formal_v2/e8_convergence_full.json          -> fig_convergence（学习曲线）

输出：paper/figures/fig_forest.pdf 等 4 个 PDF。
"""
from pathlib import Path

import json

import matplotlib
from matplotlib.transforms import blended_transform_factory

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FORMAL = REPO / "results" / "formal_v2"
OUT = Path(__file__).resolve().parent

SCENARIOS = ["low", "medium", "oversaturated", "asymmetric", "pulse"]
SC_LABELS = ["Low", "Medium", "Oversat.", "Asym.", "Pulse"]
METRICS = [
    "throughput",
    "unfinished",
    "co2_kg",
    "time_average_queue_vehicles",
    "total_waiting_increment_s",
]
METRIC_LABELS = {
    "throughput": "Throughput (veh)",
    "unfinished": "Unfinished (veh)",
    "co2_kg": "CO2 (kg)",
    "time_average_queue_vehicles": "Queue (veh)",
    "total_waiting_increment_s": "Waiting (s)",
}
# Okabe-Ito 色盲安全配色
SC_COLORS = {
    "low": "#999999",
    "medium": "#0072B2",
    "oversaturated": "#D55E00",
    "asymmetric": "#009E73",
    "pulse": "#CC79A7",
}
CTDE_COLOR = "#0072B2"
HDRL_COLOR = "#D55E00"

plt.rcParams.update(
    {
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)


def load(name):
    with open(FORMAL / name) as f:
        return json.load(f)


def save(fig, name):
    """同时输出 PDF（论文用）与 PNG（快速目检）。"""
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    print("saved", OUT / f"{name}.pdf")


# ---------------------------------------------------------------- Fig 1: forest
def fig_forest():
    d = load("e4_n10_statistics.json")
    comp = d["comparisons"]["ctde_vs_fixed_cycle"]

    # 主显著四格（medium/oversaturated x throughput/waiting）
    primary = {("medium", "throughput"), ("medium", "total_waiting_increment_s"),
               ("oversaturated", "throughput"), ("oversaturated", "total_waiting_increment_s")}

    left_metrics = ["throughput", "time_average_queue_vehicles", "co2_kg"]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(3.5, 2.9), sharey=True,
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.06},
    )

    yticks, ylabels = [], []
    group_h = 5.0  # 每场景组高度
    for i, sc in enumerate(SCENARIOS):
        base = (len(SCENARIOS) - 1 - i) * group_h
        yticks.append(base)
        ylabels.append(SC_LABELS[i])
        # 左panel: throughput / queue / co2
        offs = {"throughput": 1.2, "time_average_queue_vehicles": 0.0, "co2_kg": -1.2}
        for m, off in offs.items():
            cell = comp[sc][m]
            mean = cell["mean"]
            lo, hi = cell["bootstrap_ci95"]
            filled = (sc, m) in primary
            axL.errorbar(
                mean, base + off,
                xerr=[[mean - lo], [hi - mean]],
                fmt="o", ms=3.2, color=SC_COLORS[sc],
                mfc=SC_COLORS[sc] if filled else "white",
                mec=SC_COLORS[sc], ecolor=SC_COLORS[sc],
                elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=3,
            )
        # 右panel: waiting（与组中心对齐）
        cell = comp[sc]["total_waiting_increment_s"]
        mean = cell["mean"]
        lo, hi = cell["bootstrap_ci95"]
        filled = (sc, "total_waiting_increment_s") in primary
        axR.errorbar(
            mean, base,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o", ms=3.2, color=SC_COLORS[sc],
            mfc=SC_COLORS[sc] if filled else "white",
            mec=SC_COLORS[sc], ecolor=SC_COLORS[sc],
            elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=3,
        )

    for ax, xl in ((axL, "Delta vs fixed-time (veh / kg)"),
                   (axR, "Delta waiting (s)")):
        ax.axvline(0, color="0.3", lw=0.7, zorder=1)
        ax.set_xlabel(xl, fontsize=7)
        ax.tick_params(axis="y", length=0)
    axL.set_ylim(-1.8, (len(SCENARIOS) - 1) * group_h + 4.0)
    axL.set_yticks(yticks)
    axL.set_yticklabels(ylabels)
    axL.text(0.03, 0.06, "favors fixed-time", transform=axL.transAxes,
             fontsize=6, color="0.35", ha="left")
    axL.text(0.97, 0.06, "favors framework", transform=axL.transAxes,
             fontsize=6, color="0.35", ha="right")
    # 图例
    h_fill = plt.Line2D([], [], marker="o", ls="", ms=3.2, mfc=CTDE_COLOR, mec=CTDE_COLOR,
                        label="primary cell (significant)")
    h_open = plt.Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=CTDE_COLOR,
                        label="direction-consistent")
    axL.legend(handles=[h_fill, h_open], loc="upper left", frameon=False,
               bbox_to_anchor=(0.0, 1.14), ncol=1)
    save(fig, "fig_forest")
    plt.close(fig)


# ----------------------------------------------------------- Fig 2: attribution
def fig_attribution():
    e4 = load("e4_n10_statistics.json")
    e7 = load("e7_tail_risk.json")
    saf = e4["safety_vs_fixed_cycle"]

    fig = plt.figure(figsize=(3.5, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.35)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # (a) n=10 安全率
    x = np.arange(len(SCENARIOS))
    w = 0.38
    ctde = [saf["ctde"][sc]["rate"] * 100 for sc in SCENARIOS]
    hdrl = [saf["hdrl"][sc]["rate"] * 100 for sc in SCENARIOS]
    axA.bar(x - w / 2, ctde, w, color=CTDE_COLOR, label="Framework (CTDE)")
    axA.bar(x + w / 2, hdrl, w, color=HDRL_COLOR, label="HDRL")
    axA.axhline(95, color="crimson", ls="--", lw=0.8)
    axA.text(1.5, 96.3, "95% criterion", color="crimson",
             fontsize=6, ha="center")
    for xi, v in zip(x, ctde):
        axA.text(xi - w / 2, v + 1.0, f"{v:.0f}", fontsize=5.5, ha="center", color=CTDE_COLOR)
    axA.set_xticks(x)
    axA.set_xticklabels(SC_LABELS, fontsize=6.5)
    axA.set_ylim(60, 108)
    axA.set_ylabel("Episode safety rate (%)", fontsize=7)
    axA.set_title("(a) Safety audit ($n{=}10$)", fontsize=7.5, loc="left", pad=20)
    axA.legend(frameon=False, loc="lower left", fontsize=6, ncol=2,
               bbox_to_anchor=(0.0, 1.0))

    # (b) 违规组合构成（E7, 有失败的三个场景）
    short = {"throughput": "Thr", "unfinished": "Unf",
             "time_average_queue_vehicles": "Queue",
             "total_waiting_increment_s": "Wait", "co2_kg": "CO2"}
    fail_scen = ["medium", "oversaturated", "asymmetric"]
    yb = np.arange(len(fail_scen))[::-1]
    combo_colors = ["#0072B2", "#E69F00", "#BBBBBB"]
    for yi, sc in zip(yb, fail_scen):
        combos = e7["algorithms"]["ctde"][sc]["attribution"]["violation_structure"]["top_metric_combos"]
        taken = 0.0
        for ci, combo in enumerate(combos[:2]):
            share = combo["share_of_failures"] * 100
            label = "+".join(short[m] for m in combo["metrics"])
            axB.barh(yi, share, left=taken, color=combo_colors[ci], height=0.55)
            if share >= 20:  # 只标注足够宽的段，其余靠图注说明
                axB.text(taken + share / 2, yi, f"{label} {share:.0f}%",
                         ha="center", va="center", fontsize=5.5, color="white")
            taken += share
        axB.barh(yi, 100 - taken, left=taken, color=combo_colors[2], height=0.55)
    axB.set_yticks(yb)
    axB.set_yticklabels([SC_LABELS[SCENARIOS.index(s)] for s in fail_scen], fontsize=6.5)
    axB.set_xlim(0, 100)
    axB.set_xlabel("Share of failures (%)", fontsize=7)
    axB.set_title("(b) Violation structure", fontsize=7.5, loc="left")

    # (c) 压力相关（E7）
    lo_rates, hi_rates = [], []
    for sc in fail_scen:
        dem = e7["algorithms"]["ctde"][sc]["attribution"]["attribution"]["demand"]
        lo_rates.append(dem["fail_rate_lower_stress"] * 100)
        hi_rates.append(dem["fail_rate_upper_stress"] * 100)
    xc = np.arange(len(fail_scen))
    axC.bar(xc - 0.19, lo_rates, 0.38, color="#BBBBBB", label="lower-stress half")
    axC.bar(xc + 0.19, hi_rates, 0.38, color="#D55E00", label="upper-stress half")
    for xi, v in zip(xc, lo_rates):
        axC.text(xi - 0.19, v + 0.5, f"{v:.1f}", fontsize=5.5, ha="center")
    for xi, v in zip(xc, hi_rates):
        axC.text(xi + 0.19, v + 0.5, f"{v:.1f}", fontsize=5.5, ha="center")
    axC.set_xticks(xc)
    axC.set_xticklabels(SC_LABELS[1:4], fontsize=6.5)
    axC.set_ylabel("Failure rate (%)", fontsize=7)
    axC.set_title("(c) Congestion correlation", fontsize=7.5, loc="left")
    axC.legend(frameon=False, fontsize=5.5, loc="upper left")

    for ax in (axA, axB, axC):
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_attribution")
    plt.close(fig)


# -------------------------------------------------------------- Fig 3: ablation
def fig_ablation():
    p0 = load("residual_improvement_full/analysis.json")["variant_vs_center"]
    e5 = load("e5_ablation/analysis.json")["variant_vs_center"]

    def cell(src, variant, sc, metric, aliases=("total_waiting_increment_s", "waiting_s")):
        node = src[variant][sc]
        if metric in node:
            return node[metric]
        for a in aliases:
            if a in node:
                return node[a]
        raise KeyError(f"{variant}/{sc}/{metric}")

    # 行序：约束消融（P0）在上，结构消融（E5）在下
    rows = [
        ("gate_off", p0, "gate off"),
        ("gate_relaxed", p0, "gate relaxed"),
        ("prior1", p0, "prior 1"),
        ("prior2", p0, "prior 2"),
        ("penalty0", p0, "penalty 0"),
        ("penalty01", p0, "penalty 0/1"),
        ("no_residual", e5, "no residual"),
        ("no_goal", e5, "no goal"),
    ]
    n_struct = 2  # E5 结构行数（分隔线下方）

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.1), sharey=True)
    dodge = {"low": -0.24, "medium": -0.12, "oversaturated": 0.0,
             "asymmetric": 0.12, "pulse": 0.24}
    yticks, ylabels = [], []
    n = len(rows)
    for ax, metric, xl in zip(
        axes, ["throughput", "total_waiting_increment_s"],
        ["Delta throughput (veh)", "Delta waiting (s)"],
    ):
        for ri, (variant, src, label) in enumerate(rows):
            y = (n - 1 - ri) * 2.0
            yticks.append(y)
            ylabels.append(label)
            for sc in SCENARIOS:
                c = cell(src, variant, sc, metric)
                mean = c["mean"]
                lo, hi = c["bootstrap_ci95"]
                ax.errorbar(
                    mean, y + dodge[sc],
                    xerr=[[mean - lo], [hi - mean]],
                    fmt="o", ms=2.4, color=SC_COLORS[sc],
                    ecolor=SC_COLORS[sc], elinewidth=0.7,
                    capsize=1.0, capthick=0.7, zorder=3,
                )
        ax.axvline(0, color="0.3", lw=0.7, zorder=1)
        ax.set_xlabel(xl, fontsize=7)
        ax.tick_params(axis="y", length=0)
    ax = axes[0]
    ax.set_ylim(-1.0, (n - 1) * 2.0 + 1.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    # P0 / E5 分隔线
    sep_y = (n - 1 - (len(rows) - n_struct)) * 2.0 - 1.0
    for a in axes:
        a.axhline(sep_y, color="0.6", lw=0.7, ls=":")
    # 分组标签用 blended transform：x 取 figure 坐标（最左侧）、y 取 data 坐标
    # 放在每组纵向中心，避开 y 轴刻度行
    lbl_trans = blended_transform_factory(fig.transFigure, ax.transData)
    grp_c0 = (2.0 * (n - 1) + 4.0) / 2.0          # 约束组行中心（gate off~penalty 0/1）
    grp_c1 = (2.0 * (n_struct - 1) + 0.0) / 2.0   # 结构组行中心（no residual~no goal）
    fig.text(0.015, grp_c0, "constraint ablations (P0)",
             transform=lbl_trans,
             fontsize=5.5, color="0.35", ha="left", va="center")
    fig.text(0.015, grp_c1, "structural (E5)",
             transform=lbl_trans,
             fontsize=5.5, color="0.35", ha="left", va="center")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color=SC_COLORS[sc], label=lb)
               for sc, lb in zip(SCENARIOS, SC_LABELS)]
    axes[1].legend(handles=handles, loc="upper left", frameon=False, fontsize=6,
                   bbox_to_anchor=(1.02, 1.0), title="scenario", title_fontsize=6.5)
    fig.subplots_adjust(left=0.32, wspace=0.08)
    save(fig, "fig_ablation")
    plt.close(fig)


# ----------------------------------------------------------- Fig 4: convergence
def fig_convergence():
    d = load("e8_convergence_full.json")["curves"]
    show = [("medium", "throughput"), ("medium", "total_waiting_increment_s"),
            ("oversaturated", "throughput"), ("oversaturated", "total_waiting_increment_s")]

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.9), sharex=True)
    legend_handles = []
    for ax, (sc, m) in zip(axes.flat, show):
        for algo, color, ls, label in (
            ("ctde", CTDE_COLOR, "-", "Framework (CTDE)"),
            ("hdrl", HDRL_COLOR, "--", "HDRL"),
        ):
            pts = d[algo][sc][m]
            ck = [p["checkpoint"] for p in pts]
            mu = [p["mean_across_units"] for p in pts]
            sd = [p["std_across_units"] for p in pts]
            ln = ax.plot(ck, mu, ls=ls, color=color, label=label, ms=2.2, marker="o")
            ax.fill_between(ck, np.array(mu) - np.array(sd),
                            np.array(mu) + np.array(sd), color=color, alpha=0.12, lw=0)
            if len(legend_handles) < 2:
                legend_handles.append(ln[0])
        ax.set_title(SC_LABELS[SCENARIOS.index(sc)], fontsize=7.5)
    for ax, yl in zip(axes[0], ("Throughput (veh)", "Waiting (s)")):
        ax.set_ylabel(yl, fontsize=7, labelpad=2)
    for ax in axes[1]:
        ax.set_xlabel("Training episode", fontsize=7)
    fig.legend(handles=legend_handles, frameon=False, fontsize=6.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.36, wspace=0.52, bottom=0.16)
    save(fig, "fig_convergence")
    plt.close(fig)


if __name__ == "__main__":
    fig_forest()
    fig_attribution()
    fig_ablation()
    fig_convergence()
    print("all figures done ->", OUT)
