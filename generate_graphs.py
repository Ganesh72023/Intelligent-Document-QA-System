"""
generate_graphs.py - Generates the 10 mandatory Matplotlib performance graphs.

Outputs to results/graphs/:
1. 01_first_token_latency.png: First-token latency comparison across load scenarios.
2. 02_total_response_time.png: Total response-time comparison.
3. 03_percentiles_p50_p95_p99.png: p50, p95, p99 comparison across scenarios.
4. 04_success_vs_failure.png: Success vs rate-limited vs error distribution.
5. 05_queue_length_timeline.png: Queue length / concurrency over time.
6. 06_retry_count_comparison.png: Retry count distribution across scenarios.
7. 07_throughput_scenarios.png: Requests completed per minute across load scenarios.
8. 08_retrieval_config_comparison.png: Retrieval configurations latency & quality tradeoff.
9. 09_answer_quality_breakdown.png: Answer quality grades (correct/refused/wrong).
10. 10_tokens_per_answer.png: Average tokens consumed per answer.
"""

import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_graphs")

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetics
COLORS = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#DB2777"]
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def generate_all():
    csv_path = RESULTS_DIR / "results.csv"
    if not csv_path.exists():
        logger.error("results.csv not found at %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from results.csv", len(df))

    # Coerce numeric columns
    num_cols = ["first_token_ms", "total_ms", "retries", "tokens_in", "tokens_out", "queue_ms", "working_ms", "start_time"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 1. First-Token Latency Comparison
    plt.figure(figsize=(9, 5))
    scenarios = [s for s in df["scenario"].unique() if not s.startswith("retrieval_config")]
    if not scenarios:
        scenarios = df["scenario"].unique()

    data_ft = []
    labels_ft = []
    for s in scenarios:
        sub = df[(df["scenario"] == s) & (df["status"] == "ok") & (df["first_token_ms"] > 0)]
        if len(sub) > 0:
            data_ft.append(sub["first_token_ms"].values)
            labels_ft.append(s.replace("_", "\n"))

    if data_ft:
        bp = plt.boxplot(data_ft, labels=labels_ft, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(COLORS[i % len(COLORS)])
            box.set_alpha(0.7)
    plt.title("Graph 1: First-Token Latency Distribution Across Scenarios", fontweight="bold")
    plt.ylabel("Time to First Token (ms)")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "01_first_token_latency.png")
    plt.close()

    # 2. Total Response Time Comparison
    plt.figure(figsize=(9, 5))
    data_tt = []
    labels_tt = []
    for s in scenarios:
        sub = df[(df["scenario"] == s) & (df["status"] == "ok") & (df["total_ms"] > 0)]
        if len(sub) > 0:
            data_tt.append(sub["total_ms"].values)
            labels_tt.append(s.replace("_", "\n"))

    if data_tt:
        bp = plt.boxplot(data_tt, labels=labels_tt, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(COLORS[(i + 1) % len(COLORS)])
            box.set_alpha(0.7)
    plt.title("Graph 2: Total Response Time Distribution Across Scenarios", fontweight="bold")
    plt.ylabel("Total Latency (ms)")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "02_total_response_time.png")
    plt.close()

    # 3. p50, p95, p99 Comparison
    plt.figure(figsize=(10, 5))
    x = np.arange(len(scenarios))
    w = 0.25
    p50_list, p95_list, p99_list = [], [], []
    for s in scenarios:
        sub = df[(df["scenario"] == s) & (df["status"] == "ok")]["total_ms"]
        p50_list.append(np.percentile(sub, 50) if len(sub) else 0)
        p95_list.append(np.percentile(sub, 95) if len(sub) else 0)
        p99_list.append(np.percentile(sub, 99) if len(sub) else 0)

    plt.bar(x - w, p50_list, width=w, label="p50 (Median)", color="#2563EB", alpha=0.85)
    plt.bar(x, p95_list, width=w, label="p95", color="#D97706", alpha=0.85)
    plt.bar(x + w, p99_list, width=w, label="p99 (Tail)", color="#DC2626", alpha=0.85)
    plt.xticks(x, [s.replace("_", "\n") for s in scenarios])
    plt.title("Graph 3: Latency Percentiles (p50 / p95 / p99) by Scenario", fontweight="bold")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "03_percentiles_p50_p95_p99.png")
    plt.close()

    # 4. Success vs Failure Breakdown
    plt.figure(figsize=(10, 5))
    stat_cats = ["ok", "rate_limited", "timeout", "error"]
    stat_colors = ["#059669", "#D97706", "#7C3AED", "#DC2626"]
    cat_data = {cat: [] for cat in stat_cats}
    for s in scenarios:
        sub = df[df["scenario"] == s]
        tot = max(1, len(sub))
        for cat in stat_cats:
            cat_data[cat].append((len(sub[sub["status"] == cat]) / tot) * 100)

    bottom = np.zeros(len(scenarios))
    for cat, col in zip(stat_cats, stat_colors):
        plt.bar(x, cat_data[cat], bottom=bottom, width=0.55, label=cat.replace("_", " ").title(), color=col, alpha=0.9)
        bottom += np.array(cat_data[cat])

    plt.xticks(x, [s.replace("_", "\n") for s in scenarios])
    plt.title("Graph 4: Request Outcome Breakdown by Scenario (%)", fontweight="bold")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 105)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "04_success_vs_failure.png")
    plt.close()

    # 5. Queue Length / Active Requests Timeline
    plt.figure(figsize=(10, 5))
    for i, s in enumerate(scenarios[:4]):
        sub = df[df["scenario"] == s].sort_values("start_time")
        if len(sub) > 1:
            t0 = sub["start_time"].min()
            t_rel = (sub["start_time"] - t0).values
            active = []
            for t_curr in t_rel:
                c = sum(1 for _, row in sub.iterrows() if (row["start_time"] - t0) <= t_curr < (row["start_time"] - t0 + row["total_ms"] / 1000.0))
                active.append(c)
            plt.plot(t_rel, active, marker="o", markersize=3, label=s, color=COLORS[i % len(COLORS)], linewidth=1.8)

    plt.title("Graph 5: Concurrency / Queue Load Over Time", fontweight="bold")
    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("In-Flight / Queued Requests")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "05_queue_length_timeline.png")
    plt.close()

    # 6. Retry Count Comparison
    plt.figure(figsize=(9, 5))
    retries_avg = [df[df["scenario"] == s]["retries"].mean() for s in scenarios]
    bars = plt.bar(x, retries_avg, width=0.5, color="#7C3AED", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, [s.replace("_", "\n") for s in scenarios])
    plt.title("Graph 6: Average Retries per Request by Scenario", fontweight="bold")
    plt.ylabel("Average Retries")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "06_retry_count_comparison.png")
    plt.close()

    # 7. Throughput Under Different Load Scenarios
    plt.figure(figsize=(9, 5))
    throughputs = []
    for s in scenarios:
        sub = df[(df["scenario"] == s) & (df["status"] == "ok")]
        if len(sub) > 1:
            span_sec = max(1.0, sub["start_time"].max() - sub["start_time"].min() + (sub["total_ms"].max() / 1000.0))
            rpm = (len(sub) / span_sec) * 60.0
        elif len(sub) == 1:
            rpm = 10.0
        else:
            rpm = 0.0
        throughputs.append(rpm)

    bars = plt.bar(x, throughputs, width=0.5, color="#059669", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, [s.replace("_", "\n") for s in scenarios])
    plt.title("Graph 7: Observed Effective Throughput (Req/Min)", fontweight="bold")
    plt.ylabel("Completed Requests per Minute")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "07_throughput_scenarios.png")
    plt.close()

    # 8. Retrieval Configuration Comparison
    ret_df = df[df["scenario"].str.startswith("retrieval_config")]
    if ret_df.empty:
        ret_df = df  # fallback

    configs = sorted(ret_df["retrieval_setting"].dropna().unique())
    plt.figure(figsize=(9, 5))
    med_latencies = []
    for c in configs:
        sub_lat = ret_df[(ret_df["retrieval_setting"] == c) & (ret_df["status"] == "ok")]["total_ms"]
        med_latencies.append(np.median(sub_lat) if len(sub_lat) else 0)

    xc = np.arange(len(configs))
    bars = plt.bar(xc, med_latencies, width=0.45, color="#2563EB", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:.0f}ms", ha="center", va="bottom", fontsize=9)
    plt.xticks(xc, [c[:22] for c in configs], rotation=10, ha="right")
    plt.title("Graph 8: Median Total Latency Across Retrieval Settings", fontweight="bold")
    plt.ylabel("Median Latency (ms)")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "08_retrieval_config_comparison.png")
    plt.close()

    # 9. Answer Quality Breakdown
    plt.figure(figsize=(10, 5))
    grades = ["correct", "refused_correctly", "wrong", "refused_wrongly"]
    grade_colors = ["#059669", "#2563EB", "#DC2626", "#D97706"]
    counts_by_grade = {g: [] for g in grades}
    for c in configs:
        sub_cfg = ret_df[ret_df["retrieval_setting"] == c]
        tot = max(1, len(sub_cfg))
        for g in grades:
            counts_by_grade[g].append((len(sub_cfg[sub_cfg["answer_grade"] == g]) / tot) * 100)

    bottom_g = np.zeros(len(configs))
    for g, col in zip(grades, grade_colors):
        plt.bar(xc, counts_by_grade[g], bottom=bottom_g, width=0.5, label=g.replace("_", " ").title(), color=col, alpha=0.9)
        bottom_g += np.array(counts_by_grade[g])

    plt.xticks(xc, [c[:22] for c in configs], rotation=10, ha="right")
    plt.title("Graph 9: Answer Quality Distribution Across Retrieval Settings (%)", fontweight="bold")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 105)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "09_answer_quality_breakdown.png")
    plt.close()

    # 10. Tokens Per Answer Comparison
    plt.figure(figsize=(9, 5))
    avg_tokens = []
    for c in configs:
        sub_tok = ret_df[(ret_df["retrieval_setting"] == c) & (ret_df["tokens_out"] > 0)]["tokens_out"]
        avg_tokens.append(sub_tok.mean() if len(sub_tok) else 0)

    bars = plt.bar(xc, avg_tokens, width=0.45, color="#D97706", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(xc, [c[:22] for c in configs], rotation=10, ha="right")
    plt.title("Graph 10: Average Completion Tokens Generated per Answer", fontweight="bold")
    plt.ylabel("Tokens per Answer")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "10_tokens_per_answer.png")
    plt.close()

    logger.info("All 10 performance graphs successfully generated in %s", GRAPHS_DIR)


if __name__ == "__main__":
    generate_all()
