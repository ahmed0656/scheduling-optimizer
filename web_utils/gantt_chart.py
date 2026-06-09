import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import base64
import io

GANTT_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#673AB7", "#E91E63",
    "#009688", "#CDDC39", "#3F51B5", "#FF4081", "#00E676",
    "#FFAB00", "#7C4DFF", "#18FFFF", "#FF6E40", "#69F0AE",
    "#D500F9", "#FFD600", "#536DFE", "#00C853", "#FF1744",
]


def _setup_axes(ax, fig):
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_color("#ffffff")
        spine.set_alpha(0.2)
    ax.tick_params(colors="white", labelsize=10)


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=80)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img


def create_gantt_chart(result, title="Gantt Chart", figsize=None):
    assignments = result["assignments"]
    processing_times = np.array(result["processing_times"])
    priorities = result.get("priorities", [1] * len(processing_times))
    num_machines = len(assignments)
    makespan = float(result["makespan"])
    unavailability = result.get("unavailability", {})
    job_start_times = result.get("job_start_times", {})

    if figsize is None:
        figsize = (10, max(3.5, num_machines * 0.8 + 1.0))

    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    fig.patch.set_facecolor("#1a1a2e")
    _setup_axes(ax, fig)

    has_unavailability = any(len(unavailability.get(m, [])) > 0 for m in range(num_machines))
    bar_height = 0.50

    for m_idx in range(num_machines):
        for u_start, u_end, reason in unavailability.get(m_idx, []):
            ax.barh(
                y=m_idx, width=u_end - u_start, left=u_start,
                height=bar_height + 0.15, color="#ff1744",
                alpha=0.30, hatch="///", edgecolor="#ff5252",
                linewidth=0.8, zorder=1,
            )
            ax.text(
                u_start + (u_end - u_start) / 2, m_idx + bar_height / 2 + 0.15,
                reason if reason else "N/A",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
                color="#ff5252", zorder=2,
            )

    for m_idx in range(num_machines):
        schedule = job_start_times.get(m_idx, [])
        if schedule:
            schedule_sorted = sorted(schedule, key=lambda x: x["start"])
            for item in schedule_sorted:
                job = int(item["job"])
                start = float(item["start"])
                duration = float(item["duration"])
                color = GANTT_COLORS[job % len(GANTT_COLORS)]
                ax.barh(
                    y=m_idx, width=duration, left=start,
                    height=bar_height, color=color,
                    edgecolor="white", linewidth=0.6,
                    alpha=0.92, zorder=3,
                )
                label = f"J{job + 1}\nP{priorities[job]}"
                if duration >= 5:
                    ax.text(
                        start + duration / 2, m_idx, label,
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white", zorder=4,
                    )
                elif duration >= 2:
                    ax.text(
                        start + duration / 2, m_idx + bar_height / 2 + 0.06,
                        label, ha="center", va="bottom",
                        fontsize=6, fontweight="bold",
                        color=color, zorder=4,
                    )
        else:
            current_time = 0
            for job in sorted(assignments[m_idx]):
                duration = float(processing_times[job])
                color = GANTT_COLORS[job % len(GANTT_COLORS)]
                ax.barh(
                    y=m_idx, width=duration, left=current_time,
                    height=bar_height, color=color,
                    edgecolor="white", linewidth=1.0,
                    alpha=0.92, zorder=3,
                )
                if duration >= 5:
                    ax.text(
                        current_time + duration / 2, m_idx,
                        f"J{job + 1}\nP{priorities[job]}",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white", zorder=4,
                    )
                current_time += duration

    ax.set_yticks(list(range(num_machines)))
    ax.set_yticklabels(
        [f"Machine {i + 1}" for i in range(num_machines)],
        fontsize=10, fontweight="bold", color="white",
    )
    ax.set_xlabel("Time", fontsize=10, fontweight="bold", color="white", labelpad=8)

    has_priorities = any(p != 0 for p in priorities)
    subtitle = f"Makespan = {makespan:.2f}"
    if has_unavailability:
        subtitle += "      (/// = unavailable)"
    if has_priorities:
        subtitle += "      P = Priority"
    ax.set_title(
        f"{title}\n{subtitle}",
        fontsize=11, fontweight="bold", color="#64ffda",
        pad=12,
    )

    ax.grid(True, axis="x", alpha=0.15, color="#ffffff", linestyle="--")
    ax.set_xlim(0, makespan * 1.08)
    ax.set_ylim(-0.85, num_machines - 0.15)

    if makespan <= 30:
        ax.set_xticks(list(range(0, int(makespan) + 2, 1)))
    elif makespan <= 100:
        ax.set_xticks(list(range(0, int(makespan) + 6, 5)))
    else:
        ax.set_xticks(list(range(0, int(makespan) + 11, 10)))

    legend_elements = [mpatches.Patch(color="#2196F3", alpha=0.9, label="Jobs")]
    if has_unavailability:
        legend_elements.append(
            mpatches.Patch(color="#ff1744", alpha=0.3, hatch="///", label="Unavailable")
        )
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=9, facecolor="#1a1a2e",
        edgecolor="#64ffda", labelcolor="white",
    )

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.88)
    return _fig_to_base64(fig)


def create_convergence_plot(results, figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    fig.patch.set_facecolor("#1a1a2e")
    _setup_axes(ax, fig)

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    markers = ["o", "s", "^"]

    for idx, result in enumerate(results):
        history = result.get("history", [])
        if history:
            ax.plot(
                history, color=colors[idx], marker=markers[idx],
                linewidth=2.5, markersize=4,
                label=result["algorithm"], alpha=0.9,
                markevery=max(1, len(history) // 25),
            )

    ax.set_xlabel("Iteration", fontsize=10, fontweight="bold", color="white", labelpad=8)
    ax.set_ylabel("Makespan", fontsize=10, fontweight="bold", color="white", labelpad=8)
    ax.set_title(
        "Convergence Comparison",
        fontsize=12, fontweight="bold", color="#64ffda", pad=12,
    )
    ax.legend(
        fontsize=9, facecolor="#1a1a2e", edgecolor="#64ffda",
        labelcolor="white", loc="upper right",
    )
    ax.grid(True, alpha=0.2, color="#ffffff", linestyle="--")

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.88)
    return _fig_to_base64(fig)


def create_comparison_bar_chart(results, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    fig.patch.set_facecolor("#1a1a2e")
    _setup_axes(ax, fig)

    algorithms = [r["algorithm"] for r in results]
    makespans = [r["makespan"] for r in results]
    times = [r["execution_time"] for r in results]
    colors_list = ["#2196F3", "#4CAF50", "#FF9800"][:len(results)]

    bars = ax.bar(
        range(len(algorithms)), makespans, width=0.55,
        color=colors_list, edgecolor="white", linewidth=1, alpha=0.9,
    )

    y_max = max(makespans) * 1.18
    for bar, val, t in zip(bars, makespans, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + y_max * 0.02,
            f"{val:.2f}\n{t:.3f}s",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="white",
        )

    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=10, fontweight="bold", color="white")
    ax.set_ylabel("Makespan", fontsize=11, fontweight="bold", color="white", labelpad=8)
    ax.set_title(
        "Makespan & Time Comparison",
        fontsize=12, fontweight="bold", color="#64ffda", pad=12,
    )
    ax.grid(True, axis="y", alpha=0.2, color="#ffffff", linestyle="--")
    ax.set_ylim(0, y_max)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.88)
    return _fig_to_base64(fig)


def create_machine_utilization_chart(result, figsize=None, algorithm_name=""):
    makespan = float(result["makespan"])
    assignments = result["assignments"]
    processing_times = np.array(result["processing_times"])
    unavailability = result.get("unavailability", {})
    num_machines = len(assignments)

    if figsize is None:
        figsize = (10, max(3.5, num_machines * 0.7 + 1.0))

    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    fig.patch.set_facecolor("#1a1a2e")
    _setup_axes(ax, fig)

    busy_times = []
    idle_times = []
    unavail_times = []

    for m in range(num_machines):
        busy = sum(float(processing_times[j]) for j in assignments[m])
        unavail = sum(float(end - start) for start, end, _ in unavailability.get(m, []) if end <= makespan)
        idle = max(0, makespan - busy - unavail)
        busy_times.append(busy)
        idle_times.append(idle)
        unavail_times.append(unavail)

    y = list(range(num_machines))
    bar_height = 0.45

    ax.barh(y, busy_times, bar_height, label="Busy", color="#4CAF50", alpha=0.9)
    if any(u > 0 for u in unavail_times):
        ax.barh(y, unavail_times, bar_height, left=busy_times,
                label="Unavailable", color="#ff1744", alpha=0.5, hatch="///")
    idle_left = [b + u for b, u in zip(busy_times, unavail_times)]
    ax.barh(y, idle_times, bar_height, left=idle_left,
            label="Idle", color="#757575", alpha=0.5)

    for i in range(num_machines):
        total = busy_times[i] + unavail_times[i] + idle_times[i]
        if total > 0:
            if busy_times[i] >= total * 0.1:
                ax.text(
                    busy_times[i] / 2, i,
                    f"{busy_times[i] / total * 100:.0f}%",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                )
            if idle_times[i] >= total * 0.08:
                ax.text(
                    idle_left[i] + idle_times[i] / 2, i,
                    f"{idle_times[i] / total * 100:.0f}%",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"Machine {i + 1}" for i in range(num_machines)],
        fontsize=10, fontweight="bold", color="white",
    )
    ax.set_xlabel("Time", fontsize=10, fontweight="bold", color="white", labelpad=8)
    title_text = "Machine Utilization Breakdown"
    if algorithm_name:
        title_text += f" - {algorithm_name}"
    ax.set_title(
        title_text,
        fontsize=12, fontweight="bold", color="#64ffda", pad=12,
    )
    ax.legend(
        fontsize=9, facecolor="#1a1a2e", edgecolor="#64ffda",
        labelcolor="white", loc="upper right",
    )
    ax.grid(True, axis="x", alpha=0.15, color="#ffffff", linestyle="--")
    ax.set_xlim(0, makespan * 1.08)

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.88)
    return _fig_to_base64(fig)
