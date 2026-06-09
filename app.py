import sys
import os
import json
import time
import base64
import io
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.tabu_search import TabuSearch
from web_utils.gantt_chart import (
    create_gantt_chart, create_convergence_plot,
    create_comparison_bar_chart, create_machine_utilization_chart,
)

app = Flask(__name__)


def get_unavailability(unavailability_data, num_machines):
    unavailability = {m: [] for m in range(num_machines)}
    for entry in unavailability_data:
        try:
            machine_str = entry.get("machine", "")
            machine_idx = int(machine_str.replace("M", "")) - 1
            start = float(entry["start"])
            end = float(entry["end"])
            reason = entry.get("reason", "N/A")
            if 0 <= machine_idx < num_machines:
                unavailability[machine_idx].append((start, end, reason))
        except (ValueError, KeyError):
            continue
    for m in unavailability:
        unavailability[m].sort(key=lambda x: x[0])
    return unavailability


def compute_lower_bound(processing_times, num_machines):
    p_max = max(processing_times)
    p_sum = sum(processing_times)
    return max(p_max, p_sum / num_machines)


def run_single_algorithm(algo_name, processing_times, num_machines, priorities, params):
    unavailability = get_unavailability(params.get("unavailability", []), num_machines)
    if priorities is None:
        priorities = [1] * len(processing_times)

    if algo_name == "GA":
        algo = GeneticAlgorithm(
            processing_times=processing_times,
            num_machines=num_machines,
            priorities=priorities,
            population_size=params.get("ga_pop", 100),
            generations=params.get("ga_gen", 200),
            crossover_rate=params.get("ga_cx", 0.85),
            mutation_rate=params.get("ga_mut", 0.15),
            unavailability_periods=unavailability,
        )
    elif algo_name == "SA":
        algo = SimulatedAnnealing(
            processing_times=processing_times,
            num_machines=num_machines,
            priorities=priorities,
            initial_temp=params.get("sa_temp", 1000),
            cooling_rate=params.get("sa_cool", 0.995),
            iterations_per_temp=params.get("sa_iter", 100),
            unavailability_periods=unavailability,
        )
    elif algo_name == "TS":
        algo = TabuSearch(
            processing_times=processing_times,
            num_machines=num_machines,
            priorities=priorities,
            tabu_tenure=params.get("ts_tenure", 20),
            max_iterations=params.get("ts_iter", 500),
            unavailability_periods=unavailability,
        )
    else:
        return None

    def callback(current, total, value):
        pass

    return algo.run(progress_callback=callback)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json()
    processing_times = data.get("processing_times", [])
    priorities = data.get("priorities", [1] * len(processing_times))
    num_machines = data.get("num_machines", 3)
    algorithms = data.get("algorithms", ["GA", "SA", "TS"])
    params = data.get("params", {})

    if not processing_times or all(t == 0 for t in processing_times):
        return jsonify({"error": "No valid processing times"}), 400

    results = {}
    for algo_name in algorithms:
        result = run_single_algorithm(algo_name, processing_times, num_machines, priorities, params)
        if result:
            # Clean non-serializable data
            for key in list(result.keys()):
                if isinstance(result[key], np.integer):
                    result[key] = int(result[key])
                elif isinstance(result[key], np.floating):
                    result[key] = float(result[key])
                elif isinstance(result[key], np.ndarray):
                    result[key] = result[key].tolist()
            results[algo_name] = result

    if not results:
        return jsonify({"error": "No algorithms produced results"}), 500

    # Generate chart images
    results_list = list(results.values())
    lb = compute_lower_bound(processing_times, num_machines)

    charts = {}
    best_algo = min(results.keys(), key=lambda k: results[k]["makespan"])

    # Gantt charts (one per algorithm)
    charts["gantt"] = {}
    for name, result in results.items():
        is_best = (name == best_algo)
        title = f"Best Schedule - {result['algorithm']}" if is_best else result["algorithm"]
        charts["gantt"][name] = create_gantt_chart(result, title=title)

    # Convergence
    if results_list:
        charts["convergence"] = create_convergence_plot(results_list)

    # Comparison (if multiple)
    if len(results_list) > 1:
        charts["comparison"] = create_comparison_bar_chart(results_list)

    # Utilization
    charts["utilization"] = {}
    for name, result in results.items():
        is_best = (name == best_algo)
        algo_name_display = result["algorithm"]
        charts["utilization"][name] = create_machine_utilization_chart(result, algorithm_name=algo_name_display)

    # Serialize results for JSON response
    serializable_results = {}
    for name, result in results.items():
        serializable_results[name] = {
            "algorithm": result["algorithm"],
            "makespan": float(result["makespan"]),
            "machine_loads": [float(x) for x in result["machine_loads"]],
            "assignments": [[int(j) for j in jobs] for jobs in result["assignments"]],
            "execution_time": float(result["execution_time"]),
            "history": [float(x) for x in result.get("history", [])],
            "total_unavailable_time": float(result.get("total_unavailable_time", 0)),
            "unavailability": result.get("unavailability", {}),
            "processing_times": [float(x) for x in result.get("processing_times", processing_times)],
            "priorities": [int(x) for x in result.get("priorities", priorities)],
            "job_start_times": result.get("job_start_times", {}),
            "parameters": result.get("parameters", {}),
        }

    response_data = {
        "results": serializable_results,
        "charts": charts,
        "lower_bound": float(lb),
        "best_algorithm": best_algo,
    }

    return jsonify(response_data)


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json()
    fmt = data.get("format", "json")
    results = data.get("results", {})

    if fmt == "json":
        return jsonify(results)
    elif fmt == "csv":
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Algorithm", "Makespan", "Execution Time (s)", "Parameters", "Priorities"])
        for name, result in results.items():
            priorities = result.get("priorities", [])
            pri_str = ", ".join([f"J{j+1}:P{p}" for j, p in enumerate(priorities)])
            writer.writerow([
                result["algorithm"],
                result["makespan"],
                result["execution_time"],
                str(result.get("parameters", {})),
                pri_str,
            ])
        mem = io.BytesIO()
        mem.write(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="results.csv")

    return jsonify({"error": "Unsupported format"}), 400


if __name__ == "__main__":
    print("=" * 60)
    print("  Parallel Machine Scheduling Optimizer - Web Version")
    print("  Open in browser: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(5000)
        print(f"\n  * Public URL (للمشاركة): {public_url}")
        print("  * شارك هذا الرابط مع صديقك (أي شبكة)")
    except ImportError:
        print("\n  * لمشاركة صديقك من أي شبكة: pip install pyngrok")
        print("  * أو استخدم: https://ngrok.com/download - شغل: ngrok http 5000")
    except Exception as e:
        print(f"\n  * تعذر تشغيل ngrok: {e}")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
