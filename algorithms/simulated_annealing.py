import numpy as np
import math
import time


class SimulatedAnnealing:
    """
    Simulated Annealing for Identical Parallel Machine Scheduling with
    Machine Unavailability Periods (Pm, h_ij | | Cmax).

    Neighbor generation: move a random job to a different machine.
    Cooling: geometric cooling schedule.
    """

    def __init__(
        self,
        processing_times,
        num_machines,
        priorities=None,
        initial_temp=1000,
        cooling_rate=0.995,
        min_temp=0.001,
        iterations_per_temp=100,
        unavailability_periods=None,
        seed=None,
    ):
        self.processing_times = np.array(processing_times, dtype=float)
        self.num_jobs = len(self.processing_times)
        self.num_machines = num_machines
        self.priorities = np.array(priorities if priorities is not None else [1] * self.num_jobs, dtype=int)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
        self.rng = np.random.RandomState(seed)
        self.history = []

        # Unavailability periods: dict {machine_idx: [(start, end, reason), ...]}
        self.unavailability = unavailability_periods or {}
        self.unavailability = {int(k): v for k, v in self.unavailability.items()}
        for m in range(self.num_machines):
            if m not in self.unavailability:
                self.unavailability[m] = []
            self.unavailability[m].sort(key=lambda x: x[0])

    def _compute_completion_time(self, jobs_list):
        """
        Compute completion time for jobs on a single machine,
        accounting for unavailability periods and job priority.
        Higher priority jobs are scheduled first.
        jobs_list: list of (job_idx, machine_idx) tuples
        Returns (completion_time, job_start_times, unavailable_time_in_span)
        """
        if not jobs_list:
            return 0.0, [], 0.0

        # Sort jobs by priority (higher first) so high-priority jobs are scheduled first
        sorted_jobs = sorted(jobs_list, key=lambda x: -self.priorities[x[0]])

        # Build merged unavailability intervals
        machine = sorted_jobs[0][1]
        unavailable = self.unavailability.get(machine, [])
        merged = []
        for start, end, reason in unavailable:
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
            else:
                merged.append([start, end, reason])

        current_time = 0.0
        total_unavailable_time = 0.0
        job_start_times = []

        for job, _ in sorted_jobs:
            duration = self.processing_times[job]
            while True:
                end_time = current_time + duration
                overlaps = False
                for u_start, u_end, _ in merged:
                    if current_time < u_end and end_time > u_start:
                        current_time = u_end
                        overlaps = True
                        break
                if not overlaps:
                    break

            job_start_times.append((job, current_time, duration))
            current_time = end_time

        for u_start, u_end, _ in merged:
            if u_end <= current_time:
                total_unavailable_time += (u_end - u_start)
            elif u_start < current_time:
                total_unavailable_time += (current_time - u_start)

        return current_time, job_start_times, total_unavailable_time

    def _compute_makespan(self, assignment):
        """Compute makespan from machine assignment array with unavailability."""
        loads = np.zeros(self.num_machines)
        assignments = [[] for _ in range(self.num_machines)]
        job_start_times = {}
        total_unavail = 0.0

        machine_jobs = [[] for _ in range(self.num_machines)]
        for job, machine in enumerate(assignment):
            m = int(machine)
            machine_jobs[m].append((job, m))

        for m in range(self.num_machines):
            if machine_jobs[m]:
                completion, starts, unavail = self._compute_completion_time(machine_jobs[m])
                loads[m] = completion
                total_unavail += unavail
                for job, start, dur in starts:
                    assignments[m].append(job)
                    job_start_times[job] = (m, start, dur)

        makespan = float(np.max(loads)) if len(loads) > 0 else 0
        return makespan, loads, assignments, job_start_times, total_unavail

    def _get_neighbor(self, current):
        """Generate neighbor by moving a random job to a random different machine."""
        neighbor = current.copy()
        job = self.rng.randint(0, self.num_jobs)
        new_machine = self.rng.randint(0, self.num_machines)
        while new_machine == neighbor[job] and self.num_machines > 1:
            new_machine = self.rng.randint(0, self.num_machines)
        neighbor[job] = new_machine
        return neighbor

    def _get_neighbor_swap(self, current):
        """Generate neighbor by swapping two jobs between machines."""
        neighbor = current.copy()
        job1 = self.rng.randint(0, self.num_jobs)
        job2 = self.rng.randint(0, self.num_jobs)
        while job2 == job1:
            job2 = self.rng.randint(0, self.num_jobs)
        neighbor[job1], neighbor[job2] = current[job2], current[job1]
        return neighbor

    def _get_lpt_initial(self):
        """Generate initial solution using LPT rule with unavailability awareness and priority."""
        sorted_jobs = np.lexsort((-self.processing_times, -self.priorities))
        machine_job_lists = [[] for _ in range(self.num_machines)]
        assignment = np.zeros(self.num_jobs, dtype=int)

        for job in sorted_jobs:
            best_makespan = float("inf")
            best_machine = 0
            for m in range(self.num_machines):
                test_list = machine_job_lists[m] + [(job, m)]
                ct, _, _ = self._compute_completion_time(test_list)
                if ct < best_makespan:
                    best_makespan = ct
                    best_machine = m
            assignment[job] = best_machine
            machine_job_lists[best_machine].append((job, best_machine))

        return assignment

    def run(self, progress_callback=None):
        """Run Simulated Annealing."""
        start_time = time.time()

        # Initialize with LPT
        current = self._get_lpt_initial()
        current_makespan, _, _, _, _ = self._compute_makespan(current)

        best_solution = current.copy()
        best_makespan = current_makespan

        temperature = self.initial_temp
        total_iterations = 0
        temp_count = 0

        while temperature > self.min_temp:
            temp_count += 1
            for i in range(self.iterations_per_temp):
                total_iterations += 1
                if self.rng.random() < 0.7:
                    neighbor = self._get_neighbor(current)
                else:
                    neighbor = self._get_neighbor_swap(current)

                neighbor_makespan, _, _, _, _ = self._compute_makespan(neighbor)
                delta = neighbor_makespan - current_makespan

                if delta < 0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-10)):
                    current = neighbor.copy()
                    current_makespan = neighbor_makespan

                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_solution = current.copy()

            temperature *= self.cooling_rate
            self.history.append(best_makespan)

            if progress_callback:
                progress_callback(temp_count, int(self.iterations_per_temp * 100), best_makespan)

        elapsed = time.time() - start_time
        makespan, loads, assignments, job_starts, unavail_time = self._compute_makespan(best_solution)

        schedule = {}
        for m in range(self.num_machines):
            schedule[m] = []
            for job in assignments[m]:
                if job in job_starts:
                    _, start, dur = job_starts[job]
                    schedule[m].append({"job": job, "start": float(start), "duration": float(dur)})

        return {
            "algorithm": "Simulated Annealing",
            "makespan": float(makespan),
            "machine_loads": [float(x) for x in loads.tolist()],
            "assignments": assignments,
            "job_start_times": schedule,
            "unavailability": self.unavailability,
            "total_unavailable_time": float(unavail_time),
            "processing_times": [float(x) for x in self.processing_times.tolist()],
            "priorities": [int(x) for x in self.priorities.tolist()],
            "execution_time": elapsed,
            "history": [float(x) for x in self.history],
            "parameters": {
                "initial_temperature": self.initial_temp,
                "cooling_rate": self.cooling_rate,
                "min_temperature": self.min_temp,
                "iterations_per_temp": self.iterations_per_temp,
            },
        }
