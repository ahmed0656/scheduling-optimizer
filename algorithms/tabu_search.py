import numpy as np
import time
from collections import deque


class TabuSearch:
    """
    Tabu Search for Identical Parallel Machine Scheduling with
    Machine Unavailability Periods (Pm, h_ij | | Cmax).

    Neighborhood: move a job from one machine to another.
    Tabu: recent job-machine pairs are forbidden.
    Aspiration: override tabu if solution is better than best known.
    """

    def __init__(
        self,
        processing_times,
        num_machines,
        priorities=None,
        tabu_tenure=20,
        max_iterations=500,
        neighborhood_size=None,
        intensification_interval=50,
        unavailability_periods=None,
        seed=None,
    ):
        self.processing_times = np.array(processing_times, dtype=float)
        self.num_jobs = len(self.processing_times)
        self.num_machines = num_machines
        self.priorities = np.array(priorities if priorities is not None else [1] * self.num_jobs, dtype=int)
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size or max(5, self.num_jobs // 2)
        self.intensification_interval = intensification_interval
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
        """
        if not jobs_list:
            return 0.0, [], 0.0

        # Sort jobs by priority (higher first) so high-priority jobs are scheduled first
        sorted_jobs = sorted(jobs_list, key=lambda x: -self.priorities[x[0]])

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

    def _generate_neighborhood(self, current, tabu_list):
        """Generate neighborhood solutions and filter tabu moves."""
        # Group jobs by machine for efficient recomputation
        machine_jobs = [[] for _ in range(self.num_machines)]
        for job, machine in enumerate(current):
            machine_jobs[int(machine)].append((job, int(machine)))

        moves = []
        for job in range(self.num_jobs):
            current_machine = int(current[job])
            for m in range(self.num_machines):
                if m != current_machine:
                    tabu = False
                    for t_job, t_machine, _ in tabu_list:
                        if t_job == job and t_machine == m:
                            tabu = True
                            break
                    moves.append((job, m, tabu))

        # Limit neighborhood size for large problems
        if len(moves) > self.neighborhood_size * self.num_machines:
            self.rng.shuffle(moves)
            moves = moves[:self.neighborhood_size * self.num_machines]

        neighborhood = []
        for job, new_machine, is_tabu in moves:
            neighbor = current.copy()
            neighbor[job] = new_machine
            makespan, _, _, _, _ = self._compute_makespan(neighbor)
            neighborhood.append((makespan, neighbor, job, new_machine, is_tabu))

        return neighborhood

    def run(self, progress_callback=None):
        """Run Tabu Search."""
        start_time = time.time()

        # Initialize
        current = self._get_lpt_initial()
        current_makespan, _, _, _, _ = self._compute_makespan(current)

        best_solution = current.copy()
        best_makespan = current_makespan

        tabu_list = deque(maxlen=self.tabu_tenure)
        stagnation_counter = 0
        last_improvement = best_makespan

        for iteration in range(self.max_iterations):
            neighborhood = self._generate_neighborhood(current, tabu_list)
            neighborhood.sort(key=lambda x: x[0])

            best_move = None
            best_move_makespan = float("inf")

            for makespan, neighbor, job, new_machine, is_tabu in neighborhood:
                if makespan < best_makespan:
                    best_move = (neighbor, job, new_machine)
                    best_move_makespan = makespan
                    break
                elif not is_tabu and makespan < best_move_makespan:
                    best_move = (neighbor, job, new_machine)
                    best_move_makespan = makespan

            if best_move is None:
                best_move = (neighborhood[0][1], neighborhood[0][2], neighborhood[0][3]) if neighborhood else (current, -1, -1)
                best_move_makespan = neighborhood[0][0] if neighborhood else current_makespan

            neighbor, job, new_machine = best_move

            # Track old machine BEFORE updating current
            old_machine = current[job] if job >= 0 else -1

            current = neighbor
            current_makespan = best_move_makespan

            # Update tabu list: forbid moving back to the old machine
            if job >= 0 and old_machine >= 0:
                tabu_list.append((job, old_machine, iteration))

            # Update best
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = current.copy()
                stagnation_counter = 0
                last_improvement = best_makespan
            else:
                stagnation_counter += 1

            self.history.append(best_makespan)

            # Intensification: restart from best if stagnating
            if stagnation_counter >= self.intensification_interval:
                current = best_solution.copy()
                current_makespan, _, _, _, _ = self._compute_makespan(current)
                stagnation_counter = 0
                for _ in range(5):
                    idx = self.rng.randint(0, self.num_jobs)
                    current[idx] = self.rng.randint(0, self.num_machines)
                current_makespan, _, _, _, _ = self._compute_makespan(current)

            if progress_callback:
                progress_callback(iteration, self.max_iterations, best_makespan)

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
            "algorithm": "Tabu Search",
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
                "tabu_tenure": self.tabu_tenure,
                "max_iterations": self.max_iterations,
                "neighborhood_size": self.neighborhood_size,
                "intensification_interval": self.intensification_interval,
            },
        }
