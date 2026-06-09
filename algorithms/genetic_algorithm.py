import numpy as np
import random
import time
from copy import deepcopy


class GeneticAlgorithm:
    """
    Genetic Algorithm for Identical Parallel Machine Scheduling with
    Machine Unavailability Periods (Pm, h_ij | | Cmax).

    Chromosome encoding: job-to-machine assignment vector.
    Fitness: inverse of makespan (we want to minimize makespan).
    """

    def __init__(
        self,
        processing_times,
        num_machines,
        priorities=None,
        population_size=100,
        generations=200,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_rate=0.10,
        unavailability_periods=None,
        seed=None,
    ):
        self.processing_times = np.array(processing_times, dtype=float)
        self.num_jobs = len(self.processing_times)
        self.num_machines = num_machines
        self.priorities = np.array(priorities if priorities is not None else [1] * self.num_jobs, dtype=int)
        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(elitism_rate * population_size))
        self.rng = np.random.RandomState(seed)
        self.history = []

        # Unavailability periods: dict {machine_idx: [(start, end, reason), ...]}
        self.unavailability = unavailability_periods or {}
        self.unavailability = {int(k): v for k, v in self.unavailability.items()}

        # Ensure all machines have an entry
        for m in range(self.num_machines):
            if m not in self.unavailability:
                self.unavailability[m] = []
            # Sort by start time
            self.unavailability[m].sort(key=lambda x: x[0])

    def _compute_completion_time(self, jobs_on_machine):
        """
        Compute the completion time for a list of jobs on a machine,
        accounting for unavailability periods and job priority.
        Higher priority jobs are scheduled first.
        Returns (completion_time, job_start_times, total_idle_from_unavailability).
        """
        if not jobs_on_machine:
            return 0.0, [], 0.0

        # Sort jobs by priority (higher first) so high-priority jobs are scheduled first
        sorted_jobs = sorted(jobs_on_machine, key=lambda x: -self.priorities[x[0]])

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
            # Find the earliest start time that doesn't overlap with unavailability
            while True:
                end_time = current_time + duration
                overlaps = False
                for u_start, u_end, _ in merged:
                    if current_time < u_end and end_time > u_start:
                        # Overlap found, skip past this unavailability
                        current_time = u_end
                        overlaps = True
                        break
                if not overlaps:
                    break

            job_start_times.append((job, current_time, duration))
            current_time = end_time

        # Calculate total unavailable time within the makespan
        for u_start, u_end, _ in merged:
            if u_end <= current_time:
                total_unavailable_time += (u_end - u_start)
            elif u_start < current_time:
                total_unavailable_time += (current_time - u_start)

        return current_time, job_start_times, total_unavailable_time

    def _decode(self, chromosome):
        """
        Decode chromosome to machine assignments and compute makespan
        considering unavailability periods.
        """
        machine_loads = np.zeros(self.num_machines)
        assignments = [[] for _ in range(self.num_machines)]
        job_start_times = {}
        total_unavailable_time = 0.0

        # Group jobs by machine
        machine_jobs = [[] for _ in range(self.num_machines)]
        for job, machine in enumerate(chromosome):
            m = int(machine)
            machine_jobs[m].append((job, m))

        # Compute completion time for each machine
        for m in range(self.num_machines):
            if machine_jobs[m]:
                completion, starts, unavail = self._compute_completion_time(machine_jobs[m])
                machine_loads[m] = completion
                total_unavailable_time += unavail
                for job, start, duration in starts:
                    assignments[m].append(job)
                    job_start_times[job] = (m, start, duration)
            else:
                assignments[m] = []

        makespan = float(np.max(machine_loads)) if len(machine_loads) > 0 else 0
        return makespan, machine_loads, assignments, job_start_times, total_unavailable_time

    def _initialize_population(self):
        """Create initial population using random + LPT heuristic."""
        population = []
        # 70% random, 30% LPT-based
        random_count = int(0.7 * self.pop_size)
        for _ in range(random_count):
            population.append(self.rng.randint(0, self.num_machines, size=self.num_jobs))

        # LPT initialization considering priority and unavailability
        # Sort by priority (descending), then by processing time (descending)
        loads = np.zeros(self.num_machines)
        lpt_chromosome = np.zeros(self.num_jobs, dtype=int)
        sorted_jobs = np.lexsort((-self.processing_times, -self.priorities))
        machine_job_lists = [[] for _ in range(self.num_machines)]

        for job in sorted_jobs:
            # Find machine with earliest completion time
            best_makespan = float("inf")
            best_machine = 0
            for m in range(self.num_machines):
                test_list = machine_job_lists[m] + [(job, m)]
                ct, _, _ = self._compute_completion_time(test_list)
                if ct < best_makespan:
                    best_makespan = ct
                    best_machine = m
            lpt_chromosome[job] = best_machine
            machine_job_lists[best_machine].append((job, best_machine))

        population.append(lpt_chromosome.copy())

        # Add variations of LPT with small mutations
        for _ in range(self.pop_size - random_count - 1):
            chrom = lpt_chromosome.copy()
            if self.rng.random() < 0.3:
                idx = self.rng.randint(0, self.num_jobs)
                chrom[idx] = self.rng.randint(0, self.num_machines)
            population.append(chrom)

        return np.array(population)

    def _tournament_selection(self, population, fitness, tournament_size=5):
        """Tournament selection."""
        indices = self.rng.choice(len(population), size=tournament_size, replace=False)
        best_idx = indices[np.argmax(fitness[indices])]
        return population[best_idx].copy()

    def _uniform_crossover(self, parent1, parent2):
        """Uniform crossover."""
        mask = self.rng.random(self.num_jobs) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(self, chromosome):
        """Swap mutation: randomly reassign some jobs to different machines."""
        mutated = chromosome.copy()
        for i in range(self.num_jobs):
            if self.rng.random() < self.mutation_rate / self.num_jobs:
                new_machine = self.rng.randint(0, self.num_machines)
                mutated[i] = new_machine
        return mutated

    def _local_search(self, chromosome):
        """Simple local search: try moving each job to every other machine."""
        current_makespan, _, _, _, _ = self._decode(chromosome)
        improved = True
        max_iter = 20
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for job in range(self.num_jobs):
                original = chromosome[job]
                best_machine = original
                for m in range(self.num_machines):
                    if m != original:
                        chromosome[job] = m
                        new_makespan, _, _, _, _ = self._decode(chromosome)
                        if new_makespan < current_makespan:
                            current_makespan = new_makespan
                            best_machine = m
                            improved = True
                chromosome[job] = best_machine
        return chromosome

    def run(self, progress_callback=None):
        """Run the Genetic Algorithm."""
        start_time = time.time()
        population = self._initialize_population()
        best_overall = None
        best_fitness_overall = -np.inf

        for gen in range(self.generations):
            # Evaluate fitness
            fitness = np.array([
                1.0 / self._decode(ind)[0] for ind in population
            ])

            # Track best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_fitness_overall:
                best_fitness_overall = fitness[best_idx]
                best_overall = population[best_idx].copy()

            # Elitism
            elite_indices = np.argsort(fitness)[-self.elitism_count:]
            new_population = [population[i].copy() for i in elite_indices]

            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)

                if self.rng.random() < self.crossover_rate:
                    child = self._uniform_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                child = self._mutate(child)
                new_population.append(child)

            population = np.array(new_population)

            # Apply local search to best every 10 generations
            if gen % 10 == 0:
                best_idx = np.argmax(fitness)
                population[best_idx] = self._local_search(population[best_idx].copy())
                new_makespan, _, _, _, _ = self._decode(population[best_idx])
                if 1.0 / new_makespan > best_fitness_overall:
                    best_fitness_overall = 1.0 / new_makespan
                    best_overall = population[best_idx].copy()

            makespan, _, _, _, _ = self._decode(best_overall)
            self.history.append(makespan)

            if progress_callback:
                progress_callback(gen, self.generations, makespan)

        elapsed = time.time() - start_time
        makespan, loads, assignments, job_starts, unavail_time = self._decode(best_overall)

        # Build job start times for schedule
        schedule = {}
        for m in range(self.num_machines):
            schedule[m] = []
            for job in assignments[m]:
                if job in job_starts:
                    _, start, dur = job_starts[job]
                    schedule[m].append({"job": job, "start": float(start), "duration": float(dur)})

        return {
            "algorithm": "Genetic Algorithm",
            "makespan": float(makespan),
            "machine_loads": [float(x) for x in loads.tolist()],
            "assignments": assignments,
            "job_start_times": schedule,
            "unavailability": self.unavailability,
            "total_unavailable_time": float(unavail_time),
            "processing_times": [float(x) for x in self.processing_times.tolist()],
            "priorities": [int(x) for x in self.priorities.tolist()],
            "execution_time": elapsed,
            "history": [float(h) for h in self.history],
            "parameters": {
                "population_size": self.pop_size,
                "generations": self.generations,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "elitism_rate": self.elitism_count / self.pop_size,
            },
        }
