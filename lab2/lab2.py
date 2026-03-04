import collections
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ortools.sat.python import cp_model

FIXED_JOBS_DATA = [
    [(0, 5), (1, 8), (2, 12)],
    [(1, 4), (0, 7), (2, 3)],
    [(2, 10), (1, 2), (0, 6)],
    [(0, 8), (2, 5), (1, 9)],
    [(1, 5), (0, 6), (2, 4)],
    [(2, 7), (1, 8), (0, 3)],
    [(0, 4), (2, 9), (1, 5)],
    [(1, 8), (2, 6), (0, 7)],
    [(2, 5), (0, 8), (1, 6)],
    [(0, 7), (2, 4), (1, 9)],
    [(1, 3), (0, 11), (2, 5)],
    [(2, 8), (1, 6), (0, 4)],
    [(0, 6), (2, 7), (1, 3)],
    [(1, 9), (0, 5), (2, 8)]
]


def get_fixed_data(num_jobs):
    return FIXED_JOBS_DATA[:num_jobs]


def solve_cp_sat(jobs_data):
    model = cp_model.CpModel()
    machines_count = max(task[0] for job in jobs_data for task in job) + 1
    all_machines = range(machines_count)
    horizon = sum(task[1] for job in jobs_data for task in job)

    task_type = collections.namedtuple('task_type', 'start end interval')
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start_var, end_var, interval_var)
            machine_to_intervals[machine].append(interval_var)

    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)])
    model.Minimize(obj_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0

    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42

    start_time = time.time()
    status = solver.Solve(model)
    exec_time = time.time() - start_time

    makespan = solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None

    schedule = []
    if makespan:
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine, duration = task
                start_val = solver.Value(all_tasks[job_id, task_id].start)
                schedule.append({'job': job_id, 'machine': machine, 'start': start_val, 'duration': duration})

    return makespan, exec_time, schedule


def solve_rd_heuristic(jobs_data):
    start_time = time.time()
    num_jobs = len(jobs_data)
    num_machines = max(task[0] for job in jobs_data for task in job) + 1

    machine_ready_time = [0] * num_machines
    job_progress = [0] * num_jobs
    job_end_time = [0] * num_jobs
    remaining_work = [sum(task[1] for task in job) for job in jobs_data]

    schedule = []

    while True:
        available_ops = []
        for j in range(num_jobs):
            if job_progress[j] < len(jobs_data[j]):
                task_id = job_progress[j]
                machine, duration = jobs_data[j][task_id]
                score = (100.0 / duration) + remaining_work[j] - max(0, job_end_time[j] - machine_ready_time[machine])
                available_ops.append((score, j, machine, duration))

        if not available_ops:
            break

        available_ops.sort(reverse=True, key=lambda x: x[0])
        _, j, machine, duration = available_ops[0]

        start_t = max(machine_ready_time[machine], job_end_time[j])
        schedule.append({'job': j, 'machine': machine, 'start': start_t, 'duration': duration})

        end_t = start_t + duration
        machine_ready_time[machine] = end_t
        job_end_time[j] = end_t
        job_progress[j] += 1
        remaining_work[j] -= duration

    makespan = max(machine_ready_time)
    exec_time = time.time() - start_time
    return makespan, exec_time, schedule


def plot_gantt(schedule, title, figure_num):
    plt.figure(figure_num, figsize=(10, 4))
    colors = list(mcolors.TABLEAU_COLORS.values())

    for task in schedule:
        plt.barh(task['machine'], task['duration'], left=task['start'],
                 color=colors[task['job'] % len(colors)], edgecolor='black')
        plt.text(task['start'] + task['duration'] / 2, task['machine'],
                 f"J{task['job']}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    machines_count = max(t['machine'] for t in schedule) + 1
    plt.yticks(range(machines_count), [f'М {m}' for m in range(machines_count)])
    plt.xlabel('Час')
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

def run_experiments():
    sizes = [4, 7, 10, 14]
    cp_times = []
    rd_times = []


    for n in sizes:
        data = get_fixed_data(n)

        cp_mk, cp_t, cp_schedule = solve_cp_sat(data)
        rd_mk, rd_t, rd_schedule = solve_rd_heuristic(data)

        cp_times.append(cp_t)
        rd_times.append(rd_t)

        if n == 4:
            plot_gantt(cp_schedule, f'Діаграма Ганта: CP-SAT', 1)
            plot_gantt(rd_schedule, f'Діаграма Ганта: R&D Модель', 2)

    plt.figure(3, figsize=(10, 5))
    plt.plot(sizes, cp_times, marker='o', color='red', label='CP-SAT')
    plt.plot(sizes, rd_times, marker='s', color='green', label='R&D Багатокритеріальна')
    plt.title('Графік залежності')
    plt.xlabel('Кількість робіт')
    plt.ylabel('Час розрахунку')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    run_experiments()