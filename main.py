"""
Main file for running the tasks

https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf
https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_2.pdf
"""

import matplotlib.pyplot as plt

from tasks.fem_poisson import task_5b_avg, task_5b_max, task_5b_refinement
from tasks.heat_eqn import task_2a, task_2bc, task_2bh, task_2bk, task_2br
from tasks.kdv import task_4b, task_4c
from tasks.laplace_2D import task_3bx, task_3by
from tasks.poisson_1D import task_1a, task_1b, task_1d1, task_1d2
from tasks.task import run_task

if __name__ == "__main__":
    import sys

    # Plot params
    fontsize = 16
    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "legend.fontsize": fontsize,
            "legend.handlelength": 1.5,
            "figure.figsize": (10, 6),
            "figure.titlesize": 20,
        }
    )

    available_tasks = (
        "1a",
        "1b",
        "1d1",
        "1d2",
        "2a",
        "2bh",
        "2bk",
        "2bc",
        "2br",
        "3bx",
        "3by",
        "4b",
        "4c",
        "5b_refinement",
        "5b_avg",
        "5b_max",
    )

    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        print(f"Available tasks: {' '.join(available_tasks)}")
        print("Use 'all' to run all tasks")
        tasks = (
            input("What tasks do you want to run? (space separated): ")
            .lower()
            .split(" ")
        )

    if len(tasks) == 1 and tasks[0] == "all":
        tasks = available_tasks

    for task in tasks:
        if task not in available_tasks:
            print(f"Did not recognize task '{task}', skipping...", file=sys.stdout)
            continue

        print(f"Running task {task}")
        if task == "1a":
            run_task(task_1a, "1a_dirichlet_neumann.pdf", save=True, show=False)
        elif task == "1b":
            run_task(task_1b, "1b_dirichlet_dirichlet.pdf", save=True, show=False)
        elif task == "1d1":
            run_task(task_1d1, "1d_amr_1st_vs_2nd_order.pdf", save=True, show=False)
        elif task == "1d2":
            run_task(task_1d2, "1d_umr_vs_amr.pdf", save=True, show=False)
        elif task == "1d3":
            # Maybe include a concrete example comparing AMR with UMR for some small M?
            pass
        elif task == "2a":
            run_task(task_2a, "2a_bc_order.pdf", save=True, show=False)
        elif task == "2bh":
            run_task(task_2bh, "2b_h_refinement.pdf", save=True, show=False)
        elif task == "2bk":
            run_task(task_2bk, "2b_k_refinement.pdf", save=True, show=False)
        elif task == "2bc":
            run_task(task_2bc, "2b_c_refinement.pdf", save=True, show=False)
        elif task == "2br":
            run_task(task_2br, "2b_r_refinement.pdf", save=True, show=False)
        elif task == "3bx":
            run_task(task_3bx, "3b_x_refinement.pdf", save=True, show=False)
        elif task == "3by":
            run_task(task_3by, "3b_y_refinement.pdf", save=True, show=False)
        elif task == "4b":
            run_task(task_4b, "4b_methods.pdf", save=True, show=False)
        elif task == "4c":
            run_task(task_4c, "4c_norm.pdf", save=True, show=False)
        elif task == "5b_refinement":
            run_task(task_5b_refinement, "5b_refinement.pdf", save=True, show=False)
        elif task == "5b_avg":
            run_task(task_5b_avg, "5b_AFEM_avg.pdf", save=True, show=False)
        elif task == "5b_max":
            run_task(task_5b_max, "5b_AFEM_max.pdf", save=True, show=False)
        else:
            raise ValueError(
                f"Task '{task}' present in `available_tasks`, but not implemented"
            )
