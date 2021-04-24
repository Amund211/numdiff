"""
Main file for running the tasks

https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf
https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_2.pdf
"""

import matplotlib.pyplot as plt

from tasks.advection_diffusion import (
    task_6_solution,
    task_6a_refinement,
    task_6b_asymptotic,
    task_6b_refinement,
    task_6d_2nd_order_aperiodic,
    task_6d_4th_order_M,
    task_6d_4th_order_ndof,
)
from tasks.burgers import task_2c
from tasks.fem_poisson import (
    task_5b_afem,
    task_5b_refinement,
    task_5c_afem,
    task_5c_refinement,
    task_5d_afem,
    task_5d_refinement,
    task_5e_afem,
    task_5e_refinement,
)
from tasks.heat_eqn import (
    task_2_solution,
    task_2a,
    task_2bc,
    task_2bh,
    task_2bk,
    task_2br,
)
from tasks.kdv import task_4_solution, task_4b, task_4c
from tasks.laplace_2D import task_3_solution, task_3bx, task_3by
from tasks.poisson_1D import task_1_solution, task_1a, task_1b, task_1d1, task_1d2
from tasks.task import IMAGES_FOLDER, run_task

if __name__ == "__main__":
    import os.path
    import sys

    if not os.path.isdir(IMAGES_FOLDER):
        print(
            f"Images folder missing. Creating directory '{os.path.abspath(IMAGES_FOLDER)}'",
            file=sys.stderr,
        )
        os.mkdir(IMAGES_FOLDER)

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
        "1_solution",
        "1a",
        "1b",
        "1d1",
        "1d2",
        "2_solution",
        "2a",
        "2bh",
        "2bk",
        "2bc",
        "2br",
        "2c",
        "3_solution",
        "3bx",
        "3by",
        "4_solution",
        "4b",
        "4c",
        "5b_refinement",
        "5b_afem",
        "5c_refinement",
        "5c_afem",
        "5d_refinement",
        "5d_afem",
        "5e_refinement",
        "5e_afem",
        "6_solution",
        "6a_refinement",
        "6b_refinement",
        "6b_asymptotic",
        "6d_4th_order_ndof",
        "6d_4th_order_M",
        "6d_2nd_order_aperiodic",
    )

    if len(sys.argv) > 1:
        requested_tasks = sys.argv[1:]
    else:
        print(f"Available tasks: {' '.join(available_tasks)}")
        print("Type 'all' to run all tasks. Type a prefix to run all tasks that match.")
        requested_tasks = (
            input("What tasks do you want to run? (space separated): ")
            .lower()
            .split(" ")
        )

    if len(requested_tasks) == 1 and requested_tasks[0] == "all":
        tasks = available_tasks
    else:
        tasks = filter(
            lambda task: any(
                task.startswith(requested_task) for requested_task in requested_tasks
            ),
            available_tasks,
        )

    for task in tasks:
        if task not in available_tasks:
            print(f"Did not recognize task '{task}', skipping...", file=sys.stdout)
            continue

        print(f"Running task {task}")
        if task == "1_solution":
            run_task(task_1_solution, "1_solution.pdf", save=True, show=False)
        elif task == "1a":
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
        elif task == "2_solution":
            run_task(task_2_solution, "2_solution.pdf", save=True, show=False)
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
        elif task == "2c":
            run_task(task_2c, "2c_burgers_breaking.pdf", save=True, show=False)
        elif task == "3_solution":
            run_task(task_3_solution, "3_solution.pdf", save=True, show=False)
        elif task == "3bx":
            run_task(task_3bx, "3b_x_refinement.pdf", save=True, show=False)
        elif task == "3by":
            run_task(task_3by, "3b_y_refinement.pdf", save=True, show=False)
        elif task == "4_solution":
            run_task(task_4_solution, "4_solution.pdf", save=True, show=False)
        elif task == "4b":
            run_task(task_4b, "4b_methods.pdf", save=True, show=False)
        elif task == "4c":
            run_task(task_4c, "4c_norm.pdf", save=True, show=False)
        elif task == "5b_refinement":
            run_task(task_5b_refinement, "5b_refinement.pdf", save=True, show=False)
        elif task == "5b_afem":
            run_task(task_5b_afem, "5b_afem.pdf", save=True, show=False)
        elif task == "5c_refinement":
            run_task(task_5c_refinement, "5c_refinement.pdf", save=True, show=False)
        elif task == "5c_afem":
            run_task(task_5c_afem, "5c_afem.pdf", save=True, show=False)
        elif task == "5d_refinement":
            run_task(task_5d_refinement, "5d_refinement.pdf", save=True, show=False)
        elif task == "5d_afem":
            run_task(task_5d_afem, "5d_afem.pdf", save=True, show=False)
        elif task == "5e_refinement":
            run_task(task_5e_refinement, "5e_refinement.pdf", save=True, show=False)
        elif task == "5e_afem":
            run_task(task_5e_afem, "5e_afem.pdf", save=True, show=False)
        elif task == "6_solution":
            run_task(task_6_solution, "6_solution.pdf", save=True, show=False)
        elif task == "6a_refinement":
            run_task(task_6a_refinement, "6a_refinement.pdf", save=True, show=False)
        elif task == "6b_refinement":
            run_task(task_6b_refinement, "6b_refinement.pdf", save=True, show=False)
        elif task == "6b_asymptotic":
            run_task(task_6b_asymptotic, "6b_asymptotic.pdf", save=True, show=False)
        elif task == "6d_4th_order_ndof":
            run_task(
                task_6d_4th_order_ndof, "6d_4th_order_ndof.pdf", save=True, show=False
            )
        elif task == "6d_4th_order_M":
            run_task(task_6d_4th_order_M, "6d_4th_order_M.pdf", save=True, show=False)
        elif task == "6d_2nd_order_aperiodic":
            run_task(
                task_6d_2nd_order_aperiodic,
                "6d_2nd_order_aperiodic.pdf",
                save=True,
                show=False,
            )
        else:
            raise ValueError(
                f"Task '{task}' present in `available_tasks`, but not implemented"
            )
