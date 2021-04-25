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
    task_6d_4th_order,
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
from tasks.task import IMAGES_FOLDER, Task

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

    available_tasks = {
        "1_solution": Task(task_1_solution, "1_solution.pdf"),
        "1a_diririchlet_neumann": Task(task_1a, "1a_dirichlet_neumann.pdf"),
        "1b_dirichlet_dirichlet": Task(task_1b, "1b_dirichlet_dirichlet.pdf"),
        "1d_amr_1st_vs_2nd_order": Task(task_1d1, "1d_amr_1st_vs_2nd_order.pdf"),
        "1d_umr_vs_amr": Task(task_1d2, "1d_umr_vs_amr.pdf"),
        "2_solution": Task(task_2_solution, "2_solution.pdf"),
        "2a_bc_order": Task(task_2a, "2a_bc_order.pdf"),
        "2b_h_refinement": Task(task_2bh, "2b_h_refinement.pdf"),
        "2b_k_refinement": Task(task_2bk, "2b_k_refinement.pdf"),
        "2b_c_refinement": Task(task_2bc, "2b_c_refinement.pdf"),
        "2b_r_refinement": Task(task_2br, "2b_r_refinement.pdf"),
        "2c_burgers_breaking": Task(task_2c, "2c_burgers_breaking.pdf"),
        "3_solution": Task(task_3_solution, "3_solution.pdf"),
        "3b_x_refinement": Task(task_3bx, "3b_x_refinement.pdf"),
        "3b_y_refinement": Task(task_3by, "3b_y_refinement.pdf"),
        "4_solution": Task(task_4_solution, "4_solution.pdf"),
        "4b_euler_vs_cn": Task(task_4b, "4b_methods.pdf"),
        "4c_norm": Task(task_4c, "4c_norm.pdf"),
        "5b_refinement": Task(task_5b_refinement, "5b_refinement.pdf"),
        "5b_afem": Task(task_5b_afem, "5b_afem.pdf"),
        "5c_refinement": Task(task_5c_refinement, "5c_refinement.pdf"),
        "5c_afem": Task(task_5c_afem, "5c_afem.pdf"),
        "5d_refinement": Task(task_5d_refinement, "5d_refinement.pdf"),
        "5d_afem": Task(task_5d_afem, "5d_afem.pdf"),
        "5e_refinement": Task(task_5e_refinement, "5e_refinement.pdf"),
        "5e_afem": Task(task_5e_afem, "5e_afem.pdf"),
        "6_solution": Task(task_6_solution, "6_solution.pdf"),
        "6a_refinement": Task(task_6a_refinement, "6a_refinement.pdf"),
        "6b_refinement": Task(task_6b_refinement, "6b_refinement.pdf"),
        "6b_asymptotic": Task(task_6b_asymptotic, "6b_asymptotic.pdf"),
        "6d_4th_order": Task(task_6d_4th_order, "6d_4th_order.pdf"),
        "6d_aperiodic": Task(task_6d_2nd_order_aperiodic, "6d_2nd_order_aperiodic.pdf"),
    }

    task_names = available_tasks.keys()

    assert all(
        task_name == task_name.lower() for task_name in task_names
    ), "Task names must be lower case"

    if len(sys.argv) > 1:
        requested_tasks = sys.argv[1:]
    else:
        print("Type 'all' to run all tasks. Type a prefix to run all tasks that match.")
        print("Available tasks:")
        print("\n".join(available_tasks))
        requested_tasks = (
            input("What tasks do you want to run? (space separated): ")
            .lower()
            .split(" ")
        )

    if len(requested_tasks) == 1 and requested_tasks[0] == "all":
        tasks = task_names
    else:
        # Include all tasks that start with any of the search-terms
        tasks = filter(
            lambda task: any(
                task.startswith(requested_task.lower())
                for requested_task in requested_tasks
            ),
            task_names,
        )

    for task in tasks:
        print(f"Running task {task}", file=sys.stderr)
        available_tasks[task].run(save=True, show=False)
