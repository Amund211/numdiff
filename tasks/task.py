import matplotlib.pyplot as plt


def run_task(task, filename, save, show):
    """Run the provided task and save and/or show the result"""
    plt.clf()
    task()
    if save:
        plt.savefig(f"images/{filename}")
    if show:
        plt.show()
