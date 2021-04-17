import os.path

import matplotlib.pyplot as plt

IMAGES_FOLDER = "images"


def run_task(task, filename, save, show):
    """Run the provided task and save and/or show the result"""
    plt.clf()
    task()
    if save:
        plt.savefig(os.path.join(IMAGES_FOLDER, filename))
    if show:
        plt.show()
