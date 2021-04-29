import os.path
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt

from numdiff.settings import IMAGES_FOLDER


@dataclass
class Task:
    task: Callable
    filename: str

    def run(self, save, show):
        """Run the task and save and/or show the result"""
        plt.clf()
        self.task()
        if save:
            plt.savefig(os.path.join(IMAGES_FOLDER, self.filename))
        if show:
            plt.show()
