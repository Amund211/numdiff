"""
Settings for the project
"""

import sys

# Path to the folder where the plots will be saved
IMAGES_FOLDER = "images"

# Kwargs for Task.run in main.py
# Set save=True if you want to save the plot to the images folder
# Set show=True if you want to plt.show
TASK_KWARGS = {"save": True, "show": False}

# Print info when running tasks
INFO_PRINTING = True

# Use fine parameters that take longer to run
FINE_PARAMETERS = False

# Use multiprocessing for running multiple tasks at once
USE_MULTIPROCESSING = False

if INFO_PRINTING and USE_MULTIPROCESSING:
    print(
        "WARNING: Using INFO_PRINTING and USE_MULTIPROCESSING at the same tile will "
        "cause garbled output to the terminal due to interleaving.",
        file=sys.stderr,
    )
