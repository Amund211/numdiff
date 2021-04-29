# Numdiff

## Installation and setup
To set up the project create a virtual environment, activate the environment, and install the dependencies.

```shell
python3 --version  # Ensure you have at least python 3.7 (May work with earlier versions, but not tested)
python3 -m virtualenv venv --python=python3
source venv/bin/activate  # Or the equivalent for your shell
pip3 install -r requirements.txt
```

You may need to install `swig` to be able to build `scikit-umfpack`.
Alternatively you may remove it as a dependency, as it isn't used explicitly, but may be used by the solvers in `scipy.sparse.linalg` if present.
Do note that during testing I had some issues with memory usage using the default solvers from scipy, while the umfpack solvers didn't seem to have that issue.

## Configuring the project
[`settings.py`](./settings.py) contains some variables that are read by the rest of the project that may be changed to alter the behaviour of the code.
These are documented in the file itself.
The most important of these may be `FINE_PARAMETERS` which toggles whether the tasks will use parameters that are easy to run (< 1 min), or that give the most detailed plots (>> 5 min).

## Running the code
Running `python main.py` while in the virtual environment prompts you to enter the tasks you want to run.
To run several tasks at once you may type a common prefix of the tasks you want (e.g. `1` to run all tasks related to task 1), and you may specify several prefixes space separated.
To run the code non-interactively you may specify these parameters as commandline arguments instead of at the prompt.

## Part 2 - task 3: The advection diffusion equation
Note: Part 2 - task 3 is in this project renamed to task 6 to fit the scheme.

To run the code comparison task (part 2 - task 3c) run task `6c_runtime` e.g. `python main.py 6c`.
This will generate a plot in the [`images`](./images) folder with two figures: l2 error vs degrees of freedom and runtime vs degrees of freedom.

## Editing the tasks
The tasks are loaded into [`main.py`](./main.py) from the [tasks](./tasks) directory.
Each major task (1, 2, ...) has its own file in the [tasks](./tasks), while the functions for the different subtasks lie inside those files.
The taskname used in [`main.py`](./main.py) should, for the most part, coincide with the names of the functions.

To edit the behaviour or the parameters of any task, simply edit the corresponding function.
