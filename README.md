# Numdiff

## Installation and setup
To set up the project create a virtual environment, activate the environment, and install the dependencies.

```shell
python -m virtualenv venv
source venv/bin/activate  # Or the equivalent for your shell
pip install -r requirements.txt
```

You may need to install `swig` to be able to build `scikit-umfpack`.
Alternatively you may remove it as a dependency, as it isn't used explicitly, but may be used by the solvers in `scipy.sparse.linalg` if present.
