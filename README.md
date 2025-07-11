# Quick EDA `qeda`

## TODO

- Compare 2 dataset to detect covarian-shift
- Extract partial dependence
- Train model with paratune
- Get sample in same leaf index
- Probability calibration (how to manipulate the probability distribution, e.g. left and right tail)


## Setup

-  1. Development

```
conda create --name dascripts python=3.10 ipykernel -y
conda activate dascripts
pip install -e ".[dev]"
```

- 2. Build and Publish

```
# Using the publish script (defaults to TestPyPI)
chmod +x publish.sh
./publish.sh

# For production PyPI
./publish.sh --env pypi

# Manual commands
python -m build
python -m twine upload -r testpypi dist/*
python -m twine upload dist/*
```

# Install

```
# From TestPyPI
pip install --index-url https://test.pypi.org/simple/ dascripts

# From PyPI (production)
pip install dascripts
```
