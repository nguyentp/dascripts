#!/bin/bash

# Default environment is testpypi
ENV="testpypi"

# Check if an environment argument was provided
if [ $# -ge 1 ]; then
    ENV="$1"
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build the package
echo "Building package..."
python -m build

# Publish to the specified environment
if [ "$ENV" == "pypi" ]; then
    echo "Publishing to PyPI (production)..."
    python -m twine upload dist/*
elif [ "$ENV" == "testpypi" ]; then
    echo "Publishing to TestPyPI..."
    python -m twine upload -r testpypi dist/* --verbose
else
    echo "Error: Unknown environment '$ENV'. Use 'testpypi' or 'pypi'."
    exit 1
fi

echo "Done!"

# Provide installation instructions
if [ "$ENV" == "pypi" ]; then
    echo -e "\nInstallation command:"
    echo "pip install dascripts"
else
    echo -e "\nInstallation command:"
    echo "pip install --index-url https://test.pypi.org/simple/ dascripts"
fi
