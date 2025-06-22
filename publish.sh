#!/bin/bash

# Default environment is testpypi
ENV="testpypi"
DRY_RUN=false

# Help function
show_help() {
    echo "Usage: $0 [--env environment] [-d|--dry-run] [-h|--help]"
    echo
    echo "Publish the package to PyPI or TestPyPI"
    echo
    echo "Options:"
    echo "  --env          Specify target environment: 'pypi' or 'testpypi' (default: 'testpypi')"
    echo "  -d, --dry-run  Build the package without publishing (dry run mode)"
    echo "  -h, --help     Display this help message and exit"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build the package
echo "Building package..."
python -m build

# Publish to the specified environment
if [ "$DRY_RUN" == "true" ]; then
    echo "Dry run mode: Package built but not published."
elif [ "$ENV" == "pypi" ]; then
    echo "Publishing to PyPI (production)..."
    python -m twine upload dist/*
elif [ "$ENV" == "testpypi" ]; then
    echo "Publishing to TestPyPI..."
    python -m twine upload -r testpypi dist/* --verbose
else
    echo "Error: Unknown environment '$ENV'. Use 'testpypi' or 'pypi'."
    echo "Run '$0 --help' for usage information."
    exit 1
fi

echo "Done!"

# Provide installation instructions
if [ "$DRY_RUN" == "true" ]; then
    echo -e "\nDry run completed. Package files are available in dist/"
elif [ "$ENV" == "pypi" ]; then
    echo -e "\nInstallation command:"
    echo "pip install dascripts"
else
    echo -e "\nInstallation command:"
    echo "pip install --index-url https://test.pypi.org/simple/ dascripts"
fi
