# Import functions from submodules that should be public
from .common import merge
from .plot import decorate, bar, line, hist, scatter
from .processing import DataFrameEncoder


# Define __all__ to restrict what gets imported on 'from dascripts import *'
__all__ = [
    "merge",
    "bar", "line", "hist", "scatter",
    "DataFrameEncoder"
]