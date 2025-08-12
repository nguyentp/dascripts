# Import functions from submodules that should be public
from .plot import decorate, hist, scatter
from .data_processing import merge, DFEncoder

print("""
Hello from dascripts! You have imported the following functions:
- Plot: decorate, hist, scatter
- Data processing: merge, DFEncoder
Have fun with your data analysis!
""")

# Define __all__ to restrict what gets imported on 'from dascripts import *'
__all__ = [
    "decorate",
    "merge", "DFEncoder",
]