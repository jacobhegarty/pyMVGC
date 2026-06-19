"""pyMVGC: multivariate Granger causality tools for time-series data.

This module performs eager imports of the public functions and their numeric
dependencies so that a plain ``import pyMVGC`` behaves like a conventional
package and immediately exposes the callables.
"""

# Import required numeric dependencies synchronously so failures surface at
# import-time.
import numpy as np  # noqa: F401

from .ts_to_var import ts_to_var
from .var_to_autocov import var_to_autocov
from .autocov_to_var import autocov_to_var
from .autocov_to_MVGC import autocov_to_MVGC
from .autocov_to_pMVGC import autocov_to_pMVGC
from .ts_to_MVGC import ts_to_MVGC
from .ts_to_pMVGC import ts_to_pMVGC
from .estimate_model_order import estimate_model_order

__all__ = [
    "ts_to_var",
    "var_to_autocov",
    "autocov_to_var",
    "autocov_to_MVGC",
    "autocov_to_pMVGC",
    "ts_to_MVGC",
    "ts_to_pMVGC",
    "estimate_model_order"
]
