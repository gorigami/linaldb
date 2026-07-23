"""Python client for LINALDB.

See clients/CONTRACT.md (in the parent repository) for the wire contract
this client implements, and PYTHON_R_INTEROP_PLAN.md for what's built vs.
still planned (checkpoint 2, `.dataset()`/`/delivery` Parquet export, not
in yet as of this file).
"""

from .client import Client, connect
from .errors import LinalError
from .wire import TableResult, TensorResult

__version__ = "0.1.0"

__all__ = [
    "connect",
    "Client",
    "LinalError",
    "TableResult",
    "TensorResult",
]
