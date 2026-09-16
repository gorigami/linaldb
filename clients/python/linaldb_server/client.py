from __future__ import annotations

import re
import warnings

import requests

from .errors import LinalError
from .wire import TableResult, unwrap_result

# Server enforces a 30s query timeout (QUERY_TIMEOUT_SECS in
# src/server/mod.rs) and returns a clean status:error response when hit —
# our own request timeout is set a little above that so we see the
# server's real timeout message instead of cutting the connection first.
_DEFAULT_TIMEOUT_SECS = 35

# Matches a bare `USE <database>` statement (DSL keyword, src/dsl/lexer.rs
# -- case-sensitive, no `ignore(case)` modifier, so a lowercase `use ...`
# isn't a valid statement in the real grammar at all), which mutates the
# server's shared, process-wide active database when sent without an
# X-Linal-Database header (see clients/CONTRACT.md). Deliberately excludes
# `USE DATASET FROM ...` (an unrelated statement -- loads an external file,
# never touches the active database); the exclusion is case-sensitive too,
# so `USE dataset` (a lowercase *identifier* literally named "dataset") is
# still correctly flagged as a real database switch.
#
# This is a heuristic, not a DSL parser -- it doesn't know about string
# literals or comments, so a false negative is possible in unusual
# formatting. Accepted gap rather than a false positive risk.
_BARE_USE_DATABASE_RE = re.compile(r"^\s*USE\s+(?!DATASET\b)[A-Za-z_]\w*\b")


def connect(url: str, database: str | None = None) -> "Client":
    """Connect to a running `linal serve` instance at `url`
    (e.g. "http://localhost:8080")."""
    return Client(url, database=database)


class Client:
    def __init__(self, url: str, database: str | None = None):
        self.url = url.rstrip("/")
        self.database = database
        self._session = requests.Session()

    def execute(self, dsl: str):
        """Run one DSL command and return its unwrapped result: `None`
        (no output), a `str` (`Message`), a `wire.TableResult`, or a
        `wire.TensorResult`. Raises `LinalError` on `status: "error"`.
        """
        if self.database is None and _BARE_USE_DATABASE_RE.match(dsl):
            warnings.warn(
                "This bare `USE <database>` statement will switch the "
                "server's shared, process-wide active database, because "
                "this Client has no database= set (see clients/CONTRACT.md) "
                "-- concurrent Clients without an explicit database= will "
                "race on this. Pass database=... to connect()/Client() for "
                "an isolated session if that's not intended.",
                UserWarning,
                stacklevel=2,
            )

        headers = {"Content-Type": "text/plain"}
        if self.database:
            headers["X-Linal-Database"] = self.database

        resp = self._session.post(
            f"{self.url}/execute",
            params={"format": "json"},
            data=dsl.encode("utf-8"),
            headers=headers,
            timeout=_DEFAULT_TIMEOUT_SECS,
        )

        try:
            body = resp.json()
        except ValueError:
            # Contract §4: only fall back to a raw HTTP error if the body
            # isn't even the standard {"status": ...} shape.
            resp.raise_for_status()
            raise LinalError(
                f"Non-JSON response from server (HTTP {resp.status_code}): {resp.text}"
            )

        if body.get("status") != "ok":
            raise LinalError(
                body.get("error") or f"Unknown server error (HTTP {resp.status_code})"
            )

        return unwrap_result(body.get("result"))

    def query(self, dsl: str):
        """Run a DSL command expected to return a table and return a
        `pandas.DataFrame`. Requires the `pandas` extra
        (`pip install "linaldb-server[pandas]"`).
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Client.query() requires the `pandas` extra: "
                'pip install "linaldb-server[pandas]"'
            ) from e

        result = self.execute(dsl)
        if not isinstance(result, TableResult):
            raise LinalError(
                f"query() expects a table-shaped result, got {type(result).__name__} "
                "(use execute() for non-table results)"
            )
        return pd.DataFrame(result.rows, columns=result.columns)

    def dataset(self, name: str) -> "Dataset":
        """A handle to a saved dataset's `/delivery/*` export (contract
        §2) — `.schema()`/`.stats()`/`.manifest()`/`.to_arrow()`/
        `.to_pandas()`.
        """
        from .dataset import Dataset

        return Dataset(self, name)
