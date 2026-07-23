"""Shared fixtures. Per PYTHON_R_INTEROP_PLAN.md's testing strategy
(design decision 7): integration tests launch a real `linal serve`
subprocess rather than mocking the HTTP layer, since the whole point of
this client is matching what the real server actually sends -- fixture
JSON alone (see test_wire.py) can't catch a wire-shape drift.
"""

import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]


def _find_linal_binary() -> Path:
    for profile in ("debug", "release"):
        candidate = REPO_ROOT / "target" / profile / "linal"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"No `linal` binary found under {REPO_ROOT}/target/{{debug,release}}/. "
        "Run `cargo build --bin linal` in the repo root first."
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def linal_server(tmp_path_factory):
    binary = _find_linal_binary()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    # Run with cwd set to a fresh temp dir so the server's `./data`
    # auto-recovery (see docs/ARCHITECTURE.md's Persistence section)
    # never picks up state from a previous test run -- a real issue hit
    # while building this fixture: fixed-name `CREATE DATABASE` calls in
    # the test suite started failing with "already exists" once a
    # `clients/python/data/` directory accumulated from earlier manual
    # runs launched with that directory as cwd.
    server_cwd = tmp_path_factory.mktemp("linal_server")

    proc = subprocess.Popen(
        [str(binary), "serve", "--port", str(port)],
        cwd=server_cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        last_error = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"linal serve exited early (code {proc.returncode}): {stderr}")
            try:
                resp = requests.get(f"{url}/health", timeout=1)
                if resp.status_code == 200:
                    break
            except requests.exceptions.RequestException as e:
                last_error = e
            time.sleep(0.1)
        else:
            raise RuntimeError(f"linal serve never became healthy on {url}: {last_error}")

        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def unique_name(request):
    # `USE <db>` requires a pre-existing database (engine/db.rs:
    # use_database returns "Database not found" rather than
    # auto-creating), so tests share the default database and instead
    # get a unique *dataset* name each, sanitized to a valid DSL
    # identifier (pytest node names can contain `[`, `]`, `-` from
    # parametrization, none of which are valid identifier characters).
    import re

    safe = re.sub(r"[^0-9a-zA-Z_]", "_", request.node.name)
    return f"t_{safe}"
