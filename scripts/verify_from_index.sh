#!/usr/bin/env bash
# Installs `linaldb==<version>` from a package index the way a user would and
# checks it is the wheel the release candidate built: its sha256 must be in
# the candidate's SHA256SUMS file. Then runs scripts/verify_wheel.py on it.
#
#   scripts/verify_from_index.sh <testpypi|pypi> <version> <SHA256SUMS file> [target dir]
#
# With a target dir, packages are installed there (`pip install --target`) and
# the check is left to the caller -- the Kaggle-like job runs it afterwards
# in a container without network.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
index="${1:?usage: $0 <testpypi|pypi> <version> <SHA256SUMS> [target dir]}"
version="${2:?version}"
sums="$(cd "$(dirname "${3:?SHA256SUMS file}")" && pwd)/$(basename "$3")"
target="${4:-}"

case "$index" in
  testpypi) url="https://test.pypi.org/simple/" ;;
  pypi) url="https://pypi.org/simple/" ;;
  *) echo "unknown index $index" >&2; exit 2 ;;
esac

python=python3
command -v "$python" >/dev/null 2>&1 || python=python
work="$(mktemp -d)"

# A freshly uploaded file can take a few minutes to show up in the index.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if "$python" -m pip download --no-deps --no-cache-dir --index-url "$url" \
      --dest "$work/dl" "linaldb==$version"; then
    break
  fi
  [ "$attempt" -eq 10 ] && { echo "linaldb==$version not on $index" >&2; exit 1; }
  sleep 30
done

wheel=$(ls "$work"/dl/linaldb-*.whl)
"$python" - "$wheel" "$sums" <<'PY'
import hashlib, os, sys
wheel, sums = sys.argv[1], sys.argv[2]
digest = hashlib.sha256(open(wheel, "rb").read()).hexdigest()
expected = {}
for line in open(sums):
    h, name = line.split()
    expected[name.lstrip("*")] = h
name = os.path.basename(wheel)
if expected.get(name) != digest:
    sys.exit(f"{name} from the index has sha256 {digest}; the candidate's is {expected.get(name)}. "
             "The index holds a different build (an earlier candidate of this version?): bump the version.")
print(f"{name}: sha256 matches the candidate")
PY

# Dependencies come from PyPI; only linaldb itself from the index under test.
if [ -n "$target" ]; then
  "$python" -m pip install --no-cache-dir --target "$target" pyarrow
  "$python" -m pip install --no-cache-dir --no-deps --target "$target" "$wheel"
else
  "$python" -m pip install --no-cache-dir pyarrow
  "$python" -m pip install --no-cache-dir --no-deps --force-reinstall "$wheel"
  cd "$work" && "$python" "$here/verify_wheel.py" --version "$version"
fi
