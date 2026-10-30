#!/usr/bin/env bash
# Fails if the license metadata drifts apart: every place that declares the
# project's license must say Apache-2.0, the packaged LICENSE/NOTICE copies
# must match the root ones, and the old Community License may only be named
# where the README/NOTICE/CHANGELOGs record which versions it still covers.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
err() { echo "license check: $*" >&2; fail=1; }

# Root LICENSE is the unmodified Apache-2.0 text (so license scanners detect it).
apache_sha="cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
actual_sha=$( (sha256sum LICENSE 2>/dev/null || shasum -a 256 LICENSE) | cut -d' ' -f1)
[ "$actual_sha" = "$apache_sha" ] || err "LICENSE is not the verbatim Apache-2.0 text"
[ -f NOTICE ] || err "NOTICE is missing"

# linaldb-server ships committed copies (PEP 639 license-files can't point outside
# the package). linaldb (python-embedded) gets them copied in by CI at build time
# instead, because its maturin sdist is rooted at the repo and already packs the
# root files; a committed copy would collide with them.
for f in LICENSE NOTICE; do
  cmp -s "$f" "clients/python/$f" || err "clients/python/$f differs from the root $f"
done
grep -q 'cp LICENSE NOTICE clients/python-embedded/' .github/workflows/pypi-publish-embedded.yml \
  || err "pypi-publish-embedded.yml no longer copies LICENSE/NOTICE into the package"
for pkg in clients/python-embedded clients/python; do
  grep -q '^license = "Apache-2.0"$' "$pkg/pyproject.toml" \
    || err "$pkg/pyproject.toml: expected license = \"Apache-2.0\""
  grep -q '^license-files = \["LICENSE", "NOTICE"\]$' "$pkg/pyproject.toml" \
    || err "$pkg/pyproject.toml: expected license-files = [\"LICENSE\", \"NOTICE\"]"
  if grep -q 'License ::' "$pkg/pyproject.toml"; then
    err "$pkg/pyproject.toml: drop License :: classifiers (PEP 639 uses the license expression)"
  fi
done

for manifest in Cargo.toml clients/python-embedded/Cargo.toml clients/r-embedded/src/rust/Cargo.toml; do
  grep -Eq "^license = [\"']Apache-2.0[\"']$" "$manifest" \
    || err "$manifest: expected license = \"Apache-2.0\""
done

for desc in clients/r/DESCRIPTION clients/r-embedded/DESCRIPTION; do
  grep -q '^License: Apache License (== 2.0)$' "$desc" \
    || err "$desc: expected License: Apache License (== 2.0)"
done

stale=$(git grep -n -i "community license" -- . \
  ':!README.md' ':!NOTICE' ':!clients/python/NOTICE' \
  ':!*CHANGELOG.md' ':!IMPROVEMENTS_PLAN.md' ':!docs/archive' ':!scripts/check_license.sh' || true)
[ -z "$stale" ] || err "Community License still named outside the history notes:
$stale"

[ "$fail" -eq 0 ] && echo "license check: OK (Apache-2.0 everywhere)"
exit "$fail"
