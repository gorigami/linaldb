#!/usr/bin/env bash
# Version consistency for the two release lines: the engine (`linal`, tags
# `vX`) and the embedded Python package (`linaldb`, tags `linaldb-vX`).
#
#   scripts/check_versions.sh                  every PR: the manifests agree
#   scripts/check_versions.sh --release        also: both CHANGELOGs have a
#                                              dated entry for their version
#   scripts/check_versions.sh --tag <tag>      also: the tag names that version
#
# Prints `engine=<v>` and `linaldb=<v>` (append to $GITHUB_OUTPUT in CI).
set -euo pipefail
cd "$(dirname "$0")/.."

release=0
tag=""
while [ $# -gt 0 ]; do
  case "$1" in
    --release) release=1 ;;
    --tag) tag="${2:?--tag needs a value}"; shift ;;
    *) echo "usage: $0 [--release] [--tag <tag>]" >&2; exit 2 ;;
  esac
  shift
done

fail=0
err() { echo "version check: $*" >&2; fail=1; }

# First `version = "..."` of a manifest's [package] section.
manifest_version() { sed -n 's/^version = "\(.*\)"$/\1/p' "$1" | head -1; }
# Version a Cargo.lock records for package $2.
lock_version() { awk -v n="$2" '$0 == "name = \"" n "\"" {getline; gsub(/version = |"/, ""); print; exit}' "$1"; }

engine=$(manifest_version Cargo.toml)
linaldb=$(manifest_version clients/python-embedded/Cargo.toml)
pyproject=$(manifest_version clients/python-embedded/pyproject.toml)

[ -n "$engine" ] || err "no version in Cargo.toml"
[ "$(lock_version Cargo.lock linal)" = "$engine" ] \
  || err "Cargo.lock has linal $(lock_version Cargo.lock linal), Cargo.toml says $engine (run cargo build)"
[ "$pyproject" = "$linaldb" ] \
  || err "clients/python-embedded: pyproject.toml says $pyproject, Cargo.toml says $linaldb"
[ "$(lock_version clients/python-embedded/Cargo.lock linal-python)" = "$linaldb" ] \
  || err "clients/python-embedded/Cargo.lock has linal-python $(lock_version clients/python-embedded/Cargo.lock linal-python), Cargo.toml says $linaldb"
# The wheel links the engine through a path dependency; its lock records which
# engine version it was built against, so a release can say which engine a
# linaldb version contains.
[ "$(lock_version clients/python-embedded/Cargo.lock linal)" = "$engine" ] \
  || err "clients/python-embedded/Cargo.lock pins engine $(lock_version clients/python-embedded/Cargo.lock linal), the engine is $engine (run cargo update -p linal there)"

if [ "$release" -eq 1 ]; then
  grep -Eq "^## \[$engine\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$" CHANGELOG.md \
    || err "CHANGELOG.md has no '## [$engine] - YYYY-MM-DD' entry"
  grep -Eq "^## \[$linaldb\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$" clients/python-embedded/CHANGELOG.md \
    || err "clients/python-embedded/CHANGELOG.md has no '## [$linaldb] - YYYY-MM-DD' entry"
fi

case "$tag" in
  "") ;;
  linaldb-v*) [ "${tag#linaldb-v}" = "$linaldb" ] || err "tag $tag, but linaldb is $linaldb" ;;
  v*) [ "${tag#v}" = "$engine" ] || err "tag $tag, but the engine is $engine" ;;
  *) err "unknown tag $tag (expected vX or linaldb-vX)" ;;
esac

echo "engine=$engine"
echo "linaldb=$linaldb"
[ "$fail" -eq 0 ] || exit 1
echo "version check: OK" >&2
