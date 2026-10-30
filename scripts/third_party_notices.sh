#!/usr/bin/env bash
# Writes the third-party license notices for one build: every Rust crate
# linked into it, each license text once (`cargo about`, config in
# `about.toml`, template `about.hbs`). Fails if a dependency's license is not
# in `about.toml`'s accepted list.
#
#   scripts/third_party_notices.sh <Cargo.toml> <output file>
#
# Release binaries use the root manifest; the `linaldb` wheel uses
# `clients/python-embedded/Cargo.toml` (it also links pyo3 and its deps).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
manifest="${1:?usage: $0 <Cargo.toml> <output file>}"
out="${2:?usage: $0 <Cargo.toml> <output file>}"
cargo about generate --locked \
  --config "$root/about.toml" \
  --manifest-path "$manifest" \
  --output-file "$out" \
  "$root/about.hbs"
echo "third-party notices: $out"
