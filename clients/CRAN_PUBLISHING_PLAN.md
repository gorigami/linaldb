# CRAN Publishing Plan

## Purpose

We already validated the full PyPI publishing pipeline for the Python
clients (`linaldb` embedded + `linaldb-server` HTTP, both live on PyPI as of
2026-09-09 — see git history for the rename/CI work). This document plans
the equivalent effort for the two R packages, `clients/r` (`linaldb.server`)
and `clients/r-embedded` (`linaldb`), targeting CRAN.

**Status: planning only.** Nothing in this document has been implemented
yet — no code changes, no `.Rbuildignore`/`DESCRIPTION` edits, nothing
committed. This is meant to land as its own PR (or the first of a short
stack) the next time we pick this up.

**Audit method**: every finding below came from actually running
`R CMD build` / `R CMD check --as-cran` against both packages, and — for
the build-isolation question — extracting a real generated tarball into
`/tmp`, far from the git checkout, and building it from there. Nothing
here is inferred from reading source alone.

## Current state, in one paragraph each

**`clients/r` (linaldb.server, HTTP client)** — pure R, no compiled code.
Full `R CMD check --as-cran` takes **17 seconds**. Not a speed problem at
all. Three things stand between this and a real CRAN submission: the
license, two minor NOTEs, and a leftover file from test teardown. All
individually small.

**`clients/r-embedded` (linaldb, native binding)** — wraps the Rust engine
directly via extendr. `R CMD check --as-cran` **fails with a hard ERROR**,
not a NOTE: the crate `linalr` under `src/rust/` depends on the root
engine crate via `linal = { path = "../../../.." }`, a relative path
pointing *outside* the R package's own directory. `R CMD build` only
copies `clients/r-embedded/` in isolation, so once that happens, the path
resolves to nothing and the Rust compilation fails immediately. This is
not a CRAN-specific problem — the tarball this produces cannot be built by
anyone, on any machine, outside this exact monorepo checkout. It's a
packaging-completeness bug, and it has to be fixed before speed or license
or anything else is even worth discussing for this package.

(Checked as a side question: this does **not** affect what's already on
PyPI. Python's `maturin` understands Cargo path dependencies and bundles
the root `linal` crate into the sdist automatically, preserving the same
relative directory structure — verified by extracting a real generated
sdist into `/tmp` and building it there with zero access to the original
checkout. R's plain `R CMD build` has no equivalent awareness. Different
tools, different outcome, no action needed on the Python side.)

---

## 1. License

**Resolved (2026-10-29).** The project is relicensed under Apache-2.0 (see
`IMPROVEMENTS_PLAN.md`, "License decision"), so both `DESCRIPTION` files now
say `License: Apache License (== 2.0)`, a standard CRAN license string
(checked with `tools:::analyze_license`: canonical and verified). No
package-local `LICENSE` file is needed for a standard license, and the old
risk of reviewers declining a commercial-use restriction is gone.

## 2. Documentation

**Your framing was right**: `.Rbuildignore` already exists for both
packages, so anything CRAN shouldn't see (or doesn't need to) can be
excluded there rather than deleted — but a few things found in the audit
go beyond "just add to .Rbuildignore":

- **Non-standard top-level files** (NOTE on `clients/r`, would also apply
  to `clients/r-embedded` once it builds): `CHANGELOG.md` and `examples/`
  aren't files CRAN's `R CMD check` expects at the package root. Two
  options: add both to `.Rbuildignore` (keeps them out of the CRAN
  tarball, still visible on GitHub), or rename `CHANGELOG.md` → `NEWS.md`
  (the convention R/CRAN actually recognizes — shows up in
  `news(package = "...")` and gets its own tab on the package's CRAN
  page). I'd lean toward the rename for the CHANGELOG specifically, since
  that's real user-facing value CRAN gives packages that follow the
  convention; `.Rbuildignore` for `examples/`.
- **Stale doc comment**: `clients/r/R/linaldb.server-package.R` still says
  *"Scaffolding only as of this file -- no functions defined yet"* — false
  now, the package has nine exported functions. Pre-existing, unrelated to
  CRAN specifically, but should get fixed as part of writing real
  CRAN-facing documentation rather than carried forward.
- **Dangling reference**: both packages' `DESCRIPTION`/roxygen docs point
  readers at `PYTHON_R_INTEROP_PLAN.md`, which doesn't exist anywhere in
  the repo (confirmed via full-tree search in an earlier audit this
  session — pre-existing, not something this effort broke). A real CRAN
  submission's `Description:` field shouldn't reference a file that
  doesn't exist; needs either a real replacement pointer (e.g.
  `clients/CONTRACT.md`/`clients/EMBEDDED_CONTRACT.md`, which do exist) or
  the reference removed.
- **Missing `DESCRIPTION` fields**: neither package has `URL:` or
  `BugReports:` — not CRAN-mandatory, but expected in modern submissions
  and trivial to add (repo URL + `.../issues`).
- **Write real package-level documentation.** Right now both packages'
  `_PACKAGE`-doc files are fairly thin (a couple of sentences). Worth
  expanding into something that reads like real CRAN package
  documentation — a short worked example in the package doc itself is
  common practice and roxygen supports it directly (`@examples` blocks
  that show up in `?linaldb`/`?linaldb.server`).

**What I'd adjust**: the `.Rbuildignore` additions, the `CHANGELOG.md` →
`NEWS.md` rename (+ pointer updates in READMEs), the stale scaffolding
line, the dangling plan-doc references, the missing `URL`/`BugReports`
fields, and a real `@examples`-backed package doc for both.

## 3. Build (the real blocker)

You had this right: it's about how the Rust build is wired, not about
starting over.

**The fix**: make `clients/r-embedded/src/rust/` fully self-contained so
`R CMD build`'s isolated copy has everything it needs, without reaching
outside its own directory. Concretely, this means using `cargo vendor` to
snapshot every dependency — including the path-dependency on the root
`linal` crate — into a `vendor/` directory (or `vendor.tar.xz`) bundled
inside the R package. This isn't a new idea we'd be inventing: the
package's own `Makevars`/`Makevars.in` **already** has the offline-vendor
logic wired up (`if [ -d ./vendor ]; ... elif [ -f ./rust/vendor.tar.xz ];
...`) — it's just never had a `vendor/` directory to actually use. This
also solves a second CRAN concern for free: CRAN's build environment
shouldn't need live internet access to crates.io during compilation, and
a vendored build needs none.

**Scoped to not touch Python's setup** (your explicit ask): this only
touches files under `clients/r-embedded/` — the vendor directory, its
`Cargo.lock`, and `.cargo/config.toml` pointing at it. It does **not**
touch the root `Cargo.toml`/`Cargo.lock`, and does not touch
`clients/python-embedded/` at all. The two embedded bindings already
build via fully independent toolchain invocations (separate `Cargo.lock`
files, `clients/python-embedded/.cargo/config.toml` pointing at the
repo-root `target/` purely as a build-cache optimization, not a shared
dependency graph) — vendoring one doesn't touch the other's build.

**What this doesn't fully solve — re-measure, don't assume**: once
self-contained, the Rust compile itself still takes real time (~1m52s
measured locally with a warm cargo cache; a genuinely cold CRAN check
machine — network-free thanks to vendoring, but still compiling ~150
crates plus HDF5 from source — could plausibly run longer). The package's
`Makevars` deletes `src/rust/target/` after every successful install
(correct CRAN-safe practice — CRAN doesn't want build artifacts left
behind), which means **this cost is paid on every single check**, with no
way to cache around it. This is squarely the kind of thing that gets
commented on in CRAN review, though it isn't automatically disqualifying
— there's real precedent (`gifski`, several `extendr`/`savvy`-based
packages) for CRAN accepting Rust-backed packages with non-trivial build
times. The honest plan is: fix the vendoring first (it's a hard blocker
either way), re-measure the real end-to-end time with `vendor/` in place,
*then* decide whether it's within CRAN's tolerance or whether
`clients/r-embedded` is better distributed via GitHub only
(`remotes::install_github()`, no CRAN review process, no build-time
scrutiny) while `clients/r` (the lightweight HTTP client, already at 17
seconds) goes to CRAN on its own.

**Also found, filed under "build" since it surfaced during the same
audit**: `clients/r-embedded/DESCRIPTION`'s `SystemRequirements:` lists
`Cargo, rustc >= 1.65.0, xz` — but the vendored `linal` engine dependency
(`hdf5-metno`'s `static` feature) also needs `cmake` and a C toolchain to
build HDF5 from source, exactly like the root crate and
`clients/python-embedded` do. This field is missing `cmake`, which could
cause a confusing failure on a check machine that doesn't have it
preinstalled rather than a clear "missing system requirement" message.

## Other things the audit surfaced (not asked for, flagging anyway)

- **`clients/r`'s test teardown leaves a stray file.** `R CMD check`
  flagged `linal_server_44381.pid` left behind in the check directory —
  the integration tests spawn a real `linal serve` subprocess and
  something in that fixture's cleanup isn't removing its PID file. CRAN
  explicitly checks for "detritus in the temp directory" and will flag
  this; needs the test fixture (likely `tests/testthat/helper-server.R`)
  to clean it up on teardown.
- **Rust crate naming inconsistency** in `clients/r-embedded/src/rust/Cargo.toml`:
  `[package] name = 'linalr'` doesn't match anything else in the project
  (not the R package name `linaldb`, not the `[lib] name = 'linaldb'` in
  the same file). Harmless — Cargo package name and lib name are
  independent, and this predates our rename work — but worth a rename to
  `linaldb` (or similar) for anyone reading the crate list later, purely
  cosmetic, no functional impact either way.
- **A `cran-comments.md` will be needed at actual submission time** — the
  standard file CRAN submissions include explaining the `R CMD check`
  environment used, which NOTEs remain and why they're expected/
  acceptable (e.g. "New submission"). Not something to write now — a last step right
  before hitting submit, once everything else here is settled.

## Suggested sequencing

1. ~~**License decision**~~ — done: Apache-2.0, `DESCRIPTION` files updated.
2. **`clients/r` cleanup PR** — `.Rbuildignore` /
   `NEWS.md` rename, stale doc line, dangling plan-doc reference,
   `URL`/`BugReports`, the `.pid` test-cleanup fix. This package could
   realistically go to CRAN on its own once this lands — it's already
   fast and close.
3. **`clients/r-embedded` vendoring PR** — `cargo vendor`, wire up
   `Makevars`'s existing offline-build path, add `cmake` to
   `SystemRequirements`, same documentation cleanup as `clients/r`.
   Re-measure `R CMD check --as-cran` timing once this is real.
4. **Decide `clients/r-embedded`'s actual destination** based on step 3's
   real numbers — CRAN, or GitHub-only via `remotes::install_github()`.
5. **`cran-comments.md` + actual submission**, whenever we're ready to
   pull the trigger for real.

## Verification plan (once implemented)

- `R CMD check --as-cran --no-manual` on both packages, from a **freshly
  cloned checkout** (not this working copy) to catch anything path- or
  cache-dependent that a local run might hide.
- Repeat the isolation test we already did for Python: extract the built
  `clients/r-embedded` tarball into `/tmp`, build it there with zero
  access to this checkout, confirm it installs and a real DSL round-trip
  (`DATASET`/`INSERT`/`SELECT`) works — the same test that already proved
  Python's sdist is fine.
- Confirm `0 errors, 0 warnings` (or an explicitly justified, documented
  NOTE list) on both packages before considering either submission-ready.
