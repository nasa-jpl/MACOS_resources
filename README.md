# MACOS_API
MACOS is "Modeling and Analysis for Controlled Optical Systems"

MACOS was created at the Jet Propulsion Laboratory

This repository contains:

Generic MACOS Interface (GMI): uses MATLAB as a front end

pymacos: uses python as a front end

---

## Building / Compiling

These interfaces build against the MACOS/SMACOS engine in the **sibling
`macos` repository**. Clone both side by side under one parent directory, on
the **same branch name**, and build the engine first.

**→ Full, step-by-step compile instructions live in the `macos` repo:**

- **`macos/HOW_TO_COMPILE.md`** — the copy-paste walkthrough (start here)
- **`macos/README.md`** — full option matrix (Linux, Windows, pymacos,
  CMake-direct, troubleshooting)

Quick version (Linux), from a clean slate:

```bash
mkdir -p ~/dev && cd ~/dev
git clone git@github.com:nasa-jpl/macos.git
git clone git@github.com:nasa-jpl/MACOS_resources.git
git -C macos checkout opt-dev && git -C MACOS_resources checkout opt-dev
cd ~/dev/macos && source ./makeall.sh        # builds the engine + GMI mex
```

> **Both repositories must be on the same branch name** (`opt-dev` with
> `opt-dev`, `sls-dev` with `sls-dev`). The bindings compile against the
> engine's Fortran module files; a mismatched pair makes the API look
> "missing" at link time.
