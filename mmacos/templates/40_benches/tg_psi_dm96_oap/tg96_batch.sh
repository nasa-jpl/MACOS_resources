#!/usr/bin/env bash
# Headless, memory-capped launcher for tg96_run (model 1024 ~ 11 GB).
# Usage: ./tg96_batch.sh <tag> [MATLAB-syntax name/value args]
#   ./tg96_batch.sh lens
#   ./tg96_batch.sh oap "'bench.optics','oap'"
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag="${1:?usage: tg96_batch.sh <tag> [args]}"; shift || true
args="$*"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
mkdir -p "$here/runs"
call="tg96_run_batch('tag','$tag'${args:+, $args})"
cd "$here"
log="runs/${tag}.log"
echo "[tg96_batch] $call  (MemoryMax=${TG96_MEMMAX:-14G})" | tee "$log"
if command -v systemd-run >/dev/null 2>&1; then
    systemd-run --user --scope -p MemoryMax="${TG96_MEMMAX:-14G}" \
        matlab -batch "$call" >>"$log" 2>&1
else
    matlab -batch "$call" >>"$log" 2>&1
fi
rc=$?
echo "[tg96_batch] exit $rc" | tee -a "$log"
exit $rc
