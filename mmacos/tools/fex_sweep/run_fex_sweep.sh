#!/bin/bash
# FEX compatibility sweep: run the engine exit-pupil finder over a
# legacy Rx corpus and log both radius legs per prescription:
#   ***** FEX: zp_iEm1 = <legacy iEm1->EP>  zp = <EP->next (default)>
# Purpose: quantify the 2026-07-03 FEX EP-radius rework (radius =
# chief-ray distance EP -> iElt+1 plane, the far-field propagation
# distance; legacy leg = fallback) across real prescriptions, and
# exercise the new guards (telecentric, beam-footprint autoswitch,
# Return-order flag).
#
# Corpus: every .in under the directories listed in CORPUS_DIRS.
# An Rx is swept when it has >= 4 "Element=" lines (multi-line format)
# and its next-to-last element is Return or Reference (a FEX-able EP).
# Rx without ApStop= get a heuristic stop at the first reflective
# element.  One MATLAB process per Rx (crash isolation + clean logs).
#
# Usage:  ./run_fex_sweep.sh [outdir]     (default ./sweep_logs)
# Summary: grep -h "zp_iEm1\|WARNING\|AUTOSWITCH" <outdir>/*.log

CORPUS_DIRS=(
  "$HOME/dev/MACOS_sandbox/old_Rx"
  "$HOME/dev/macos/docs/macos-manual/examples"
)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/sweep_logs}"
mkdir -p "$OUT"

n=0
for d in "${CORPUS_DIRS[@]}"; do
  for f in "$d"/*.in; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .in)
    nblk=$(grep -c "Element=" "$f")
    if [ "$nblk" -lt 4 ]; then
      echo "SKIP(one-line/odd)   $base"; continue
    fi
    penult=$(grep "Element=" "$f" | tail -2 | head -1 \
             | tr -d ' \r\t' | sed 's/Element=//')
    case "$penult" in
      Return|Reference) ;;
      *) echo "SKIP(penult=$penult)  $base"; continue ;;
    esac
    hasstop=$(grep -c "ApStop" "$f")
    stopelt=$(grep "Element=" "$f" | tr -d ' \r\t' | sed 's/Element=//' \
              | grep -n -m1 -E "Reflector|Segment|NSReflector|Refractor" \
              | cut -d: -f1)
    [ -z "$stopelt" ] && stopelt=1
    n=$((n+1))
    FEXRX="$f" FEXHASSTOP="$hasstop" FEXSTOPELT="$stopelt" \
      timeout 120 matlab -sd "$HERE" -batch fex_one \
      > "$OUT/${n}_${base}.log" 2>&1
    rc=$?
    legs=$(grep -m1 "zp_iEm1" "$OUT/${n}_${base}.log" | tr -s ' ')
    echo "$n $base exit=$rc ${legs:-<no fex output>}"
  done
done
echo "sweep done: $n Rx, logs in $OUT"
