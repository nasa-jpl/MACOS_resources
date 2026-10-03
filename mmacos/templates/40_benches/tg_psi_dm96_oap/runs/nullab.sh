#!/usr/bin/env bash
# Where the redo bench's 59 nm flat-DM null comes from (TO 2026-09-17).
#
# The record's lens rig nulls at 0.134 nm tuned / 9.1 nm on the geometric seed.
# The redo's seed-station tail on the collimated bench with the substrates in
# reports 59.3168 nm, unmoved to four decimals across the whole tail search.
# 59 nm rms is a fixed pattern the reference frame removes, but it is also
# within reach of the four-step's own lambda/4 = 158 nm, so it is worth knowing
# which change put it there before package C spends a day on rows.
#
# Each run's value is its SEED line -- "TAIL SEED: cost ... (null X nm ...)" --
# at the tuner's own reduced resolution (model 512, 193 rays).  The 1-D tune
# after it is just the cheapest way to make the tuner print that line; its
# winner is not used.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
REC="'bench.SRC_AT_FOCUS',false,'bench.SRC_TRIM',0,'bench.L1_Kr',236.866,'bench.L1_Kc',-0.5829,'bench.L2_Kc',-0.5826,'bench.MASK_TRIM',0"
NOSUB="'bench.PLATE_SUB',[],'bench.MASK_SUB',[],'bench.EDGE_MARGIN',2.0,'bench.BS_T',1.5"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"

wait_matlab () {
  local n=0
  while pgrep -x MATLAB >/dev/null 2>&1; do
    sleep 30; n=$((n+1))
    if [ $n -gt 360 ]; then echo "[nullab] TIMED OUT"; exit 1; fi
  done
}
run () {   # run <tag> <extra args>
  wait_matlab
  echo "[nullab] $(date '+%F %T') $1"
  systemd-run --user --scope -p MemoryMax=10G matlab -batch \
    "tg96_tail_batch('tag','$1','tail.objective','null','tail.free',{'DET_TRIM'}${2:+,$2})" \
    > "runs/nullab_$1.log" 2>&1
  echo "[nullab] $1 exit $?"
  grep -E "TAIL SEED|TAIL WINNER" "runs/nullab_$1.log" || echo "[nullab] NO SEED LINE for $1"
}

run nullab_new    ""                 # the redo bench: collimated, substrates in
run nullab_nosub  "$NOSUB"           # collimated, the record's ideal elements
run nullab_rec    "$REC"             # the record's optics and conjugate, substrates in
run nullab_recnos "$REC,$NOSUB"      # the record's bench outright (expect ~9 nm: its seed)
# The mirror rig is the sharper version of the same question: package A did not
# touch its optics (a parabola fed at its focus was already exact), yet its seed
# null went from the 0.0223 nm of the reflective record to 73.3227 nm on the
# redo bench.  Only the substrates, the seat and the beam changed, so one leg
# with the substrates off attributes it outright.  The "on" number is already in
# runs/tail_oap96.log's SEED line and is not re-run here.
run nullab_oapnos "$OAP,$NOSUB"      # mirror rig, the record's ideal elements
echo "[nullab] $(date '+%F %T') done"
