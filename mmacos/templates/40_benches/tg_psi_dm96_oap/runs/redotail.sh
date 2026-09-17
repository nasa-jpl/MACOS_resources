#!/usr/bin/env bash
# Everything that runs AFTER redoseq.sh, unattended (TO 2026-09-17):
#   redoseq2 (the seed tail on the same bench + the own-cone pupil check)
#   nullab   (where the 59 / 73 nm flat-DM nulls come from)
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
n=0
until grep -qE "chain complete|TIMED OUT|exit [1-9]" runs/redoseq.nohup 2>/dev/null; do
  sleep 60; n=$((n+1))
  if [ $n -gt 300 ]; then echo "[redotail] redoseq never finished; stopping"; exit 1; fi
done
if grep -qE "TIMED OUT|exit [1-9]" runs/redoseq.nohup; then
  echo "[redotail] redoseq did not end clean -- NOT starting the follow-ons"; tail -5 runs/redoseq.nohup; exit 1
fi
echo "[redotail] $(date '+%F %T') redoseq done, starting redoseq2"
./runs/redoseq2.sh >> runs/redoseq2.nohup 2>&1
echo "[redotail] redoseq2 exit $?"
echo "[redotail] $(date '+%F %T') starting nullab"
./runs/nullab.sh >> runs/nullab.nohup 2>&1
echo "[redotail] nullab exit $?"
echo "[redotail] $(date '+%F %T') all done"
