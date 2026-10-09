#!/usr/bin/env bash
# RUN_WALL_FLEET  The walled, converged tilt-vs-price frontier, one MATLAB
# process per point, run in parallel and assembled afterwards by AFOCAL4_WALL.
#
# WHY A FLEET.  A converged point here is 300-900 function evaluations at
# ~12 s each (the union wall adds 51% to an 8.2 s evaluation), i.e. one to
# three hours.  The frontier is 17 of them.  Run serially it would not land
# inside any useful window; run as one process per point on a 16-core box it
# lands overnight.  Each point checkpoints itself into wall_<tag>.mat, so a
# process that dies costs one point and not the run -- the same reason
# AFOCAL4_BASIN2 carries a 'tag'.
#
# ONE MODEL SIZE PER PROCESS is a standing MACOS rule and is why this is a
# shell fleet rather than a parfor.
#
#   ./run_wall_fleet.sh [JOBS]        default 6 concurrent
#
# Environment:
#   WALL_EVALS   evaluations per restart round (300)
#   WALL_ROUNDS  restart rounds (3)
#   MACOS_HOME   defaults to ~/dev/macos/macos_f90

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${1:-6}"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
export WALL_EVALS="${WALL_EVALS:-300}"
export WALL_ROUNDS="${WALL_ROUNDS:-3}"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

# tilt:union_min_mm:wall_on
JOBLIST=""
for t in -6 -7 -8 -9 -10 -11 -12; do
  JOBLIST+="$t:0:1"$'\n'
  JOBLIST+="$t:15:1"$'\n'
done
# the wall-OFF arm: the non-vacuity A/B (-8 and -9 are where the clearing
# stage measured the solver spending its margin) plus the central-difference
# polish of the delivered -10 deg deck.
for t in -8 -9 -10; do
  JOBLIST+="$t:0:0"$'\n'
done

run_one() {
  IFS=: read -r t u w <<< "$1"
  if [ "$w" = "1" ]; then tag="t$(printf '%+03.0f' "$((t*10))")_u$(printf '%02d' "$u")";
  else tag="t$(printf '%+03.0f' "$((t*10))")_u$(printf '%02d' "$u")_nowall"; fi
  log="$LOGDIR/wall_$tag.log"
  echo "[$(date +%H:%M:%S)] start $tag -> $log"
  WALL_TILT="$t" WALL_UMIN="$u" WALL_ON="$w" WALL_TAG="$tag" \
    matlab -batch "run('$HERE/run_wall_point.m')" > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $tag (exit $?)"
}
export -f run_one
export HERE LOGDIR

echo "$JOBLIST" | grep -v '^$' | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] fleet complete; assemble with afocal4_wall()"
