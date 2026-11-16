#!/usr/bin/env bash
# clean_results.sh -- age-based cleanup of coro experiment workspaces.
#
# The coro E1/E2/... batch runs save full MATLAB workspaces (1024x1024
# intensity arrays + DM states, tens of MB each) under results/ so
# analysis can resume after an eval/dialog cycle without re-tracing.
# save_coro_workspace.m already keeps only the last 2 per experiment
# tag (deterministic guard); THIS script is the scheduled age-based
# catch-all so nothing lingers and fills the disk.
#
# Usage:   ./clean_results.sh                 # delete *.mat > 7 days old
#          RETAIN_DAYS=2 ./clean_results.sh   # custom retention
#
# Scheduled via the user crontab (weekly); see the install line printed
# by the session that set it up, or `crontab -l | grep clean_results`.
set -e
RDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
RETAIN_DAYS="${RETAIN_DAYS:-7}"
[ -d "$RDIR" ] || exit 0
find "$RDIR" -maxdepth 1 -name '*.mat' -type f -mtime +"$RETAIN_DAYS" \
     -print -delete
