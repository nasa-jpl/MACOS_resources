#!/usr/bin/env bash
set -u
B=~/dev/MACOS_resources/mmacos/templates/40_benches
export MACOS_HOME=$HOME/dev/macos/macos_f90
say(){ echo "[$(date '+%F %T')] $*"; }
cd "$B/zwfs_dm96"
say "3/4 node figures + vector layout (retry: dm_gauge_lib on the path)"
matlab -batch "run('$B/../../mmacos_setup.m'); bench_node_figs; zwfs_vlayout; exit(0)" </dev/null; say "  rc=$?"
cd "$B/pdi_dm96"
say "4/4 pdi layout (retry: coat_* stripped from the builder args)"
matlab -batch "run('$B/../../mmacos_setup.m'); pdi_layout_fig; exit(0)" </dev/null; say "  rc=$?"
say "LAYOUT FIGURES 3-4 DONE"
