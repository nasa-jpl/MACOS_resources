#!/usr/bin/env bash
# Regenerate every LAYOUT figure the deck uses, on the bench as it stands
# (substrates in, the beam opened, the vlayout's reference arm drawn first so
# the shared output leg reads blue).  One MATLAB at a time.
set -u
B=~/dev/MACOS_resources/mmacos/templates/40_benches
export MACOS_HOME=$HOME/dev/macos/macos_f90
say(){ echo "[$(date '+%F %T')] $*"; }

cd "$B/tg_psi_dm96_oap"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
say "1/4 OAP rig vlayout (lay96_oap)"
TG96_NOWAIT=1 ./tg96_batch.sh lay96_oap "$OAP,'stages',{'bench','figs'}" </dev/null; say "  rc=$?"
say "2/4 lens rig vlayout (lay96_lens)"
TG96_NOWAIT=1 ./tg96_batch.sh lay96_lens "'stages',{'bench','figs'}" </dev/null; say "  rc=$?"

cd "$B/zwfs_dm96"
say "3/4 splitter-angle node figures (7 / 22.5 / 30) + the vector sensor layout"
matlab -batch "run('$B/../../mmacos_setup.m'); bench_node_figs; zwfs_vlayout; exit(0)" </dev/null; say "  rc=$?"

cd "$B/pdi_dm96"
say "4/4 point-diffraction layout"
matlab -batch "run('$B/../../mmacos_setup.m'); pdi_layout_fig; exit(0)" </dev/null; say "  rc=$?"
say "LAYOUT FIGURES DONE"
