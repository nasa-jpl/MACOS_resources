#!/usr/bin/env bash
# Gauge-deck sequence 1 (2026-09-13, BRIEF_to_gauge_deck deliverables 1 and 2):
#   pfdeck / pfdeck_loop  -- the P/SRI with BOTH arms traced (pdi.bench 'psri'):
#                            the reference through psri_ref.in's pinhole and the
#                            test through psri_test.in, instead of the
#                            synthesized LP01 mode.  Rows on the 30 nm surface,
#                            photons, the loop rows.
#   cap385p*              -- capture range to 10%: the aging ladder (matrix at
#                            30 nm) and the re-measured rungs (matrix on the
#                            surface), P and PF, 385 rays.
#   noise193p_b*          -- N(1 pm) at 30 / 60 / 120 / 160 nm, 193 rays.
cd "$(dirname "$0")/.."
B="'readings',{'P','PF'}, 'dm_use',1"
S="'battery.calib_surface','base', 'battery.ladder_sites','grid'"
R3="'battery.rows',{'base/single','base/grid','base/rand'}"

# ---- deliverable 1: PF through the two decks ------------------------------
D="'pdi.bench','psri', 'readings',{'PF'}, 'noise.readings',{'PF'}, 'loop.readings',{'PF'}, 'dm_use',1"
./pdi_batch.sh pfdeck      "pdi_params, 'stages',{'bench','battery','noise','figs'}, $D, $S, $R3"
./pdi_batch.sh pfdeck_frz  "pdi_params, 'stages',{'bench','battery'}, $D, $S, $R3, 'pdi.ref_frozen',true"
./pdi_batch.sh pfdeck_loop "pdi_params, 'stages',{'bench','loop'}, $D, 'loop.drifts',{'walk'}, 'loop.steps',[1e-6 10e-6]"

# ---- deliverable 2: capture range and photons, P and PF -------------------
./pdi_batch.sh cap385p "pdi_params, 'NGRID',385, 'stages',{'bench','battery'}, $B, $S, 'battery.rows',{'base/grid'}, 'battery.ladder',[30 40 50 60 80 100 120 160 240 480]*1e-6"
for b in 60 90 120 160; do
    ./pdi_batch.sh cap385p_b$b "pdi_params, 'NGRID',385, 'stages',{'bench','battery'}, $B, $S, 'battery.rows',{'base/grid'}, 'battery.base_rms',${b}e-6, 'battery.ladder',[]"
done
for b in 30 60 120 160; do
    ./pdi_batch.sh noise193p_b$b "pdi_params, 'stages',{'bench','noise'}, $B, 'noise.readings',{'P','PF'}, 'battery.calib_surface','base', 'battery.base_rms',${b}e-6, 'noise.nstates',10.^(8:2:14), 'noise.nreal',6"
done
echo "gseq1 done"
