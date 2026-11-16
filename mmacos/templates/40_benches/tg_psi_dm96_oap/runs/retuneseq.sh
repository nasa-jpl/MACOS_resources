#!/usr/bin/env bash
# Re-tune the reflective tail with the FIXED objective (section 4.4):
#   localization dominant, the null BOUNDED not minimized, a wrap guard, and a
#   100 nm tuning poke.  Gated non-vacuous before launching: the fixed cost
#   scores the geometric seed 0.1629 and the old broken winner 4.3944, i.e. it
#   REJECTS by 27x the configuration the old objective chose.
# tg96_tail always seeds from the params' geometric values (it reads no tail
# mat), so this starts from the seed that tailB proved reads at 0.99 -- the
# tune can only be asked to improve on a working point, never to rediscover
# the broken one.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh oapfix "$D,'tag','oapfix'"
echo "[retuneseq] done"
