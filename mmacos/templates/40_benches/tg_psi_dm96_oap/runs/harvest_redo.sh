#!/usr/bin/env bash
# Package A's gate record, in one place (TO 2026-09-17).
# Prints the lines REPORT_bench_realism section 8 quotes, each under its run tag,
# so the report can be checked against the runs rather than against itself.
#   ./runs/harvest_redo.sh            # everything that exists
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
sec () { printf '\n===== %s =====\n' "$*"; }
grepq () { f="$1"; shift; if [ -f "$f" ]; then grep -hE "$@" "$f" || echo "(no match in $f)"; else echo "(missing: $f)"; fi; }

sec "the collimation solve (coll_lens)"
grepq runs/coll_lens/coll_lens_report.txt 'winner|GATES|PASS|FAIL|the winner at|beam |sheet describes|record.s lens'

sec "the tail tunes"
for t in lens96 oap96; do
  printf -- '--- %s\n' "$t"
  grepq runs/tail_$t.log 'TAIL free|TAIL SEED|TAIL WINNER|TAIL GATE|wrote '
done

sec "the rigs as emitted"
for t in redo_lens redo_oap redo_lens_seed; do
  printf -- '--- %s\n' "$t"
  grepq runs/$t/${t}_report.txt 'Tail:|Stage A2|sampling|Stage B --|clearance|worst|stations figure|flat-DM null'
done

sec "the pupil stage = the gate (package A item 2)"
for t in pupilsim_redo_lens pupilsim_redo_oap pupilsim_redo_lens_seed pupilsim_redo_lens_owncone; do
  printf -- '--- %s\n' "$t"
  grepq runs/$t/${t}_report.txt 'BEST PUPIL|phase gain|image surface|astigmatic split|pupil distortion|working surface|Nyquist|THE BEAM at the DM'
done

sec "the flat-DM null, decomposed (nullab)"
for t in nullab_new nullab_nosub nullab_rec nullab_recnos; do
  printf -- '--- %s ' "$t"
  grepq runs/nullab_$t.log 'TAIL SEED'
done

sec "gates, in the brief's words"
cat <<'EOF'
 item 1  exit-ray spread after L1 < 1e-4 rad rms | focal spot < 1 um rms | marker within 0.5 mm  -> coll_lens
 item 2  Nyquist gain >= 0.998 worst | distortion < 0.01 mm rms | band-edge phase < 0.06 rad max -> pupilsim_redo_lens
         (the flat-DM null is REPORTED, not gated)
 item 3  the same, mirror rig                                                                    -> pupilsim_redo_oap
 item 4  both rigs re-emitted, the pupil stage run on the EMITTED decks                          -> the two above
EOF
