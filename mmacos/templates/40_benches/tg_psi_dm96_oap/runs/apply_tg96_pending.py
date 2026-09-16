#!/usr/bin/env python3
"""Two tg96_run.m fixes, blocked until item2seq releases the file.

1. THE WRAP METER SATURATES.  max|h|/(lambda/4) reads 1.00 at 60, 120, 240 and
   480 nm alike (measured, oapuw2), so it cannot discriminate the rungs it was
   added to discriminate.  Replace it with the two quantities that do: the
   MEASURED base rms (what actually wraps, as opposed to the commanded
   surface) and the FRACTION of the lit pupil within a whisker of the fold --
   which is the statistic the ZWFS battery has always reported as fold0/fold.

2. THE CAMERA LINE MOVES TO THE SHARED TOOL.  dmg_cam_line (dm_gauge_lib) is
   already used by zwfs_run; it adds the binning that lands nearest the modeled
   sampling (the sheet's 4 undersamples the model by 22% on this rig) and fixes
   the polarization camera's per-orientation count to N/2 across a line, not
   N/4.  tg96_run keeps a local copy from before that tool existed.

Run, from THIS directory, once nothing is executing tg96_run.m:
    python3 apply_tg96_pending.py
Editing tg96_run.m while an hour-class run is inside it is how a long job gets
corrupted, which is why this is a staged patch and not an edit.
"""
import io, sys, re
p = '/home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/tg_psi_dm96_oap/tg96_run.m'
s = open(p, encoding='utf-8').read()

# ---- 1. the ladder's meter -------------------------------------------
old = """    say('  %-10s %8s %10s %8s %7s  %s\\n','base rms','gain','floor pm','corr','wrap','note');"""
new = """    say('  %-10s %8s %10s %8s %9s %7s  %s\\n', 'base rms','gain','floor pm','corr', ...
        'meas rms','fold','note');"""
assert old in s, 'meter header not found'
s = s.replace(old, new, 1)

old = """        pwrap = max(abs(hb(msk)));                    % how far the base reading reaches vs lambda/4"""
new = """        % The base reading's MEASURED rms, and the FRACTION of the lit pupil
        % within 2 % of the four-step fold.  max|h| was the meter here and it
        % SATURATES: measured on oapuw2 it reads 1.00 of lambda/4 at 60, 120,
        % 240 and 480 nm alike, so it cannot tell a rung that holds from one
        % that breaks.  The fraction is what the ZWFS battery reports
        % (fold0/fold) and what the wrapped-absolute subtraction actually
        % scales with -- the difference measr(bb+d)-measr(bb) is exact wherever
        % both maps wrapped equally and wrong by lambda/2 at the pixels the
        % poke pushes across a boundary.  The measured rms sits beside it
        % because what wraps is the MEASURED map, not the commanded surface,
        % and the two rigs differ there.
        meas_rms = std(hb(msk));
        fold_fr  = mean(abs(hb(msk)) > 0.98*qwave);"""
assert old in s, 'pwrap line not found'
s = s.replace(old, new, 1)

old = """        say('  %6.0f nm %8.4f %10.1f %8.4f %7.2f  %s\\n', lad(j), g, r, cc(1,2), pwrap/qwave, note);"""
new = """        say('  %6.0f nm %8.4f %10.1f %8.4f %8.1f nm %6.3f  %s\\n', ...
            lad(j), g, r, cc(1,2), 1e6*meas_rms, fold_fr, note);"""
assert old in s, 'ladder print not found'
s = s.replace(old, new, 1)

old = "        brk(j,:) = [lad(j) g r cc(1,2) double(broke)];"
new = "        brk(j,:) = [lad(j) g r cc(1,2) double(broke) meas_rms fold_fr];"
assert old in s, 'brk row not found'
s = s.replace(old, new, 1)
s = s.replace("lad = P.battery.break_ladder;  brk = zeros(numel(lad),5);",
              "lad = P.battery.break_ladder;  brk = zeros(numel(lad),7);", 1)

# ---- 2. the camera line -> the shared tool ----------------------------
old = "    cam_line_(P, say, PL.dxd_mm, msk);"
new = "    if isfield(P,'cam'), dmg_cam_line(say, P.cam, PL.dxd_mm, msk); end"
assert old in s, 'cam_line_ call not found'
s = s.replace(old, new, 1)
# drop the local copy, now superseded -- matched by its exact first and last
# lines so no brace counting is involved
start = "\nfunction cam_line_(P, say, dxd_mm, msk)"
tail  = "        d_pup_mm/(P.cam.pol_pitch_um*1e-3)/2);\nend\nend\n"
a = s.find(start);  assert a > 0, 'local cam_line_ start not found'
b = s.find(tail, a);  assert b > a, 'local cam_line_ end not found'
s = s[:a] + s[b+len(tail):]

# ---- 3. a WRAP stage: the base reading's own statistics, no matrix -----
old = "battery = struct();\nif want('battery')"
new = "if want('wrap')\n    stage_wrap_(P, s, G, bench, say);\nend\n\nbattery = struct();\nif want('battery')"
assert old in s, 'stage dispatch not found'
s = s.replace(old, new, 1)
stage = open('stage_wrap.m.txt', encoding='utf-8').read()
s = s.rstrip(chr(10)) + chr(10) + stage

# ---- 4. the stations figure lands at the width the brief asks for ------
# exportgraphics(...,'Resolution',150) on an 1800 px figure lands near 2440 px,
# not 1800.  The sibling ZWFS figure uses print -dpng -r130 on a 2000 px figure
# and lands at 2709 (measured: 2000 * 130/96), i.e. this box renders at 96 dpi.
# So -r96 on an 1800 px figure gives exactly 1800.  Matching the sibling's
# MECHANISM as well as its intent, because print and exportgraphics scale
# differently and the deck holds both figures side by side.
old = "exportgraphics(f, [P.tag \'_stations.png\'], \'Resolution\', 150);  close(f);"
new = "print(f, [P.tag \'_stations.png\'], \'-dpng\', \'-r96\');  close(f);   % 1800 px wide"
assert old in s, 'stations export line not found'
s = s.replace(old, new, 1)

open(p, 'w', encoding='utf-8').write(s)
print('tg96_run.m: ladder meter replaced, camera line moved to dmg_cam_line')
