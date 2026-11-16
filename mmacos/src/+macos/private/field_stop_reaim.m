function fh = field_stop_reaim(session, rx_path, stop_elt, stop_obj_pos)
%FIELD_STOP_REAIM  Per-field stop re-enforcement action (a closure).
%   FH = FIELD_STOP_REAIM(SESSION, RX_PATH, STOP_ELT, STOP_OBJ_POS)
%   resolves ONCE how this harvest's aperture stop is declared and
%   returns a no-arg handle that re-issues it.  Calling FH() after a
%   per-field set_src_fov re-aims the chief ray THROUGH THE STOP at
%   that field -- the stop-enforced chief IS the field's chief ray
%   (the CLI STOP/PERTURB convention; Dave's ruling 2026-08-28: the
%   supervisors must enforce the stop-enforced chief -- "Do the right
%   thing always").  Without the re-issue the chief stays aimed at the
%   NOMINAL field and the bundle walks on the stop: measured on the
%   zoom fixture (REPORT_wnom_cli_ab A-B4), 1.8-3.1e-6 mm rms of the
%   corner nominal (~7%) and a written EP radius up to 0.76 mm off
%   (sphere CENTER invariant to 8e-5 mm).  How much it matters scales
%   with the source-to-stop geometry -- deck-dependent, so it is
%   enforced ALWAYS, not only where it was measured large.
%
%   Resolution order mirrors the supervisors' stop plumbing:
%     explicit STOP_ELT > explicit STOP_OBJ_POS > the deck's own
%     ApStop= header (object-space 3-vector or element form) > none
%   (FH = no-op; FEX raises its own no-stop error as before).
if ~isempty(stop_elt)
    se = int32(stop_elt);
    fh = @() session.stop(se);
    return
end
if ~isempty(stop_obj_pos)
    p = double(stop_obj_pos);
    fh = @() session.stop_obj(p(1), p(2), p(3));
    return
end
% Deck-declared ApStop= (the segmented-primary idiom): parse the header
% once.  Object-space form = 3-vector; element form = one integer.
fh = @() [];
try
    txt = fileread(rx_path);
catch
    return
end
tok = regexp(txt, '^\s*ApStop=\s*([^\n%]*)', 'tokens', 'once', ...
             'lineanchors');
if isempty(tok), return; end
vals = sscanf(tok{1}, '%f');
if numel(vals) >= 3
    p = double(vals(1:3));
    fh = @() session.stop_obj(p(1), p(2), p(3));
elseif isscalar(vals) && vals == round(vals) && vals >= 1
    se = int32(vals);
    fh = @() session.stop(se);
end
end
