function E = tls_e2e(P, deck, opts)
%TLS_E2E  End to end: a long-slit telescope deck joined to a spectrometer deck, scored by the dyson5 engine join.
%
%   E = TLS_E2E(P, DECK) joins the telescope DECK (a TLS_SECTION / TLS_FIGURE
%   emission) to the spectrometer of P.e2e (default: the dyson5 3k Dyson of
%   record, CaF2 240 mm, 'size:F:240', whose Slit + Dyson blocks come from
%   P.e2e.template) through challenges/dyson5/dyson5_t5f -- the join placed
%   from ENGINE traces, scored by spectrometer_score: smile, keystone, CRF, SRF,
%   energy in a pixel per (field, wavelength), the grating's admitted fraction,
%   the live clearance of the joined deck.  Nothing is re-implemented here.
%
%   The one adaptation: t5f aims every sky field through an OBJECT-SPACE
%   stop (the deck header's ApStop), while this template's decks carry an
%   ELEMENT stop on M2.  In the joined deck the instrument's stop is the
%   grating (t5f re-aims every launch through it), so the telescope stop
%   only places the field directions; the join therefore takes a COPY of
%   DECK with the header ApStop = the entrance pupil (the StopPos the engine
%   computes from the M2 stop at load: the chief's object-space crossing) and
%   the element ApStop removed.
%
%   E = TLS_E2E(..., 'launch', L) picks t5f's field launch: 'chief' (each field's
%   chief on the slit line -- the record's POINT-SOURCE convention: the
%   telescope's chief-minus-centroid offset across the slit reads as smile) or
%   'centroid' (the bundle centroid on the slit line -- the SLIT-FILLED proxy,
%   the spec's convention per CC 2026-10-07).
%   E = TLS_E2E(..., 'roll_deg', R) rolls the telescope about the exit
%   chief in the join (t5f's tel5e_roll_deg; default [0 180], both scored).
%
%   See also TMA_LONGSLIT_RUN, dyson5_t5f.
arguments
    P struct
    deck (1,:) char
    opts.roll_deg (1,:) double = [0 180]
    opts.suffix (1,:) char = '_tls'
    opts.launch (1,:) char {mustBeMember(opts.launch, {'chief', 'centroid'})} = 'chief'
end
here = fileparts(mfilename('fullpath'));
ddir = fullfile(here, '..', '..', '..', 'challenges', 'dyson5');
addpath(ddir);  addpath(fullfile(here, '..', '..', '..', 'design', 'src'));
e = struct('tel_dyson', 'size:F:240', 'template', 'dyson5_t5e_tA_EP_3k_m30_B1_e2e.in');
if isfield(P, 'e2e') && ~isempty(P.e2e), fn = fieldnames(P.e2e); for i = 1:numel(fn), e.(fn{i}) = P.e2e.(fn{i}); end, end
% ---- the entrance pupil from the engine: where the object-space chiefs of two fields cross, each aimed by the
% deck's ELEMENT stop at load (the centre field and a 1-deg strip field; a collimated source: the source frame's
% chief line IS the object-space chief)
macos.init(P.model);
txt = fileread(deck);  tmp = [tempname '_tlse2e.in'];  L = zeros(3, 2);  D = zeros(3, 2);
for j = 1:2
    dd = [sind(j - 1); 0; cosd(j - 1)];
    t2 = regexprep(txt, '(?m)^(\s*ChfRayDir=).*$', sprintf('$1  %.16E  %.16E  %.16E', dd), 'dotexceptnewline');
    fid = fopen(tmp, 'w');  fwrite(fid, t2);  fclose(fid);  macos.load_rx(tmp);
    sf = macos.get_src_fov();  L(:, j) = sf.src_pos(:);  D(:, j) = sf.src_dir(:)/norm(sf.src_dir);
end
delete(tmp);
% closest approach of the two lines
w0 = L(:, 1) - L(:, 2);  a = D(:, 1).'*D(:, 1);  b = D(:, 1).'*D(:, 2);  c = D(:, 2).'*D(:, 2);
dd1 = D(:, 1).'*w0;  e2 = D(:, 2).'*w0;  den = a*c - b^2;
t1 = (b*e2 - c*dd1)/den;  t2 = (a*e2 - b*dd1)/den;
ep = (L(:, 1) + t1*D(:, 1) + L(:, 2) + t2*D(:, 2))/2;
ep_miss = norm((L(:, 1) + t1*D(:, 1)) - (L(:, 2) + t2*D(:, 2)));
txt = regexprep(txt, '(?m)^\s*ApStop=[^\n]*\n', '');                   % drop the element stop
txt = regexprep(txt, '(?m)^(\s*GridType=)', sprintf('         ApStop=  %.16E  %.16E  %.16E\n$1', ep), 'once');
jdeck = fullfile(ddir, sprintf('tls_e2e%s_tel.in', opts.suffix));
fid = fopen(jdeck, 'w');  fwrite(fid, txt);  fclose(fid);
% ---- the dyson5 join, once per roll
olddir = cd(ddir);  cln = onCleanup(@() cd(olddir));
E = struct('deck', deck, 'join_deck', jdeck, 'ap_stop', ep, 'ep_miss', ep_miss, 'launch', opts.launch, 'rows', []);
for r = opts.roll_deg
    ov = struct('tel_dyson', e.tel_dyson, 'tel5f_e2e_template', e.template, 'tel5e_roll_deg', r, ...
                'tel5f_deck', jdeck, 'tel5f_suffix', sprintf('%s_%s_roll%03d', opts.suffix, opts.launch, r), 'tel5f_launch', opts.launch);
    % every other P.e2e field is a dyson5_params override too (the 1.5k join: tel_npix_xt 1500)
    for f = setdiff(fieldnames(e)', {'tel_dyson', 'template'}), ov.(f{1}) = e.(f{1}); end
    Pd = dyson5_params(ov);
    % (t5f's LIVE clearance is NOT used: its body model lifts each mirror onto the PARENT's base sphere about the parent
    % vertex, and these sections' poles sit up to ~1.7 m off their parent axes -- beyond that sphere; tel_deck_geom's chain
    % also loses the centre chief on them.  The joined-deck clearance is TLS_CLEARANCE_JOINED: footprint bodies from
    % ENGINE rays on every element, the rule of the record.)
    S5 = dyson5_t5f(Pd, 'dyson5');
    E.rows = [E.rows, struct('roll_deg', r, 'S', S5)];
end
end
