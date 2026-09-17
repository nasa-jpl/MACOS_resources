function out = tg96_pupil_s2s(varargin)
%TG96_PUPIL_S2S  The engine's STATION-TO-STATION propagation of the detector leg (the CTB idiom).
%   out = TG96_PUPIL_S2S('rig','lens', name/value ...)
%
%   Every leg from the DM propagated, the field never handed back to rays, so the
%   geometric-to-physical hand-off happens where it is exact -- at the DM, the pupil:
%     * collimated gaps between the DM and the focuser: Prop<k>_start (Reference flat,
%       NFPlane, zElt -L) / Prop<k>_end (Reference flat, Geometric, zElt 0) at the chief
%       pierces, the CTB station-to-station idiom (plates between them per index);
%     * the focus: S1 (Reference conic Kr -R1, NF1, zElt +R1) just after the focuser,
%       centered on the TRUE focus; F (Reference flat, NF2) at the focus; S2 (Reference
%       conic Kr -R2, Geometric, zElt -R2) just before the field lens, centered on the
%       focus -- the Rx_Coro through-focus form (asymmetric radii allowed: the chirp is
%       (Z2-Z1) Z1/Z2), PROPER-validated to 6e-13 in the symmetric case;
%     * the field lens per index; S3 (Reference conic, NFS1surf) just after it, concentric
%       with the exit beam (center R3 downstream, from the rays); the Detector with its
%       zElt for the scaled converging step.  This last step's zElt convention is the one
%       thing no deck of record validates; 'conv' tries the candidates and a known 10 mm
%       detector defocus discriminates them against the zone-PSF model.
%   VALIDATION MODE (default 'validate'): the flat DM and the Nyquist sinusoid only, at the
%   plane as built and at +10 mm; per candidate convention: the disc at S2 and at the
%   detector, the Nyquist gain at the center and the edge, against tg96_pupilsim.
%   Name/value: 'rig', 'sim' (the pupilsim run), 'tag' ('pupils2s_<rig>'), 'model' 512,
%   'ngrid' 385, 'n_g' 256, 'dx_g' 0.4, 'conv' ({'+dec','-inc','+inc','-dec'}),
%   'defocus' (10 mm), 'skip_gaps' (false: the collimated NFPlane pairs), 'outdir',
%   'checks' (true: the leg-by-leg gates of BRIEF_to_tg_redo section 2, run before
%   the convention sweep -- each one a number, so a failure names its own leg).
o = struct('rig','lens','sim','','tag','','model',512,'ngrid',385,'n_g',256,'dx_g',0.4, ...
           'conv',{{'+dec','-inc','+inc','-dec'}},'defocus',10,'skip_gaps',false,'gap_min',30,'outdir','', ...
           'checks',true);
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir,'..','..','..','mmacos_setup.m')); end
if isempty(o.sim), o.sim = fullfile(exdir,'runs',['pupilsim_' o.rig]); end
if isempty(o.tag), o.tag = ['pupils2s_' o.rig]; end
if isempty(o.outdir), o.outdir = fullfile(exdir,'runs',o.tag); end
if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
rep = fopen(fullfile(o.outdir,[o.tag '_report.txt']),'w');  say = @(varargin) say_(rep, varargin{:});
% The pupil run this chain is built on is named by its TAG, not by its rig:
% the redo's gate runs are pupilsim_redo_lens / pupilsim_redo_oap, and building
% the filename from 'rig' alone would look for pupilsim_lens.mat inside
% pupilsim_redo_lens/ and fail on a directory that is plainly right there.
[~, simtag] = fileparts(o.sim);
fmat = fullfile(o.sim, [simtag '.mat']);
if ~isfile(fmat), fmat = fullfile(o.sim, ['pupilsim_' o.rig '.mat']); end
assert(isfile(fmat), 'tg96_pupil_s2s: no pupil-stage .mat in %s (looked for %s.mat)', o.sim, simtag);
S = load(fmat);  S = S.out;  lam = S.o.lambda;  pitch = S.o.pitch;
say('=== tg96_pupil_s2s: %s  rig %s  (%s) ===\n', o.tag, o.rig, datestr(now,'yyyy-mm-dd HH:MM'));

% ---- the deck: the simulation's, with the DM's GridData restored ----
fdeck0 = fullfile(o.sim, [simtag '_deck.in']);
if ~isfile(fdeck0), fdeck0 = fullfile(o.sim, ['pupilsim_' o.rig '_deck.in']); end
[hdr, blocks] = split_deck_(fileread(fdeck0));
[~, rblocks] = split_deck_(fileread(S.o.deck));
names = cellfun(@(b) getv_(b,'EltName'), blocks, 'uni', 0);
iDM = find(strcmp(names,'TestOptic'),1);
rDM = rblocks{find(cellfun(@(b) strcmp(getv_(b,'EltName'),'TestOptic'), rblocks),1)};
grid_keys = regexp(rDM, '(?m)^[ \t]*(pData|xData|yData|zData)=[^\n]*\n', 'match');
blocks{iDM} = regexprep(blocks{iDM}, 'Surface=[^\n]*', 'Surface=  GridData');
blocks{iDM} = regexprep(blocks{iDM}, '([ \t]*ApType=)', ['         nGridMat=  ' num2str(o.n_g) '\n         GridFile=  GRIDFILE\n        GridSrfdx=  ' sprintf('%.10E', o.dx_g) '\n' [grid_keys{:}] '$1'], 'once');
hdr = regexprep(hdr, 'nGridpts=\s*[^\n]*', sprintf('nGridpts=  %d', o.ngrid));
V = @(b) getv_(b,'VptElt');  P = @(b) getv_(b,'psiElt');
psi_dm = P(blocks{iDM});  V_dm = V(blocks{iDM});  x_dm = getv_(blocks{iDM},'xObs');  y_dm = cross(psi_dm, x_dm);  R_dm = getv_(blocks{iDM},'ApVec');  R_dm = R_dm(1);
iFoc = find(strcmp(names,'FocalMask'),1);  iFLp = find(strcmp(names,'FLpow'),1);  iFLf = find(strcmp(names,'FLflat'),1);  iDet = numel(blocks);
iL2f = find(strcmp(names,'L2flat'),1);  if isempty(iL2f), iL2f = find(strcmp(names,'L2') | strcmp(names,'OAP2'),1); end   % the mirror rig's focuser is the single reflector L2
psi_det = P(blocks{iDet});  V_det = V(blocks{iDet});  x_det = getv_(blocks{iDet},'xObs');  y_det = cross(psi_det, x_det);
dir_f = unit_(V(blocks{iFoc}) - V(blocks{iL2f}));                      % the beam after the focuser
% the TRUE focus from the rays (the marker is the tuned seat): best focus along the chief from a flat-DM trace
macos.init(o.model);  cd(o.outdir);
macos.write_grid_file('s2s_flat.txt', zeros(o.n_g));
f0 = write_(hdr, blocks, fullfile(o.outdir,[o.tag '_probe.in']), 's2s_flat.txt');  macos.load_rx(f0);  macos.stop(iDM);
s = macos.trace(iFoc);  r = macos.get_ray_info(s.nRays);  okf = r.ok_trace & r.ok_pass;
Vf = V(blocks{iFoc});  pf = r.pos(:,okf) - Vf(:);  df = r.dir(:,okf);  xf = x_det(:)'*pf;  yf = y_det(:)'*pf;
zn = dir_f(:)'*df;  sx = (x_det(:)'*df)./zn;  sy = (y_det(:)'*df)./zn;  z0 = -(dir_f(:)'*pf)./zn;
spotz = @(z) sqrt(var(xf + sx.*(z - z0), 1) + var(yf + sy.*(z - z0), 1));  zb = fminbnd(spotz, -60, 60);
V_focus = V(blocks{iFoc}) + zb*dir_f;
say('true focus %+.2f mm from the mask marker along the beam (spot %.2f um there)\n', zb, spotz(zb)*1e3);
% the exit crossing after the field lens (from the rays)
s = macos.trace(iFLf);  r = macos.get_ray_info(s.nRays);  okF = r.ok_trace & r.ok_pass;
Vfl = V(blocks{iFLf});  pF = r.pos(:,okF) - Vfl(:);  dF = r.dir(:,okF);  xF = x_det(:)'*pF;  yF = y_det(:)'*pF;
aX = (x_det(:)'*dF)./(psi_det(:)'*dF);  aY = (y_det(:)'*dF)./(psi_det(:)'*dF);  r_ = hypot(xF,yF);  a_ = (xF.*aX + yF.*aY)./max(r_,1e-9);
sel = r_ > 0.2*max(r_);  Rc = -median(r_(sel)./a_(sel));
say('exit rays after the field lens cross the axis %+.1f mm from its exit face (+ = downstream, converging)\n', Rc);
delete(f0);

% ---- build the station-to-station deck ----
eps_ = 0.5;
B = {};  ip = 1;
for i = 1:numel(blocks)
    B{end+1} = blocks{i}; %#ok<AGROW>
    if i == iDM || (i < iL2f && i > iDM)
        % a collimated gap after element i?  (up to the focuser)
        Vn = V(blocks{i+1});  Vi = V(blocks{i});  L = norm(Vn - Vi);
        if ~o.skip_gaps && L >= o.gap_min && i+1 <= iL2f
            d = unit_(Vn - Vi);
            B{end+1} = flat_block_(blocks{iFoc}, sprintf('Prop%d_start', ip), Vi + eps_*d, -d, 'NFPlane', -(L - 2*eps_)); %#ok<AGROW>
            B{end+1} = flat_block_(blocks{iFoc}, sprintf('Prop%d_end',   ip), Vn - eps_*d, -d, 'Geometric', 0); %#ok<AGROW>
            say('collimated leg %d: %s -> %s, %.1f mm as NFPlane\n', ip, names{i}, names{i+1}, L - 2*eps_);  ip = ip + 1;
        end
    end
    if i == iL2f
        % the through-focus quartet: S1 just after the focuser's exit, F at the true focus, S2 just before the field lens
        R1 = norm(V_focus - V(blocks{iL2f})) - eps_;  V_S1 = V_focus - R1*dir_f;
        R2 = norm(V(blocks{iFLp}) - V_focus) - eps_;  V_S2 = V_focus + R2*dir_f;
        B{end+1} = sphere_block_(blocks{iFoc}, 'S1', V_S1, dir_f, -R1, 'NF1', R1); %#ok<AGROW>    % psi toward the focus: center = V + R psi (the Rx_Coro convention)
        B{end+1} = flat_block_(blocks{iFoc}, 'F', V_focus, dir_f, 'NF2', -R2); %#ok<AGROW>
        B{end+1} = sphere_block_(blocks{iFoc}, 'S2', V_S2, -dir_f, -R2, 'Geometric', -R2); %#ok<AGROW>
        say('through-focus quartet: S1 at %.2f before the focus (zElt +%.2f, NF1), F at the focus (NF2), S2 at %.2f after it (zElt -%.2f); scale %.4f -> pupil %.2f mm at S2\n', R1, R1, R2, R2, R2/R1, 2*R_dm*R2/R1);
    end
    if i == iFoc, B(end) = []; end                                       % the deck's mask marker is replaced by F
    if i == iFLf
        R3 = abs(Rc) - eps_;  V_S3 = V(blocks{iFLf}) + eps_*psi_det;
        psi3 = psi_det * sign(Rc);   % center = V + R psi: along the beam when the exit beam converges (Rc > 0), against it when it diverges (Rc < 0)
        B{end+1} = sphere_block_(blocks{iFLf}, 'S3', V_S3, psi3, -R3, 'NFS1surf', R3); %#ok<AGROW>
        say('S3 after the field lens: radius %.2f mm, concentric with the converging exit beam; detector %.2f mm beyond it\n', R3, norm(V_det - V_S3));
    end
end
names2 = cellfun(@(b) getv_(b,'EltName'), B, 'uni', 0);  iS2 = find(strcmp(names2,'S2'));  iS3 = find(strcmp(names2,'S3'));  iD2 = numel(B);  iDM2 = find(strcmp(names2,'TestOptic'));
d_S3_det = norm(V_det - (V(blocks{iFLf}) + eps_*psi_det));
deck = write_(hdr, B, fullfile(o.outdir,[o.tag '_deck.in']), 'GRIDFILE');
if isfield(o,'build_only') && o.build_only, out = struct('deck',deck,'B',{B},'hdr',hdr,'R3',R3); fclose(rep); return; end

% ---- the legs checked ONE AT A TIME (BRIEF_to_tg_redo section 2) ----
% The convention sweep below reports the END of the chain, where every leg's
% error is already mixed into one gain map.  These four gates sit on the legs
% themselves, in the order light meets them, so a failure names its own leg.
iS1c = find(strcmp(names2,'S1'));  iFc = find(strcmp(names2,'F'));
if o.checks && ~isempty(iS1c)
    chk = struct();
    amp_c = 5e-6;  fN_c = 0.5;                       % the Nyquist sinusoid, 5 nm
    xa_c = ((1:o.n_g) - (o.n_g+1)/2)*o.dx_g;  [UC, VC] = meshgrid(xa_c, xa_c);
    macos.write_grid_file('s2s_chk_sin.txt', amp_c*sin(2*pi*fN_c*UC'));
    fflat = write_(hdr, B, fullfile(o.outdir,[o.tag '_chk_flat.in']), 's2s_flat.txt');
    fsin  = write_(hdr, B, fullfile(o.outdir,[o.tag '_chk_sin.in']),  's2s_chk_sin.txt');

    % (1) the collimated legs: a flat DM must arrive at the entrance sphere
    % FLAT.  S1 is a reference sphere centred on the focus and the beam there
    % is converging to it, so the sphere removes the nominal curvature and
    % what is left is the leg's own error -- 1.5 waves of it on the lens rig
    % of record, which is what walked the rays off the grid.
    macos.load_rx(fflat);  macos.stop(iDM2);
    E1 = macos.complex_field(iS1c);  dx1 = abs(macos.dx_at(iS1c,'mm'));
    I1 = abs(E1).^2;  in1 = I1 > 0.2*max(I1(:));
    ph = angle(E1(in1));  ph = unwrap_med_(ph);
    chk.s1_phase_wv = std(ph)/(2*pi);
    chk.s1_r = eqr_(E1, dx1);
    say('CHECK 1a  collimated legs: the flat pupil at S1 carries %.4f wave rms of phase (gate < 0.02); pupil radius there %.2f mm, pitch %.4f mm\n', ...
        chk.s1_phase_wv, chk.s1_r, dx1);

    % (1b) and a Nyquist sinusoid must still be its own height there.  The
    % legs are collimated, so S1's pupil is the DM's: demodulate on the same
    % frequency, in the field's own pitch.
    macos.load_rx(fsin);  macos.stop(iDM2);
    E1s = macos.complex_field(iS1c);
    h1 = angle(E1s .* conj(E1)) * lam/(4*pi);        % the surface the leg reports, mm
    % Read the amplitude off the TRANSFORM's own peak, not off a carrier
    % built from the DM's axes: S1 inherits the mask's frame, which need not
    % be the DM's, and a demodulation on a rotated or flipped carrier reads
    % zero and looks exactly like a leg that lost the mode.
    hm = h1 .* in1;  Hf = fft2(hm)/max(nnz(in1),1);  Hf(1,1) = 0;
    [~, ipk] = max(abs(Hf(:)));  [kr_, kc_] = ind2sub(size(Hf), ipk);
    n1 = size(E1,1);  fr = mod(kr_-1 + n1/2, n1) - n1/2;  fc = mod(kc_-1 + n1/2, n1) - n1/2;
    chk.s1_amp_rel = 2*abs(Hf(ipk))/amp_c;
    chk.s1_freq = hypot(fr, fc)/(n1*dx1);
    say('CHECK 1b  a Nyquist sinusoid reaches S1 at %.4f of its height on the DM (gate 1.00 +- 0.01), at %.4f cycles/mm (the DM''s %.4f)\n', ...
        chk.s1_amp_rel, chk.s1_freq, fN_c);

    % (2a) the quartet: the focal field must be the pupil's Airy pattern.
    macos.load_rx(fflat);  macos.stop(iDM2);
    EF = macos.complex_field(iFc);  dxF = abs(macos.dx_at(iFc,'mm'));
    IF_ = abs(EF).^2;  [~, im] = max(IF_(:));  [rF, cF] = ind2sub(size(IF_), im);
    prof = radial_(IF_, rF, cF, dxF);
    z1 = first_zero_(prof.r, prof.v);
    airy = 1.22*lam*R1/(2*R_dm);
    chk.airy_meas = z1;  chk.airy_pred = airy;
    say('CHECK 2a  the focal field''s first zero %.1f um against the pupil''s Airy %.1f um (1.22 lam R1/D, R1 %.1f mm, D %.1f mm); pitch %.2f um\n', ...
        1e3*z1, 1e3*airy, R1, 2*R_dm, 1e3*dxF);

    % (2b) the far-side pupil: the FIELD's radius at S2 against the RAYS' own
    % footprint there -- not against R_dm*R2/R1, which is the prediction under
    % test.
    ES2 = macos.complex_field(iS2);  dxS2 = abs(macos.dx_at(iS2,'mm'));
    st = macos.trace(iS2);  ri = macos.get_ray_info(st.nRays);  okS = ri.ok_trace & ri.ok_pass;
    VS2 = getv_(B{iS2},'VptElt');  pS = ri.pos(:,okS) - VS2(:);
    psiS = getv_(B{iS2},'psiElt');  pS = pS - psiS(:)*(psiS(:)'*pS);
    chk.s2_ray_r = max(sqrt(sum(pS.^2,1)));
    chk.s2_fld_r = eqr_(ES2, dxS2);
    chk.s2_pred_r = R_dm*R2/R1;
    % THE PITCH IS THE DECISIVE NUMBER, and it is worth knowing why.  The
    % engine's dxElt is set by the propagator only for PropType 3/10/15 (the
    % far-field pitch lam*z1/(N dx1), which is how the focal plane F gets its
    % label) and for the regridded sphere-to-sphere pair 5/14.  PropType 11 --
    % PL2SPH, our F -> S2 leg -- is in NEITHER list, so S2's pitch falls
    % through to the RAY-SPACING branch (utilsub.F ~:590, a slice across the
    % centre of the pupil).  Meanwhile PL2SPH is a plain inverse FFT: the
    % ARRAY comes back in the entrance sphere's own pitch, i.e. in the
    % Sziklas-Siegman SCALED frame, with the pupil occupying the same pixels
    % it did at S1.  The two conventions are consistent only if the ray
    % spacing at S2 really is dx(S1)*R2/R1 -- so compare them directly, and a
    % disagreement here is the pitch label, not the estimator.
    chk.s2_dx = dxS2;  chk.s2_dx_scaled = dx1*R2/R1;
    say('CHECK 2b  pupil radius at S2: field %.3f mm, rays %.3f mm (%.1f%%), thin-lens prediction R_dm*R2/R1 %.3f mm\n', ...
        chk.s2_fld_r, chk.s2_ray_r, 100*(chk.s2_fld_r/chk.s2_ray_r - 1), chk.s2_pred_r);
    say('CHECK 2c  the pitch at S2: the engine labels %.6f mm (ray spacing); the scaled frame says dx(S1)*R2/R1 = %.6f mm (%.1f%%)\n', ...
        chk.s2_dx, chk.s2_dx_scaled, 100*(chk.s2_dx/chk.s2_dx_scaled - 1));
    say('CHECK gates: 1a %s, 1b %s, 2a %s, 2b %s, 2c %s\n', ...
        pf_(chk.s1_phase_wv < 0.02), pf_(abs(chk.s1_amp_rel-1) < 0.01), ...
        pf_(abs(z1/airy - 1) < 0.10), pf_(abs(chk.s2_fld_r/chk.s2_ray_r - 1) < 0.02), ...
        pf_(abs(chk.s2_dx/chk.s2_dx_scaled - 1) < 0.02));
    delete(fflat);  delete(fsin);
else
    chk = struct();
end

% ---- the DM frame and the two test surfaces ----
N = S.o.N;  dx = S.o.dx;  xg = (-N/2:N/2-1)*dx;  [XG, YG] = meshgrid(xg, xg);  rr = hypot(XG, YG);
R_beam = min(R_dm, max(hypot(S.uv(1,S.ok), S.uv(2,S.ok))) + 0.5);  Rin = R_beam - 1.0;  lit = rr <= Rin;
fx = ifftshift((-N/2:N/2-1)/(N*dx));  [FU, FV] = meshgrid(fx, fx);
xa_g = ((1:o.n_g) - (o.n_g+1)/2)*o.dx_g;  [UG, VG] = meshgrid(xa_g, xa_g);
fN = 0.5;  amp = 5e-6;  h_sin = @(U,Vv) amp*sin(2*pi*fN*U);
Msin = h_sin(UG', VG');  macos.write_grid_file('s2s_sin.txt', Msin);
truth = interp2(UG, VG, Msin', XG, YG, 'linear', 0);
% the ray map DM -> detector array (dmg_frame convention), orientation resolved on the sinusoid's phase later
u2 = macos.design.Bench.perp(psi_det(:));  v2 = cross(psi_det(:), u2);  Rrot = [u2'; v2'] * [x_det(:) y_det(:)];
ok = S.ok;  M = [S.uv(1,ok); S.uv(2,ok); ones(1,nnz(ok))]';  Ax = M \ S.r0(1,ok)';  Ay = M \ S.r0(2,ok)';
XD = Ax(1)*XG + Ax(2)*YG + Ax(3);  YD = Ay(1)*XG + Ay(2)*YG + Ay(3);  UD = Rrot(1,1)*XD + Rrot(1,2)*YD;  VD = Rrot(2,1)*XD + Rrot(2,2)*YD;
variants = {@(u,v) [u v], @(u,v) [-u v], @(u,v) [u -v], @(u,v) [-u -v], @(u,v) [v u], @(u,v) [-v u], @(u,v) [v -u], @(u,v) [-v -u]};
jz = find([S.res(1).G.f] == fN & [S.res(1).G.dir] == 1, 1);  Gref = S.res(1).G(jz);
say('zone-PSF model, Nyquist, as built: gain by radius [%s] (mean %.4f, min %.4f)\n', sprintf('%.3f ', Gref.gain_r), Gref.gmean, Gref.gmin);
% the zone model's prediction at +defocus: W -> W + dz a^2/2 for every zone
k0 = 2*pi/lam;  ab = S.ab;
phi_c = k0*abs(S.zimg(ok) - o.defocus)*ab^2/2;   % band-edge phase per zone at the shifted plane (the zone image is zimg downstream; the plane moves by +defocus)
say('zone model at +%.0f mm: band-edge phase %.3f rad rms -> Nyquist gain ~ cos: mean %.3f, worst %.3f (as built: mean %.3f)\n', o.defocus, rms_(phi_c), mean(cos(phi_c)), min(cos(phi_c)), mean(cos(k0*abs(S.zimg(ok))*ab^2/2)));

res = struct();
for ic = 1:numel(o.conv)
    cv = o.conv{ic};
    for ip = 1:2
        dz = (ip-1)*o.defocus;
        B2 = B;
        Vd = V_det + dz*psi_det;
        B2{iD2} = regexprep(B2{iD2}, 'VptElt=[^\n]*', sprintf('VptElt=  %.12g  %.12g  %.12g', Vd));
        B2{iD2} = regexprep(B2{iD2}, 'RptElt=[^\n]*', sprintf('RptElt=  %.12g  %.12g  %.12g', Vd));
        Ls = d_S3_det + dz;
        switch cv    % zElt(S3), zElt(Detector): 'dec' = the distance to the center decreases along the beam (converging); for a diverging beam (Rc < 0) 'inc' is the physical one
            case '+dec', z1 = R3;  z2 = R3 - Ls;
            case '-inc', z1 = -R3; z2 = -(R3 - Ls);
            case '+inc', z1 = R3;  z2 = R3 + Ls;
            case '-dec', z1 = -R3; z2 = -(R3 + Ls);
        end
        B2{iS3} = set_prop_(B2{iS3}, 'NFS1surf', z1);  B2{iD2} = set_prop_(B2{iD2}, 'Geometric', z2);
        E = cell(1,2);  dxd = NaN;  dxs2 = NaN;  rS2 = NaN;
        for is = 1:2
            gf = {'s2s_flat.txt','s2s_sin.txt'};  fdeck = write_(hdr, B2, fullfile(o.outdir, sprintf('%s_%s_p%d_%d.in', o.tag, strrep(cv,'+','p'), ip, is)), gf{is});
            macos.load_rx(fdeck);  macos.stop(iDM2);
            if is == 1, Es2 = macos.complex_field(iS2);  dxs2 = abs(macos.dx_at(iS2,'mm'));  rS2 = eqr_(Es2, dxs2); end
            E{is} = macos.complex_field(iD2);  dxd = abs(macos.dx_at(iD2,'mm'));  delete(fdeck);
        end
        Nd = size(E{1},1);  I = abs(E{1}).^2;  [C1, C2] = meshgrid(1:Nd, 1:Nd);  c1 = sum(C1(:).*I(:))/sum(I(:));  c2 = sum(C2(:).*I(:))/sum(I(:));
        rdet = eqr_(E{1}, dxd);
        % orientation: the sinusoid must demodulate to a gain map; pick the variant with the largest mean demodulated gain (a flipped map demodulates to ~0)
        best = [];
        for iv = 1:numel(variants)
            q = variants{iv}(UD, VD);  ROWm = q(:,1:N)/dxd + c2;  COLm = q(:,N+1:end)/dxd + c1;
            samp = @(A) interp2(A, COLm, ROWm, 'linear', 0);
            Er = samp(E{1});  ho = angle(samp(E{2}).*conj(Er))*lam/(4*pi);  car = exp(-1i*2*pi*fN*XG);  ap = double(rr <= R_beam);
            gmap = abs(lpf_(ho.*car.*ap, FU, FV, 2/fN)) ./ max(abs(lpf_(truth.*car.*ap, FU, FV, 2/fN)), 1e-12);
            sc = mean(abs(gmap(lit) - 1));   % the right orientation reads a gain near 1; a mirrored map demodulates to noise
            if isempty(best) || sc < best.g, best = struct('iv',iv,'g',sc,'gmap',gmap,'Er',Er); end
        end
        gmap = best.gmap;  rb = 0:0.1:1;  rc = rr/Rin;  gr = zeros(1,numel(rb)-1);  for b = 1:numel(rb)-1, m = lit & rc >= rb(b) & rc < rb(b+1); gr(b) = mean(gmap(m)); end
        say('conv %-4s plane %+3.0f mm: pupil at S2 %.2f mm radius (expected %.2f); at the detector %.2f mm (rays %.2f); |E| flat rms-var %.3f; Nyquist gain by radius [%s] mean %.4f min %.4f (orientation %d)\n', ...
            cv, dz, rS2, R_dm*R2/R1, rdet, R_beam/(1/S.magL), std(abs(best.Er(lit)))/mean(abs(best.Er(lit))), sprintf('%.3f ', gr), mean(gmap(lit)), min(gmap(lit)), best.iv);
        res(ic,ip).conv = cv;  res(ic,ip).dz = dz;  res(ic,ip).gain_r = gr;  res(ic,ip).gmean = mean(gmap(lit));  res(ic,ip).gmin = min(gmap(lit));  res(ic,ip).rS2 = rS2;  res(ic,ip).rdet = rdet;
    end
end
out = struct('o',o,'res',res,'deck',deck,'R1',R1,'R2',R2,'R3',R3,'zb',zb,'chk',chk);
save(fullfile(o.outdir,[o.tag '.mat']), 'out');  say('run complete\n');  fclose(rep);
end

% ---------------------------------------------------------------------------
function b = sphere_block_(tmpl, name, vpt, psi, kr, ptype, zelt)
b = flat_block_(tmpl, name, vpt, psi, ptype, zelt);
b = regexprep(b, 'Surface=[^\n]*', 'Surface=  Conic');
b = regexprep(b, 'KrElt=[^\n]*', sprintf('KrElt=  %.10E', kr));
b = regexprep(b, 'KcElt=[^\n]*', 'KcElt=  0.0000000000E+00');
end
function b = flat_block_(tmpl, name, vpt, psi, ptype, zelt)
b = tmpl;
b = regexprep(b, 'EltName=[^\n]*', ['EltName=  ' name]);
b = regexprep(b, 'Element=[^\n]*', 'Element=  Reference');
b = regexprep(b, 'Surface=[^\n]*', 'Surface=  Flat');
b = regexprep(b, 'KrElt=[^\n]*', 'KrElt=  -1.0000000000E+22');
b = regexprep(b, 'KcElt=[^\n]*', 'KcElt=  0.0000000000E+00');
b = regexprep(b, 'psiElt=[^\n]*', sprintf('psiElt=  %.12g  %.12g  %.12g', psi));
x = macos.design.Bench.perp(psi(:));
b = regexprep(b, 'xObs=[^\n]*', sprintf('xObs=  %.12g  %.12g  %.12g', x));
b = regexprep(b, 'VptElt=[^\n]*', sprintf('VptElt=  %.12g  %.12g  %.12g', vpt));
b = regexprep(b, 'RptElt=[^\n]*', sprintf('RptElt=  %.12g  %.12g  %.12g', vpt));
b = regexprep(b, 'IndRef=[^\n]*', 'IndRef=  1.000000E+00');
b = regexprep(b, '[ \t]*ApType=[^\n]*\n', '         ApType=  None\n');
b = regexprep(b, '[ \t]*ApVec=[^\n]*\n', '');
b = regexprep(b, '[ \t]*nObs=[^\n]*\n', '             nObs=  0\n');
b = set_prop_(b, ptype, zelt);
end
function b = set_prop_(b, ptype, zelt)
b = regexprep(b, 'PropType=[^\n]*', ['PropType=  ' ptype]);
if isempty(regexp(b, '(?m)^[ \t]*zElt=', 'once')), b = regexprep(b, '(PropType=[^\n]*\n)', sprintf('$1             zElt=  %.10E\n', zelt));
else, b = regexprep(b, 'zElt=[^\n]*', sprintf('zElt=  %.10E', zelt)); end
end
function f = write_(hdr, blocks, f, gridfile)
for i = 1:numel(blocks), blocks{i} = regexprep(blocks{i}, '(?m)^([ \t]*iElt=[ \t]*)\d+', sprintf('$1%d', i)); blocks{i} = strrep(blocks{i}, 'GRIDFILE', gridfile); end
hdr = regexprep(hdr, 'nElt=\s*\d+', sprintf('nElt=  %d', numel(blocks)));
fid = fopen(f,'w'); fwrite(fid, [hdr blocks{:}]); fclose(fid);
end
function r = eqr_(E, d), I = abs(E).^2; r = sqrt(sum(I(:) > 0.5*median(I(I > 0.05*max(I(:)))))*d^2/pi); end
function m = lpf_(x, FU, FV, sig), m = ifft2(fft2(x) .* exp(-2*pi^2*sig^2*(FU.^2 + FV.^2))); end
function p = unwrap_med_(ph), p = angle(exp(1i*(ph - median(ph)))); end
function s = pf_(tf), if tf, s = 'PASS'; else, s = 'FAIL'; end, end
function prof = radial_(I, r0, c0, dx)
[NR, NC] = size(I);  [C, R] = meshgrid(1:NC, 1:NR);  rr = hypot(R-r0, C-c0)*dx;
edges = 0:dx/2:min(30*dx, max(rr(:)));  v = zeros(1, numel(edges)-1);
for k = 1:numel(edges)-1, m = rr >= edges(k) & rr < edges(k+1); if any(m(:)), v(k) = mean(I(m)); end, end
prof = struct('r', (edges(1:end-1)+edges(2:end))/2, 'v', v);
end
function z = first_zero_(r, v)
% the first radius at which the azimuthal mean stops falling: an Airy null
v = v/max(v);  z = NaN;
for k = 2:numel(v)-1
    if v(k) < v(k-1) && v(k) <= v(k+1) && v(k) < 0.1, z = r(k); return; end
end
if isnan(z), [~, k] = min(v);  z = r(k); end
end
function [hdr, blocks] = split_deck_(txt)
idx = regexp(txt, '\n[ \t]*iElt=');  hdr = txt(1:idx(1));  blocks = cell(1, numel(idx));
for i = 1:numel(idx), e = numel(txt); if i < numel(idx), e = idx(i+1); end; blocks{i} = txt(idx(i)+1:e); end
end
function v = getv_(blk, key)
m = regexp(blk, ['(?m)^[ \t]*' key '=[ \t]*([^\n]*)'], 'tokens', 'once');
if isempty(m), v = []; return; end
v = str2num(regexprep(m{1}, '[DdEe]([+-]?\d)', 'e$1')); %#ok<ST2NM>
if isempty(v), v = strtrim(m{1}); end
end
function u = unit_(v), u = v(:)' / norm(v); end
function r = rms_(x), x = x(isfinite(x)); r = sqrt(mean(x(:).^2)); end
function say_(rep, varargin), fprintf(varargin{:}); fprintf(rep, varargin{:}); end
