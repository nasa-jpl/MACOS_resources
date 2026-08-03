function out = rodgers2_deck(iv, opts)
%RODGERS2_DECK  Render one RODGERS2_SEQ variant to a MACOS .in deck.
%
%   out = RODGERS2_DECK(IV) writes the deck for variant IV (1..4) of
%   RODGERS2_SEQ and returns the geometry it computed on the way.
%
%   This is the TRANSCRIPTION renderer: it converts the CODE V surface
%   list to MACOS elements by hand, in one place, with every convention
%   named.  Nothing is parsed and nothing is fitted.
%
%   THE CONVERSION, surface by surface (PACKET.md section 2 is the audit):
%
%     CODE V                          MACOS
%     ------                          -----
%     SO (infinity proxy)             zSource = 1e22, collimated
%     dummy / "tilt" / "thru"         dropped -- flats with no decenter and
%                                     no aperture carry no ray effect
%     STO (50 mm ahead of M1)         ApStop = (0,0,-0.050) m; the chief ray
%                                     is back-projected through it, so the
%                                     0.5 mm stop-offset walk at 0.6 deg is
%                                     in the trace, not approximated away
%     S REFL + CON + K                Element=Reflector, Surface=Conic,
%                                     KrElt = -|R|, KcElt = K, psi=(0,0,-1)
%                                     -- convex is GEOMETRY (where the CoC
%                                     lands), never a radius sign
%     CIR HOL r  (M1)                 nObs=1, ObsType=Circle, ObsVec=(r,0,0)
%     DAR YDE/ADE (M2, M3; variant 4) VptElt=(0,YDE,z),
%                                     psiElt=(0,-sin ADE,-cos ADE)
%     "recenter" (no DAR)             a COORDINATE BREAK -- consumed here to
%                                     place the coldstop in global coords;
%                                     it is not an element
%     "coldstop" (DAR ADE)            Element=Reference, Surface=Flat, at the
%                                     recenter station, normal tilted by
%                                     (ADE_recenter + ADE_coldstop)
%     S 0 -1000 / SI / AFI -1000      CODE V's afocal evaluation plane; the
%                                     MACOS afocal metric is referenced at
%                                     the COLDSTOP, so this is emitted only
%                                     when 'with_si' is set (off by default)
%
%   THE RECENTER SIGN, and the check that makes it a decode.  The recenter
%   surface has no DAR, so its decenter+tilt persist: the coldstop sits at
%       Vpt = (0, YDE, z_M3) + t * zhat_rec,   zhat_rec = R_x(-ADE)*zhat
%   under the rodgers1 ADE decode (his ADE = -(our alpha)).  With the
%   OPPOSITE sign the same arithmetic puts the coldstop vertex ~222 mm off
%   the exit chief ray -- on a 33 mm exit beam, six pupil diameters away.
%   RODGERS2_DECK returns .chief_miss_mm, the distance from the emitted
%   coldstop vertex to the traced exit chief ray, so the decode is
%   MEASURED on this design and not merely inherited.  (Run the check with
%   'verify', true -- it costs one trace.)
%
%   Name-value:
%     'variant'     override the RODGERS2_SEQ struct (for what-if runs)
%     'file'        output path (default rodgers2_<name>.in beside this file)
%     'field'       [thx thy] BOX-RELATIVE field offset, rad (default [0 0])
%     'sampling'    nGridpts (default 41)
%     'coldstop'    'Reference' (default) | 'Return' | 'FocalPlane'
%                   USE THE DEFAULT.  Reference and FocalPlane behave
%                   identically here, but `Element= Return` REVERSES the ray
%                   directions at that surface, so any metric that builds
%                   its reference from the exit chief builds it backwards --
%                   the afocal rung 1 reads 4 mm instead of 359 nm.  The OPL
%                   itself is unchanged, which is why it hides from a
%                   piston-only check.  Measured in CALIB_AFOCAL_PROBE
%                   section 0; it also retires the plan's proposal to emit
%                   the afocal terminal as a flat Return.
%     'with_si'     append his SI plane 1000 mm past the coldstop (false)
%     'hole'        override the M1 hole SEMI-diameter, m (NaN = his 0.130;
%                   0 removes it)
%     'stand'       source standoff along the chief, m (default 8)
%     'verify'      trace and measure .chief_miss_mm / .exit_dir (false)
%     'quiet'       (true)
%
%   Returns OUT with .file .txt .nElt .coldstop (Vpt, psi, ADE_total_deg)
%   .M3_z_m .elt (the emitted element table) and, with 'verify',
%   .chief_miss_mm .exit_dir .exit_beam_mm .mag.
%
%   See also RODGERS2_SEQ, AFOCAL_LADDER_DECK, PUPIL_MAP.

    arguments
        iv (1,1) double {mustBeInteger, mustBePositive}
        opts.variant   struct = struct([])
        opts.file      (1,:) char = ''
        opts.field     (1,2) double = [0 0]
        opts.sampling  (1,1) double {mustBeInteger,mustBePositive} = 41
        opts.coldstop  (1,:) char {mustBeMember(opts.coldstop, ...
                        {'Reference','Return','FocalPlane'})} = 'Reference'
        opts.with_si   (1,1) logical = false
        opts.hole      (1,1) double = NaN
        opts.stand     (1,1) double = 8.0
        opts.recenter_sign (1,1) double {mustBeMember(opts.recenter_sign,[-1 1])} = 1
        opts.verify    (1,1) logical = false
        opts.quiet     (1,1) logical = true
    end

    here = fileparts(mfilename('fullpath'));
    S = rodgers2_seq();
    if isempty(fieldnames(opts.variant)), V = S.v(iv); else, V = opts.variant; end

    mm = 1e-3;                              % his DIM mm -> our BaseUnits m
    hole_m = S.M1_hole_semi_mm*mm;
    if ~isnan(opts.hole), hole_m = opts.hole; end

    % ---- stations, M1 vertex at z = 0 -----------------------------------
    z_M1 = 0;
    z_M2 = S.s_M1_M2_mm;                                    % negative
    z_M3 = z_M2 + S.s_M2_thru_mm + S.s_thru_img_mm + V.s_img_to_M3_mm;

    % ---- rigid body (variant 4 only), VERBATIM CODE V -------------------
    yde = [0 0 0];  ade = [0 0 0];
    if ~isempty(V.rb)
        for r = 1:size(V.rb,1)
            ie = V.rb(r,1);  yde(ie) = V.rb(r,2);  ade(ie) = V.rb(r,3);
        end
    end

    % ---- the recenter coordinate break, consumed ------------------------
    % 'recenter_sign' = -1 emits the OPPOSITE ADE sense; it exists only so
    % the audit can quote what the rejected convention costs, and is never
    % used for a result.
    sg   = opts.recenter_sign;
    Arec = sg * V.recenter.ADE_deg;
    zrec = [0, sind(Arec), cosd(Arec)];       % R_x(-ADE)*zhat  (the decode)
    cs_vpt_mm = [0, V.recenter.YDE_mm, z_M3] + V.recenter.t_mm * zrec;
    Atot = Arec + sg*V.coldstop_ADE_deg;      % DAR adds in the recenter frame
    zcs  = [0, sind(Atot), cosd(Atot)];
    cs_psi = -zcs;                            % MACOS flat-normal convention
                                              % (rodgers1's FocalPlane sign)

    % ---- element table ---------------------------------------------------
    E = struct('name',{},'kind',{},'surface',{},'Kr',{},'Kc',{}, ...
               'psi',{},'Vpt',{},'obs',{});
    E(1) = mkelt('M1','Reflector','Conic', -abs(V.ROC_mm(1))*mm, V.K(1), ...
                 [0,-sind(ade(1)),-cosd(ade(1))], [0, yde(1)*mm, z_M1*mm], hole_m);
    E(2) = mkelt('M2','Reflector','Conic', -abs(V.ROC_mm(2))*mm, V.K(2), ...
                 [0,-sind(ade(2)),-cosd(ade(2))], [0, yde(2)*mm, z_M2*mm], 0);
    E(3) = mkelt('M3','Reflector','Conic', -abs(V.ROC_mm(3))*mm, V.K(3), ...
                 [0,-sind(ade(3)),-cosd(ade(3))], [0, yde(3)*mm, z_M3*mm], 0);
    E(4) = mkelt('ColdStop', opts.coldstop, 'Flat', -1e22, 0, ...
                 cs_psi, cs_vpt_mm*mm, 0);
    if opts.with_si
        % his last two surfaces: a flat at the coldstop station on the
        % RECENTER axis (the DAR tilt has reverted), then -1000 mm to SI.
        si_vpt_mm = cs_vpt_mm + (-1000.0)*zrec;
        E(5) = mkelt('SI','Reference','Flat', -1e22, 0, -zrec, si_vpt_mm*mm, 0);
    end

    % ---- source ----------------------------------------------------------
    bx = opts.field(1);
    by = deg2rad(V.YAN_abs_deg) + opts.field(2);
    cdir = [sin(bx), sin(by), sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    apst = [0, 0, -S.stop_ahead_of_M1_mm*mm];
    cpos = apst - opts.stand*cdir;
    ygrid = [0, cos(by), -sin(by)];

    txt = emit_(S, V, E, cdir, cpos, apst, ygrid, opts.sampling);

    file = opts.file;
    if isempty(file), file = fullfile(here, sprintf('rodgers2_%s.in', V.name)); end
    fid = fopen(file,'w');  fprintf(fid,'%s',txt);  fclose(fid);

    out = struct('file',file, 'txt',txt, 'nElt',numel(E), 'elt',E, ...
                 'variant',V, 'M3_z_m',z_M3*mm, ...
                 'coldstop',struct('Vpt',cs_vpt_mm*mm, 'psi',cs_psi, ...
                                   'ADE_total_deg',Atot, 'zhat_rec',zrec), ...
                 'ApStop',apst, 'stand',opts.stand, 'lambda',S.lambda_nm*1e-9);

    if opts.verify
        out = verify_(out, S);
        if ~opts.quiet
            fprintf(['  %-13s coldstop vertex-to-chief %8.3f mm  ' ...
                     'exit beam %7.3f mm  M = %8.4fx\n'], ...
                    V.name, out.chief_miss_mm, out.exit_beam_mm, out.mag);
        end
    end
end

% =====================================================================
function e = mkelt(name, kind, surface, Kr, Kc, psi, Vpt, obs)
    e = struct('name',name,'kind',kind,'surface',surface, ...
               'Kr',Kr,'Kc',Kc,'psi',psi(:).','Vpt',Vpt(:).','obs',obs);
end

function txt = emit_(S, V, E, cdir, cpos, apst, ygrid, npts)
    v3 = @(a) sprintf('%.16E  %.16E  %.16E', a(1), a(2), a(3));
    v6 = @(u,w) sprintf('%.16E  %.16E  %.16E  %.16E  %.16E  %.16E', ...
                        u(1),u(2),u(3),w(1),w(2),w(3));
    L = {};
    L{end+1} = sprintf('%% MACOS prescription -- Rodgers2 %s (%s)', V.name, V.file);
    L{end+1} = sprintf('%% verbatim transcription of "%s"', V.title);
    L{end+1} = '% emitted by design/rodgers2/rodgers2_deck.m -- do not hand-edit';
    L{end+1} = '% Source Definition';
    L{end+1} = ['        ChfRayDir=  ' v3(cdir)];
    L{end+1} = ['        ChfRayPos=  ' v3(cpos)];
    L{end+1} = '          zSource=1.0E+22';
    L{end+1} = '        BaseUnits=  m';
    L{end+1} = '        WaveUnits=  m';
    L{end+1} = '           IndRef=1.0E+00';
    L{end+1} = '           Extinc=0.0E+00';
    L{end+1} = sprintf('          Wavelen=%.16E', S.lambda_nm*1e-9);
    L{end+1} = '             Flux=1.0E+00';
    L{end+1} = sprintf('         Aperture=%.16E', S.EPD_mm*1e-3);
    L{end+1} = '         Obscratn=0.0E+00';
    L{end+1} = ['           ApStop=  ' v3(apst)];
    L{end+1} = '         GridType=  Circular';
    L{end+1} = sprintf('         nGridpts=  %d', npts);
    L{end+1} = ['            xGrid=  ' v3([1 0 0])];
    L{end+1} = ['            yGrid=  ' v3(ygrid)];
    L{end+1} = '% Element Definitions';
    L{end+1} = sprintf('             nElt=  %d', numel(E));
    for k = 1:numel(E)
        e = E(k);
        R = frame_(e.psi);
        L{end+1} = sprintf('             iElt=  %d', k);                    %#ok<AGROW>
        L{end+1} = ['          EltName=  ' e.name];                         %#ok<AGROW>
        L{end+1} = ['          Element=  ' e.kind];                         %#ok<AGROW>
        L{end+1} = ['          Surface=  ' e.surface];                      %#ok<AGROW>
        L{end+1} = sprintf('            KrElt=%.16E', e.Kr);                %#ok<AGROW>
        L{end+1} = sprintf('            KcElt=%.16E', e.Kc);                %#ok<AGROW>
        L{end+1} = ['           psiElt=  ' v3(e.psi)];                      %#ok<AGROW>
        L{end+1} = ['           VptElt=  ' v3(e.Vpt)];                      %#ok<AGROW>
        L{end+1} = ['           RptElt=  ' v3(e.Vpt)];                      %#ok<AGROW>
        L{end+1} = '           IndRef=1.0E+00';                             %#ok<AGROW>
        L{end+1} = '           Extinc=0.0E+00';                             %#ok<AGROW>
        L{end+1} = '            nCoat=  0';                                 %#ok<AGROW>
        if e.obs > 0
            L{end+1} = '             nObs=  1';                             %#ok<AGROW>
            L{end+1} = '          ObsType=  Circle';                        %#ok<AGROW>
            L{end+1} = ['           ObsVec=  ' v3([e.obs 0 0])];            %#ok<AGROW>
        else
            L{end+1} = '             nObs=  0';                             %#ok<AGROW>
        end
        % ApType=None throughout: the system stop is the source aperture
        % (CODE V's CA APE + EPD), and a vertex-centred circle would clip
        % the biased beam.  His CIR EDG 0.1 is a drawing datum.
        L{end+1} = '           ApType=  None';                              %#ok<AGROW>
        L{end+1} = '         PropType=  Geometric';                         %#ok<AGROW>
        L{end+1} = '             zElt=1.0E+20';                             %#ok<AGROW>
        L{end+1} = '          nECoord=  6';                                 %#ok<AGROW>
        L{end+1} = ['             TElt=  ' v6(R(:,1),[0;0;0])];             %#ok<AGROW>
        L{end+1} = ['                    ' v6(R(:,2),[0;0;0])];             %#ok<AGROW>
        L{end+1} = ['                    ' v6(R(:,3),[0;0;0])];             %#ok<AGROW>
        L{end+1} = ['                    ' v6([0;0;0],R(:,1))];             %#ok<AGROW>
        L{end+1} = ['                    ' v6([0;0;0],R(:,2))];             %#ok<AGROW>
        L{end+1} = ['                    ' v6([0;0;0],R(:,3))];             %#ok<AGROW>
    end
    L{end+1} = '% Output Coordinate System Definition';
    L{end+1} = '         nOutCord=  5';
    L{end+1} = ['             Tout=  ' v3([1 0 0]) '  ' v3([0 0 0]) '  0.0E+00'];
    L{end+1} = ['                    ' v3([0 1 0]) '  ' v3([0 0 0]) '  0.0E+00'];
    L{end+1} = ['                    ' v3([0 0 0]) '  ' v3([1 0 0]) '  0.0E+00'];
    L{end+1} = ['                    ' v3([0 0 0]) '  ' v3([0 1 0]) '  0.0E+00'];
    L{end+1} = ['                    ' v3([0 0 0]) '  ' v3([0 0 0]) '  1.0E+00'];
    txt = [strjoin(L, newline) newline];
end

function R = frame_(psi)
%FRAME_  Element TElt frame: z along psi, x/y tangent.  Trace-neutral.
%   Same construction as Telescope/surf_frame_ so the two agree.
    z = psi(:)/norm(psi);
    yhat = [0;1;0];  if abs(z(2)) > 0.95, yhat = [1;0;0]; end
    y = yhat - (yhat.'*z)*z;  y = y/norm(y);
    x = cross(y, z);
    R = [x, y, z];
end

% =====================================================================
function out = verify_(out, S)
%VERIFY_  Trace the emitted deck at the box centre and measure:
%   (1) the distance from the coldstop VERTEX to the exit CHIEF RAY --
%       the recenter-sign decode witness;
%   (2) the exit beam diameter and the traced angular magnification --
%       the first-order gate (M = 30 by design; Mike's slides report the
%       unoptimised offset variant slipping to 28.7x).
    macos.init(256);
    macos.load_rx(out.file);
    if ~macos.has_rx()
        error('rodgers2_deck:load','deck failed to load: %s', out.file);
    end
    nE = macos.num_elt();
    tr = macos.trace(3);                      % rays leaving M3
    r3 = macos.get_ray_info(tr.nRays);
    p1 = r3.pos(:,1);  d1 = r3.dir(:,1)/norm(r3.dir(:,1));
    w  = out.coldstop.Vpt(:) - p1;
    out.chief_miss_mm = norm(w - d1*(d1.'*w)) * 1e3;
    out.exit_dir = d1(:).';

    tr = macos.trace(nE);
    rc = macos.get_ray_info(tr.nRays);
    ok = rc.ok_trace(:) & rc.ok_pass(:);
    P  = rc.pos(:,ok);
    n  = out.coldstop.psi(:)/norm(out.coldstop.psi);
    [a1,a2] = perp_(n);
    u = a1.'*(P - out.coldstop.Vpt(:));  v = a2.'*(P - out.coldstop.Vpt(:));
    out.exit_beam_mm = 2*max(hypot(u,v))*1e3;
    out.mag = S.EPD_mm / out.exit_beam_mm;
    out.nrays = nnz(ok);
    % traced ANGULAR magnification: exit-chief swing per unit field swing,
    % measured symmetrically about the box centre so the leading error is
    % O(dth^2) rather than O(dth) (the pupil distortion is real and would
    % otherwise bias a one-sided difference).
    dth = 1e-4;
    tmp = [tempname '.in'];
    c = onCleanup(@() delrm_(tmp)); %#ok<NASGU>
    dp = chief_at_field_(out, [0 +dth], tmp);
    dm = chief_at_field_(out, [0 -dth], tmp);
    out.mag_ang = angle_between_(dp, dm) / (2*dth);
    macos.load_rx(out.file);              % leave the engine on the nominal field
end

function d = chief_at_field_(out, fld, tmp)
%CHIEF_AT_FIELD_  Exit-chief direction (leaving M3) at a box-relative field.
    V = out.variant;
    bx = fld(1);  by = deg2rad(V.YAN_abs_deg) + fld(2);
    cdir = [sin(bx), sin(by), sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
    cpos = out.ApStop - out.stand*cdir;
    s = regexprep(out.txt, '(ChfRayDir=\s*)[^\n]*', ...
                  ['$1' sprintf('%.16E  %.16E  %.16E', cdir)]);
    s = regexprep(s, '(ChfRayPos=\s*)[^\n]*', ...
                  ['$1' sprintf('%.16E  %.16E  %.16E', cpos)]);
    s = regexprep(s, '(yGrid=\s*)[^\n]*', ...
                  ['$1' sprintf('%.16E  %.16E  %.16E', [0, cos(by), -sin(by)])]);
    fid = fopen(tmp,'w'); fprintf(fid,'%s',s); fclose(fid);
    macos.load_rx(tmp);
    tr = macos.trace(3);  r = macos.get_ray_info(tr.nRays);
    d = r.dir(:,1).'/norm(r.dir(:,1));
end

function a = angle_between_(d1, d2)
    d1 = d1(:)/norm(d1);  d2 = d2(:)/norm(d2);
    a = atan2(norm(cross(d1,d2)), dot(d1,d2));
end

function [a1,a2] = perp_(n)
    n = n(:)/norm(n);  t = [1;0;0];
    if abs(n.'*t) > 0.9, t = [0;1;0]; end
    a1 = t - (n.'*t)*n;  a1 = a1/norm(a1);  a2 = cross(n,a1);
end

function delrm_(p),  if exist(p,'file'), delete(p); end,  end
