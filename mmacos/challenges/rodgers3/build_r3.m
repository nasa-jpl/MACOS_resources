function D = build_r3(opts)
%BUILD_R3  Emit the five rodgers3 Stage-0 decks (r3_r1.in .. r3_r5.in)
%   from the machine-generated .seq truth (rodgers3_seq.m).  Pure text
%   emission -- no engine calls.  Every convention decision is explicit
%   and parameterized so a failed gate can be screened over a BOUNDED
%   sign set (NOTES_s0.md item 3 discipline) without touching the truth.
%
%   CONVENTIONS ENCODED (see NOTES_s0.md + r3_s0_report.txt):
%   * Global frame: metres, one global +z = initial axis, beam enters
%     travelling +z.  First dummy at z=0, so m1 sits at +664.9568 mm
%     (the NOTES item-2 chain).  psiElt (0,0,-1)-style for every
%     element (the rodgers1 emission precedent: KrElt = signed CODE V
%     radius in metres, verbatim).
%   * z chain from the signed CODE V thicknesses: m1 = th2+th3,
%     stop/m2 = m1 + th4, m3 = m2 + th6, SI = recenter chain below.
%   * Aspheres: CODE V ASP A,B,C multiply r^4,r^6,r^8 of SAG ALONG THE
%     LOCAL +z.  Engine truth (surfsub.F AsphSrf, verified 2026-08-19):
%     AsphCoef(i) multiplies h^(2i+2) of sag along +psiElt.  psiElt is
%     -local-z here, so AsphCoef = -A * unit factor.  Units: mm -> m
%     factors 1e9 / 1e15 / 1e21 for r^4 / r^6 / r^8.
%   * Zernike (r5): engine truth (surfsub.F FreeFormSrf line 2174):
%     Mon/Zernike sag is applied along +zMon (NOT psi), polynomial
%     (x,y) from xMon/yMon/lMon.  So the deck carries the CODE V local
%     surface frame directly (zMon = local +z) and the SCO coefficients
%     go in VERBATIM (mm -> m, *1e-3).  ZernType= BornWolf ==
%     CODE V standard Zernike term-for-term; modes = zrn_C_idx - 1
%     (C1 = NRADIUS slot); C2 -> mode 1 = PISTON, carried.
%     lMon = NRADIUS * 1e-3.
%   * Parity map (NOTES item 3, screenable): CODE V (YDE, ADE) after an
%     ODD number of reflections -> (-YDE, -ADE) in the macos frame,
%     EVEN -> unchanged, where "macos ADE" alpha means
%     psi = (0, sind(alpha), -cosd(alpha))  [rigid_of convention].
%     rodgers3 parity: m1 EVEN, stop ODD, m2 ODD, m3 EVEN,
%     recenter/SI ODD.
%   * Surface SHAPE data (R, K, ASP, ZRN + its frame) are in the fixed
%     running-axis frame -- radii signs do NOT alternate with
%     reflections, so neither do the asphere/Zernike terms.  Only the
%     PERTURBATION data (YDE/ADE) get the parity map.
%   * Fields: CODE V XAN/YAN are tangent-composed:
%     dir = [tand(XAN), tand(YAN), 1]/norm.
%
%   SIGN MAP -- MEASURED (probe_cfg.m, 8-config screen on r2), and it
%   REFUTES the NOTES item-3 odd/even reflection-parity rule:
%     * ALL YDE VERBATIM (one global frame, no reflection parity,
%       stop plane included),
%     * ALL ADE sense-flipped uniformly (alpha_macos = -ADE_codev in
%       the alpha = atan2d(psi_y,-psi_z) convention)
%       == the rodgers1 empirical decode (convention_decode.m winner).
%   Evidence: of the 8 {stop, recY, recA} sign combinations, only the
%   all-verbatim-YDE + flipped-ADE one lands the centre-field chief on
%   the SI origin to 0.1 um AND reproduces his map's dynamic range
%   (rows 12/22/32 = 2.81/2.56/7.38 waves vs his min 1.13 / avg 2.57 /
%   max 8.81).  A second origin-admissible combination exists
%   ({stop flipped, rec verbatim}, miss 0.017 mm) but produces a FLAT
%   map (3.1..4.0 waves) -- the map shape arbitrates.  The notes' map
%   ({stop -, rec -}) missed the origin by 219.6 mm = 2x recenter YDE.
%
%   Name-value (bounded-screen hooks):
%     'sgn_stop' +1   stop-plane YDE multiplier (MEASURED verbatim)
%     'sgn_yde'  +1   mirror + recenter YDE multiplier
%     'sgn_ade'  -1   mirror + recenter + SI ADE multiplier
%     'sgn_m'    [+1 +1 +1]  extra per-mirror (YDE,ADE) multiplier, r4/r5
%                screens only
%     'zrn_flip_m2'  false  180-deg-about-x flip of m2's Zernike frame
%                    (the "shape data flips too" hypothesis; screen only)
%     'suffix'    ''   filename suffix (screen artifacts don't clobber)
%
%   Returns D: per-rung struct with the emitted geometry (for the
%   layout gate in run_r3_s0.m) and the deck path.

    arguments
        opts.sgn_stop (1,1) double = +1
        opts.sgn_yde  (1,1) double = +1
        opts.sgn_ade  (1,1) double = -1
        opts.sgn_m    (1,3) double = [+1 +1 +1]
        opts.zrn_flip_m2 (1,1) logical = false
        opts.suffix   (1,:) char = ''
    end
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    S = rodgers3_seq();
    mm = 1e-3;

    rungs = {'r1','r2','r3','r4','r5'};
    D = struct();
    for k = 1:5
        rk = rungs{k};
        R  = S.(rk);
        s  = R.s;

        % ---- z chain (m), NOTES item-2 --------------------------------
        z_m1   = (s(2).th + s(3).th)*mm;        % m1 vertex plane
        z_stop = z_m1 + s(4).th*mm;             % stop plane == m2 plane
        z_m2   = z_stop + s(5).th*mm;           % th5 = 0
        z_m3   = z_m2 + s(6).th*mm;             % m3 (th7 = 0 to recenter)

        % ---- stop centre (parity ODD) ----------------------------------
        y_stop = opts.sgn_stop * getf_(s(5),'YDE',0)*mm;
        stopC  = [0; y_stop; z_stop];

        % ---- mirrors ----------------------------------------------------
        iSrf  = [4 6 7];                        % .seq surface index of m1,m2,m3
        zEl   = [z_m1, z_m2, z_m3];
        M = struct('name',{},'Vpt',{},'psi',{},'Kr',{},'Kc',{}, ...
                   'asph',{},'zern',{},'frame',{});
        for m = 1:3
            ss = s(iSrf(m));
            dy = opts.sgn_m(m)*opts.sgn_yde*getf_(ss,'YDE',0)*mm;
            al = opts.sgn_m(m)*opts.sgn_ade*getf_(ss,'ADE',0);  % deg, macos alpha
            V  = [0; dy; zEl(m)];
            psi = [0;  sind(al); -cosd(al)];
            zp  = [0; -sind(al);  cosd(al)];            % local +z (CODE V)
            yp  = [0;  cosd(al);  sind(al)];            % local +y
            xp  = [1; 0; 0];
            fr  = struct('x',xp,'y',yp,'z',zp);
            E = struct('name',sprintf('M%d',m), 'Vpt',V, 'psi',psi, ...
                       'Kr',ss.R*mm, 'Kc',ss.K, 'asph',[], 'zern',[], ...
                       'frame',fr);
            if isfield(ss,'asph_ABCD') && any(ss.asph_ABCD(1:3) ~= 0)
                % engine: sag along +psi = -localz -> negate; mm->m units
                E.asph = -ss.asph_ABCD(1:3) .* [1e9 1e15 1e21];
            end
            if isfield(ss,'sps') && strcmp(ss.sps,'ZRN')
                zr = struct();
                zr.lMon  = ss.nradius*mm;
                zr.modes = ss.zrn_C_idx - 1;             % C(k+1) = mode k
                zr.coef  = ss.zrn_C_val*mm;              % VERBATIM, mm->m
                if m == 2 && opts.zrn_flip_m2            % screen hook only
                    E.frame.y = -E.frame.y;  E.frame.z = -E.frame.z;
                    zr.coef = -zr.coef;      % sag now along -localz
                end
                E.zern = zr;
            end
            M(m) = E;
        end

        % ---- recenter break -> SI (YDE verbatim, ADE sense-flipped) -----
        dy_r = opts.sgn_yde*getf_(s(8),'YDE',0)*mm;
        al_r = opts.sgn_ade*getf_(s(8),'ADE',0);        % deg
        al_i = opts.sgn_ade*getf_(s(9),'ADE',0);        % SI own DAR tilt
        zr_ax = [0; -sind(al_r); cosd(al_r)];           % tilted axis +z
        recO  = [0; dy_r; z_m3];
        V_si  = recO + s(8).th*mm*zr_ax;                % th < 0: back along axis
        al_t  = al_r + al_i;                            % both about global x
        psi_si= [0; sind(al_t); -cosd(al_t)];

        % ---- source (centre field, crude seed; run_r3_s0 re-aims) -------
        yanC = R.YAN(2);                                % centre of the box
        cdir = [tand(0); tand(yanC); 1];  cdir = cdir/norm(cdir);
        cdR  = [cdir(1); cdir(2); -cdir(3)];            % post-flat-m1 approx
        tq   = (z_m1 - z_stop)/cdir(3);
        q    = stopC - tq*cdR;                          % approx m1 hit point
        sback= 0.75/cdir(3);
        cpos = q - sback*cdir;

        % ---- gate data ---------------------------------------------------
        G = struct('rk',rk, 'file',R.file, 'title',R.title, ...
                   'z_m1',z_m1, 'z_stop',z_stop, 'z_m3',z_m3, ...
                   'stopC',stopC, 'M',M, 'V_si',V_si, 'psi_si',psi_si, ...
                   'recO',recO, 'al_rec',al_r, 'al_si',al_i, ...
                   'EPD_m',R.EPD_mm*mm, 'WL_m',R.WL_nm*1e-9, ...
                   'XAN',R.XAN, 'YAN',R.YAN, 'cdir',cdir, 'cpos',cpos, ...
                   'sgn',opts);

        deck = fullfile(here, sprintf('r3_%s%s.in', rk, opts.suffix));
        emit_deck_(deck, G);
        G.deck = deck;
        D.(rk) = G;
        fprintf('built %s  (%s)\n', deck, R.title);
    end
    save(fullfile(here, sprintf('r3_build%s.mat',opts.suffix)), 'D');
end

% =====================================================================
function v = getf_(s, f, d)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function emit_deck_(path, G)
    f = fopen(path,'w');
    c = onCleanup(@() fclose(f));
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));

    fprintf(f,'%% MACOS prescription -- rodgers3 Stage 0, %s\n', G.rk);
    fprintf(f,'%% emitted by build_r3.m from %s (%s)\n', G.file, G.title);
    fprintf(f,'%% Source Definition\n');
    fprintf(f,'        ChfRayDir=  %s\n', v3(G.cdir));
    fprintf(f,'        ChfRayPos=  %s\n', v3(G.cpos));
    fprintf(f,'          zSource=1.0E+22\n');
    fprintf(f,'        BaseUnits=  m\n');
    fprintf(f,'        WaveUnits=  m\n');
    fprintf(f,'           IndRef=1.0E+00\n');
    fprintf(f,'           Extinc=0.0E+00\n');
    fprintf(f,'          Wavelen=%.16E\n', G.WL_m);
    fprintf(f,'             Flux=1.0E+00\n');
    fprintf(f,'         Aperture=%.16E\n', G.EPD_m);
    fprintf(f,'         Obscratn=0.0E+00\n');
    fprintf(f,'%% NOTE: no ApStop= line ON PURPOSE.  The engine ApStop triggers\n');
    fprintf(f,'%% straight-line source aiming through StopPos (srcaim.inc), wrong for\n');
    fprintf(f,'%% this design (stop is on the m2 plane, AFTER one reflection).  The\n');
    fprintf(f,'%% physical stop centre is [%s] (m);\n', v3(G.stopC));
    fprintf(f,'%% run_r3_s0.m aims each field chief through it by real-ray iteration.\n');
    fprintf(f,'         GridType=  Circular\n');
    fprintf(f,'         nGridpts=  41\n');
    fprintf(f,'            xGrid=  %s\n', v3([1;0;0]));
    fprintf(f,'            yGrid=  %s\n', v3([0;1;0]));
    fprintf(f,'%% Element Definitions\n');
    fprintf(f,'             nElt=  4\n');

    zNext = [abs(G.M(2).Vpt(3)-G.M(1).Vpt(3)), ...
             abs(G.M(3).Vpt(3)-G.M(2).Vpt(3)), ...
             norm(G.V_si-G.M(3).Vpt), 1e20];
    for m = 1:3
        E = G.M(m);
        fprintf(f,'             iElt=  %d\n', m);
        fprintf(f,'          EltName=  %s\n', E.name);
        fprintf(f,'          Element=  Reflector\n');
        if ~isempty(E.zern)
            fprintf(f,'          Surface=  Zernike\n');
        elseif ~isempty(E.asph)
            fprintf(f,'          Surface=  Aspheric\n');
        else
            fprintf(f,'          Surface=  Conic\n');
        end
        fprintf(f,'            KrElt=%.16E\n', E.Kr);
        fprintf(f,'            KcElt=%.16E\n', E.Kc);
        if ~isempty(E.asph)
            fprintf(f,'       nAsphCoefs=  3\n');
            fprintf(f,'        AsphCoefs=  %.16E %.16E %.16E\n', E.asph);
        end
        if ~isempty(E.zern)
            zr = E.zern;  n = numel(zr.modes);
            fprintf(f,'         ZernType=  BornWolf\n');
            fprintf(f,'             lMon=%.16E\n', zr.lMon);
            fprintf(f,'             pMon=  %s\n', v3(E.Vpt));
            fprintf(f,'             xMon=  %s\n', v3(E.frame.x));
            fprintf(f,'             yMon=  %s\n', v3(E.frame.y));
            fprintf(f,'             zMon=  %s\n', v3(E.frame.z));
            fprintf(f,'        nZernCoef=  %d\n', n);
            fprintf(f,'        ZernModes= %s\n', sprintf(' %d', zr.modes));
            % ZernCoef wraps at 6 per line (msmacosio.inc Grp=6)
            fprintf(f,'         ZernCoef= %s\n', ...
                    sprintf(' %.16E', zr.coef(1:min(6,n))));
            i = 7;
            while i <= n
                j = min(i+5, n);
                fprintf(f,'                   %s\n', ...
                        sprintf(' %.16E', zr.coef(i:j)));
                i = j + 1;
            end
        end
        fprintf(f,'           psiElt=  %s\n', v3(E.psi));
        fprintf(f,'           VptElt=  %s\n', v3(E.Vpt));
        fprintf(f,'           RptElt=  %s\n', v3(E.Vpt));
        fprintf(f,'           IndRef=1.0E+00\n');
        fprintf(f,'           Extinc=0.0E+00\n');
        fprintf(f,'             nObs=  0\n');
        fprintf(f,'           ApType=  None\n');
        fprintf(f,'         PropType=  Geometric\n');
        fprintf(f,'             zElt=%.16E\n', zNext(m));
    end
    fprintf(f,'             iElt=  4\n');
    fprintf(f,'          EltName=  FP\n');
    fprintf(f,'          Element=  FocalPlane\n');
    fprintf(f,'          Surface=  Flat\n');
    fprintf(f,'            KrElt=-1.0000000000000000E+22\n');
    fprintf(f,'            KcElt=0.0000000000000000E+00\n');
    fprintf(f,'           psiElt=  %s\n', v3(G.psi_si));
    fprintf(f,'           VptElt=  %s\n', v3(G.V_si));
    fprintf(f,'           RptElt=  %s\n', v3(G.V_si));
    fprintf(f,'           IndRef=1.0E+00\n');
    fprintf(f,'           Extinc=0.0E+00\n');
    fprintf(f,'             nObs=  0\n');
    fprintf(f,'           ApType=  None\n');
    fprintf(f,'         PropType=  Geometric\n');
    fprintf(f,'             zElt=%.16E\n', zNext(4));
    fprintf(f,'%% Output Coordinate System Definition\n');
    fprintf(f,'         nOutCord=  5\n');
    T = [1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 0 1 0 0 0; ...
         0 0 0 0 1 0 0; 0 0 0 0 0 0 1];
    fprintf(f,'             Tout=');
    for r = 1:5
        if r > 1, fprintf(f,'                  '); end
        fprintf(f,'  %s\n', strtrim(sprintf('%.16E  ', T(r,:))));
    end
end
