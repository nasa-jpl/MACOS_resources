function out = strict_wfe_deck(deck, Frel, opts)
%STRICT_WFE_DECK  The strict WFE metric evaluated on a COMMITTED .in deck.
%
%   Sibling of STRICT_WFE for the case where the design to be scored is a
%   saved prescription rather than a live macos.design.Telescope -- i.e.
%   pure evaluation of an already-solved stage, with no rebuild and no
%   re-optimization.  Same metric, same kernel (STRICT_SPHERE_OPL), same
%   exit-pupil finder; the only difference is how the per-field chief ray
%   is produced.
%
%   out = STRICT_WFE_DECK(DECK, FREL) scores DECK over the BOX-RELATIVE
%   field offsets FREL (K x 2, rad).  The deck's OWN chief ray sets the
%   field bias, so FREL is offset from wherever the deck already points --
%   there is no bias to double (cf. STRICT_WFE, which goes through
%   trace_at_field and therefore takes box-relative offsets for the same
%   reason).
%
%   Per field the deck is re-emitted with ONLY its two source lines
%   rewritten, exactly as macos.design.Telescope/emit_ builds them
%   (Telescope.m:2438-2445):
%
%       cdir = [sin(bx), sin(by), sqrt(1 - sin(bx)^2 - sin(by)^2)]
%       cpos = ApStop - stand*cdir
%
%   with `bx, by` read back from the deck's own ChfRayDir and offset by
%   FREL, and `stand` recovered from the deck as dot(ApStop - ChfRayPos,
%   ChfRayDir).  Every other byte of the deck is untouched, so what is
%   scored is the committed solve.
%
%   The reference is the deck's OWN FocalPlane (its solved FPA), held
%   frozen: sphere centred on each field's chief-ray incidence point on
%   that plane, anchored at the exit pupil, piston-only removal.
%
%   Name-value:
%     'detector'   struct('Vpt',1x3,'psi',1x3) overriding the deck's FP.
%     'dz'         extra displacement along the detector normal (m).
%     'strip_ap'   true (DEFAULT) rewrites every `ApType=` to `None`.
%                  The committed rodgers1_epd4060_stage*.in decks were
%                  saved AFTER realize_apertures ran, so they carry that
%                  call's clip apertures, sized to the box it was handed.
%                  Scoring through them would aperture-limit the metric
%                  with stale state -- the contamination trap.  The
%                  optics, the FPA and the chief ray are untouched.
%     'reference'  'strict-chief' (DEFAULT) | 'strict-centroid'.  Which
%                  sphere centre lands in .wfe / .wfe_m.  BOTH are always
%                  computed and returned (.wfe_m_chief, .wfe_m_centroid), so
%                  nothing is lost either way.
%
%                  DEFAULT NOTE.  Dave's 2026-07-31 ruling makes the CENTROID
%                  the primary reference.  This function nevertheless still
%                  DEFAULTS to 'strict-chief', deliberately: flipping the
%                  default here would silently rewrite every committed
%                  EPD-2000/EPD-4060 artifact, which the same brief requires
%                  to stay bit-intact.  The ruling is applied where it belongs
%                  -- in the study drivers at the .seq truth, which pass
%                  'strict-centroid' explicitly.
%     'probe'      probe field for the exit-pupil finder, rad (def 1e-5).
%     'model_size' engine model size (default 256).
%     'quiet'      default true.
%
%   Returns the same struct shape as STRICT_WFE.
%
%   See also STRICT_WFE, STRICT_SPHERE_OPL, STRICT_REFS,
%   STRICT_LADDER_DECK, STRICT_RUNGS.

    arguments
        deck (1,:) char
        Frel (:,2) double
        opts.detector struct = struct([])
        opts.dz (1,1) double = 0
        opts.strip_ap (1,1) logical = true
        opts.probe (1,1) double = 1e-5
        opts.model_size (1,1) double = 256
        opts.quiet (1,1) logical = true
        opts.reference (1,:) char {mustBeMember(opts.reference, ...
            {'strict-chief','strict-centroid','chief','centroid'})} = 'strict-chief'
    end
    use_centroid = ~isempty(strfind(lower(opts.reference),'centroid')); %#ok<STREMP>

    txt = fileread(deck);
    if opts.strip_ap
        txt = regexprep(txt, '(ApType=\s*)\S+', '$1None');
    end
    [cdir0, cpos0, apst, lam] = deck_source_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));

    [Vd, Nd] = deck_fp_(txt);
    if ~isempty(fieldnames(opts.detector))
        Vd = opts.detector.Vpt(:);  Nd = opts.detector.psi(:);
    end
    Nd = Nd/norm(Nd);  Vd = Vd + Nd*opts.dz;

    K = size(Frel,1);
    out = struct('fields',Frel, 'wfe',nan(K,1), 'wfe_m',nan(K,1), ...
                 'nrays',zeros(K,1), 'c',nan(3,K), 'xp',nan(3,K), ...
                 'R',nan(K,1), 'eng_rms_m',nan(K,1), ...
                 'detector',struct('Vpt',Vd.','psi',Nd.'), 'dz',opts.dz, ...
                 'lambda',lam, 'deck',deck, ...
                 'bias_deg',[bx0 by0]*180/pi, 'strip_ap',opts.strip_ap, ...
                 'reference',opts.reference, ...
                 'wfe_m_chief',nan(K,1), 'wfe_m_centroid',nan(K,1), ...
                 'c_chief',nan(3,K), 'c_centroid',nan(3,K), ...
                 'dcen',nan(3,K), 'dcen_m',nan(K,1), 'dcen_2d',nan(2,K), ...
                 'dcen_err_m',nan(K,1), 'tilt_resid',nan(K,1), ...
                 'defoc_resid',nan(K,1), 'dcen_perp_m',nan(K,1), ...
                 'dcen_long_m',nan(K,1), 'diff_rms',nan(K,1));

    macos.init(opts.model_size);
    tmp = [tempname '.in'];
    cleanup = onCleanup(@() delete_if_(tmp)); %#ok<NASGU>

    for k = 1:K
        [ri, nE] = trace_field_(txt, tmp, apst, stand, ...
                                bx0 + Frel(k,1), by0 + Frel(k,2), out, k);
        if isempty(ri), continue; end
        out.eng_rms_m(k) = ri.eng;
        ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
        if nnz(ok) < 10, continue; end

        p1 = ri.pos(:,1);  d1 = ri.dir(:,1);
        den = dot(Nd, d1);
        if abs(den) < 1e-12, continue; end
        c = p1 + d1*(dot(Nd, Vd - p1)/den);
        out.c(:,k) = c;

        rp = trace_field_(txt, tmp, apst, stand, ...
                          bx0 + Frel(k,1) + opts.probe, by0 + Frel(k,2), out, k);
        X = fex_cross_(p1, d1, rp.pos(:,1), rp.dir(:,1));
        out.xp(:,k) = X;
        R = norm(X - c);   out.R(k) = R;

        rf = strict_refs(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), p1, d1, Vd, Nd, X);
        out.wfe_m_chief(k)    = rf.wfe_chief;
        out.wfe_m_centroid(k) = rf.wfe_centroid;
        out.c_chief(:,k)      = rf.c_chief;
        out.c_centroid(:,k)   = rf.c_centroid;
        out.dcen(:,k)         = rf.dcen;
        out.dcen_m(k)         = rf.dcen_m;
        out.dcen_2d(:,k)      = rf.centroid_2d;
        out.dcen_err_m(k)     = rf.dcen_err_m;
        out.tilt_resid(k)     = rf.tilt_resid;
        out.defoc_resid(k)    = rf.defoc_resid;
        out.dcen_perp_m(k)    = rf.dcen_perp_m;
        out.dcen_long_m(k)    = rf.dcen_long_m;
        out.diff_rms(k)       = rf.diff_rms;
        if use_centroid
            out.wfe_m(k) = rf.wfe_centroid;   out.c(:,k) = rf.c_centroid;
            out.R(k)     = rf.R_centroid;
        else
            out.wfe_m(k) = rf.wfe_chief;
        end
        out.wfe(k)   = out.wfe_m(k)/lam;
        out.nrays(k) = nnz(ok);
        if ~opts.quiet
            fprintf('   [%+7.3f %+7.3f]'' : %10.4g nm  (%d rays, nElt %d)\n', ...
                Frel(k,1)*180/pi*60, Frel(k,2)*180/pi*60, out.wfe_m(k)*1e9, nnz(ok), nE);
        end
    end
end

% ---------------------------------------------------------------------
function [ri, nE] = trace_field_(txt, tmp, apst, stand, bx, by, out, k) %#ok<INUSD>
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
    macos.load_rx(tmp);
    if ~macos.has_rx()
        error('macos:design:strict_wfe_deck:load', 'deck failed to load: %s', tmp);
    end
    nE = macos.num_elt();
    tr = macos.trace(nE);
    ri = macos.get_ray_info(tr.nRays);
    ri.eng = tr.rmsWFE;          % BaseUnits (m) -- NOT waves
end

function s = v3_(v)
    s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
end

function [cdir, cpos, apst, lam] = deck_source_(txt)
    cdir = grab3_(txt,'ChfRayDir');
    cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    t = regexp(txt,'Wavelen=\s*([-\d.EeD+]+)','tokens','once');
    lam = str2double(strrep(t{1},'D','E'));
end

function [V, N] = deck_fp_(txt)
%DECK_FP_  Vpt/psi of the LAST element -- the FocalPlane.
    Vs = grabAll3_(txt,'VptElt');   Ps = grabAll3_(txt,'psiElt');
    V = Vs(:,end);  N = Ps(:,end);
end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function M = grabAll3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens');
    M = zeros(3,numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'),'%f',3); end
end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function delete_if_(p)
    if exist(p,'file'), delete(p); end
end
