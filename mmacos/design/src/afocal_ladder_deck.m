function [L, info] = afocal_ladder_deck(deck, Frel, opts)
%AFOCAL_LADDER_DECK  The 3-rung afocal reference ladder on a COMMITTED deck.
%
%   L = AFOCAL_LADDER_DECK(DECK, FREL) scores DECK over the BOX-RELATIVE
%   field offsets FREL (K x 2, rad) and returns L (K x 3) in METRES -- one
%   column per rung of AFOCAL_RUNGS:
%
%       1 piston-only (chief-normal plane)   2 + tip/tilt   3 + power
%
%   Deck-path sibling of AFOCAL_WFE_DECK, which returns ONE rung; this
%   returns all three so a result can be BRACKETED rather than quoted
%   against a single convention.  USE IT FOR EXTERNAL COMPARISONS: a CODE V
%   afocal field map removes SOME reference freedom and the vendors do not
%   agree on which, so the honest comparison names the rung (rodgers1
%   Addendum 8.7: on comatic fields the focal rungs spread by 1.3-1.7x and
%   the tilt treatment was the whole game).
%
%   SCORE ON A UNIFORM GRID, NOT ON THE SOLVE SET.  An edge-weighted solve
%   sampling biased the rodgers1 AVERAGE by ~8% at an identical max.
%
%   Name-value:
%     'ref_elt'    element anchoring the reference plane and at which the
%                  rays are read (default: the last element)
%     'anchor'     explicit 3-vector anchor overriding the element vertex
%     'strip_ap'   true (DEFAULT) rewrites every `ApType=` to `None`
%     'min_rays'   fields with fewer good rays stay NaN (10)
%     'init'       call macos.init(model_size) first (default false -- the
%                  caller usually owns the session)
%     'model_size' model size for that init (256)
%     'lambda'     design wavelength, m.  When given (>0) INFO also carries
%                  .strehl (K x 3), the EXACT aperture form
%                  |mean(exp(i*2*pi*W/lambda))|^2 evaluated on each rung's
%                  residual wavefront -- always <= 1, unlike a psf-peak
%                  ratio, and computed from W rather than inferred from its
%                  RMS through Marechal.
%
%   [L, INFO] = ... also returns per-field diagnostics: the exit chief ray,
%   the anchor, the removed boresight (.tilt_urad) and residual divergence
%   (.divergence_urad, .R_curv_m), the AFOCAL_REFS cross-check
%   (.bore_split_urad), the pupil radius, the ray count, and the deck's own
%   bias in degrees.
%
%   See also AFOCAL_RUNGS, AFOCAL_REFS, AFOCAL_WFE_DECK, FIELD_GRID.

    arguments
        deck (1,:) char
        Frel (:,2) double
        opts.ref_elt    (1,1) double  = 0
        opts.anchor     (1,3) double  = [NaN NaN NaN]
        opts.strip_ap   (1,1) logical = true
        opts.min_rays   (1,1) double  = 10
        opts.init       (1,1) logical = false
        opts.model_size (1,1) double  = 256
        opts.lambda     (1,1) double  = 0
    end

    if opts.init, macos.init(opts.model_size); end

    txt = fileread(deck);
    if opts.strip_ap
        txt = regexprep(txt, '(ApType=\s*)\S+', '$1None');
    end
    [cdir0, cpos0, apst] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));   by0 = asin(cdir0(2));
    Vs = grab_all3_(txt, 'VptElt');
    ie = opts.ref_elt;   if ie <= 0, ie = size(Vs,2); end
    anch = Vs(:,ie);
    if all(isfinite(opts.anchor)), anch = opts.anchor(:); end

    K = size(Frel,1);
    L = nan(K,3);
    info = struct('deck',deck, 'fields',Frel, 'bias_deg',[bx0 by0]*180/pi, ...
                  'ref_elt',ie, 'anchor',anch.', 'lambda',opts.lambda, ...
                  'strehl',nan(K,3), 'nrays',zeros(K,1), ...
                  'chief_dir',nan(3,K), 'chief_pos',nan(3,K), ...
                  'tilt_urad',nan(K,1), 'divergence_urad',nan(K,1), ...
                  'R_curv_m',nan(K,1), 'power_sag_nm',nan(K,1), ...
                  'bore_split_urad',nan(K,1), 'rho_max',nan(K,1));

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    for k = 1:K
        ri = trace_field_(txt, tmp, apst, stand, ...
                          bx0+Frel(k,1), by0+Frel(k,2), ie);
        ok = ri.ok_trace(:) & ri.ok_pass(:);
        ok(1) = false;
        if nnz(ok) < opts.min_rays, continue; end
        p1 = ri.pos(:,1);   d1 = ri.dir(:,1);
        [L(k,:), Wr, f] = afocal_rungs(ri.pos(:,ok), ri.dir(:,ok), ...
                                       ri.opl(ok), p1, d1, anch);
        info.nrays(k)           = nnz(ok);
        info.chief_dir(:,k)     = d1/norm(d1);
        info.chief_pos(:,k)     = p1;
        info.tilt_urad(k)       = f.tilt_urad;
        info.divergence_urad(k) = f.divergence_urad;
        info.R_curv_m(k)        = f.R_curv_m;
        info.power_sag_nm(k)    = f.power_sag_m*1e9;
        info.bore_split_urad(k)   = f.bore_split_urad;
        info.rho_max(k)         = f.rho_max;
        if opts.lambda > 0
            for r = 1:3
                info.strehl(k,r) = abs(mean(exp(2i*pi*Wr(:,r)/opts.lambda)))^2;
            end
        end
    end
end

% ---------------------------------------------------------------------
% Deck helpers.  Kept local (mirrors of AFOCAL_WFE_DECK's) so this scorer
% is self-contained; TAFOCALKERNEL gates that the two agree exactly.
% ---------------------------------------------------------------------
function ri = trace_field_(txt, tmp, apst, stand, bx, by, ie)
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
    s = regexprep(s,   '(yGrid=\s*)[^\n]*', ...
                  ['$1' v3_([0; cos(by); -sin(by)])]);
    fid = fopen(tmp,'w');   fprintf(fid,'%s',s);   fclose(fid);
    macos.load_rx(tmp);
    if ~macos.has_rx()
        error('macos:design:afocal_ladder_deck:load', ...
              'deck failed to load: %s', tmp);
    end
    tr = macos.trace(ie);
    ri = macos.get_ray_info(tr.nRays);
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end

function [cdir, cpos, apst] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');   cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function M = grab_all3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens');
    M = zeros(3,numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'),'%f',3); end
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
