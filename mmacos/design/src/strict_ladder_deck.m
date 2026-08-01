function [L, info] = strict_ladder_deck(deck, Frel, opts)
%STRICT_LADDER_DECK  The 4-rung reference ladder on a COMMITTED .in deck.
%
%   L = STRICT_LADDER_DECK(DECK, FREL) scores DECK over the BOX-RELATIVE
%   field offsets FREL (K x 2, rad) and returns L (K x 4) in METRES -- one
%   column per rung of STRICT_RUNGS:
%
%       1 strict-chief   2 strict-centroid   3 +best focus   4 +LS tip/tilt
%
%   Deck path sibling of STRICT_WFE_DECK, which returns ONE reference; this
%   returns all four so a result can be BRACKETED rather than quoted against
%   a single convention.  Same mechanics: the deck's OWN chief ray sets the
%   field bias (so FREL is an offset from wherever the deck already points),
%   the deck's OWN FocalPlane is the frozen detector, and the exit pupil is
%   found per field by the two-probe chief-ray crossing FEX uses.
%
%   USE IT FOR EXTERNAL COMPARISONS AND FOR HEADLINE STATISTICS.  Report the
%   rung a number came from: on comatic fields the four rungs spread by
%   1.3-1.7x, and per-field focus is worth <3% of that -- the tilt treatment
%   is the whole game (PACKET.md Addendum 8.7 / 9.2).
%
%   Name-value:
%     'probe'      probe field offset for the exit-pupil finder, rad (1e-5)
%     'strip_ap'   true (DEFAULT) rewrites every `ApType=` to `None`.  Decks
%                  saved after realize_apertures carry that call's clip
%                  apertures, sized to whatever box it was handed; scoring
%                  through them aperture-limits the metric with stale state.
%     'min_rays'   fields returning fewer good rays than this stay NaN (10)
%     'init'       call macos.init(model_size) first (default false -- the
%                  caller usually owns the session)
%     'model_size' model size for that init (256)
%     'lambda'     design wavelength, m.  When given (>0) INFO also carries
%                  .strehl (K x 4), the EXACT aperture form
%                  |mean(exp(i*2*pi*W/lambda))|^2 evaluated on each rung's
%                  residual wavefront -- always <= 1, unlike a psf-peak
%                  ratio, and computed from W rather than inferred from its
%                  RMS through Marechal.
%
%   [L, INFO] = ... also returns per-field diagnostics: the chief intercept,
%   the exit pupil, the ray count, the deck's own bias in degrees, and
%   (with 'lambda') the Strehl ratio per field per rung.
%
%   SCORE ON A UNIFORM GRID, NOT ON THE SOLVE SET.  An edge-weighted solve
%   sampling biases the AVERAGE ~8% at an identical max (rodgers1
%   DENSE_FIELD_CHECK); the max is insensitive because the worst field is a
%   box corner both sets contain.
%
%   See also STRICT_RUNGS, STRICT_REFS, STRICT_WFE_DECK, FIELD_GRID.

    arguments
        deck (1,:) char
        Frel (:,2) double
        opts.probe      (1,1) double  = 1e-5
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
    [Vd, Nd] = deck_fp_(txt);   Nd = Nd/norm(Nd);

    K = size(Frel,1);
    L = nan(K,4);
    info = struct('deck',deck, 'fields',Frel, 'bias_deg',[bx0 by0]*180/pi, ...
                  'detector',struct('Vpt',Vd.','psi',Nd.'), ...
                  'c',nan(3,K), 'xp',nan(3,K), 'nrays',zeros(K,1), ...
                  'lambda',opts.lambda, 'strehl',nan(K,4));

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    for k = 1:K
        ri = trace_field_(txt, tmp, apst, stand, bx0+Frel(k,1), by0+Frel(k,2));
        p1 = ri.pos(:,1);   d1 = ri.dir(:,1);

        % exit pupil: where this field's chief and a probe field's chief cross
        rp = trace_field_(txt, tmp, apst, stand, ...
                          bx0+Frel(k,1)+opts.probe, by0+Frel(k,2));
        X  = fex_cross_(p1, d1, rp.pos(:,1), rp.dir(:,1));

        % the probe trace replaced the engine state -- retrace this field
        ri = trace_field_(txt, tmp, apst, stand, bx0+Frel(k,1), by0+Frel(k,2));
        ok = ri.ok_trace(:) & ri.ok_pass(:);
        ok(1) = false;                       % the engine OPD skips the chief
        if nnz(ok) < opts.min_rays, continue; end

        den = dot(Nd, d1);
        if abs(den) > 1e-12
            info.c(:,k) = p1 + d1*(dot(Nd, Vd - p1)/den);
        end
        info.xp(:,k)  = X;
        info.nrays(k) = nnz(ok);
        [L(k,:), Wr] = strict_rungs(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), ...
                                    p1, d1, Vd, Nd, X);
        if opts.lambda > 0
            for r = 1:4
                info.strehl(k,r) = abs(mean(exp(2i*pi*Wr(:,r)/opts.lambda)))^2;
            end
        end
    end
end

% ---------------------------------------------------------------------
% Deck helpers.  Kept local (mirrors of STRICT_WFE_DECK's) so this scorer
% is self-contained and cannot be broken by a change there.
% ---------------------------------------------------------------------
function [cdir, cpos, apst] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');   cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
end

function [V, N] = deck_fp_(txt)
%DECK_FP_  Vpt/psi of the LAST element -- the FocalPlane.
    Vs = grabAll3_(txt,'VptElt');   Ps = grabAll3_(txt,'psiElt');
    V = Vs(:,end);   N = Ps(:,end);
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

function ri = trace_field_(txt, tmp, apst, stand, bx, by)
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
    fid = fopen(tmp,'w');   fprintf(fid,'%s',s);   fclose(fid);
    macos.load_rx(tmp);
    if ~macos.has_rx()
        error('macos:design:strict_ladder_deck:load', ...
              'deck failed to load: %s', tmp);
    end
    tr = macos.trace(macos.num_elt());
    ri = macos.get_ray_info(tr.nRays);
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);   d2 = d2/norm(d2);
    w0 = p1-p2;   b = dot(d1,d2);   den = 1-b^2;
    if abs(den) < 1e-14, X = p1; return; end
    X = 0.5*((p1 + d1*((b*dot(d2,w0)-dot(d1,w0))/den)) + ...
             (p2 + d2*((dot(d2,w0)-b*dot(d1,w0))/den)));
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
