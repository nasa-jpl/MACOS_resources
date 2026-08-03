function out = afocal_wfe_deck(deck, Frel, opts)
%AFOCAL_WFE_DECK  The afocal WFE metric evaluated on a COMMITTED .in deck.
%
%   Plane-reference sibling of STRICT_WFE_DECK, for a system that delivers a
%   COLLIMATED beam instead of an image.  Same mechanics -- the deck's own
%   chief ray sets the field bias, so FREL is BOX-RELATIVE and there is no
%   bias to double; only the two source lines are rewritten per field, so
%   what is scored is the committed deck.
%
%   out = AFOCAL_WFE_DECK(DECK, FREL) scores DECK over the box-relative
%   field offsets FREL (K x 2, rad).
%
%   THE REFERENCE is the plane through the REFERENCE ELEMENT's vertex,
%   normal to that field's exit chief ray, with piston-only removal
%   (rung 1 of AFOCAL_RUNGS).  The reference element defaults to the LAST
%   element -- the coldstop / interface pupil.  Its own psi does NOT set the
%   reference orientation: a coldstop deliberately tilted several degrees
%   off the chief (Rodgers2 tilts it 0 to 4.3 deg) would inject
%   rho*tan(tilt) of pure tilt -- millimetres on a 33 mm beam -- and drown
%   the wavefront.  The plane's ORIENTATION is the chief; the element only
%   supplies the ANCHOR, and moving the anchor along the chief is pure
%   piston (AFOCAL_REFS).
%
%   Name-value:
%     'reference'  'chief' (DEFAULT, rung 1) | 'boresight' (rung 2, LS
%                  tip/tilt removed) | 'collimated' (rung 3, + power).
%                  ALL THREE are always computed and returned, so nothing
%                  is lost either way.
%     'ref_elt'    element whose vertex anchors the reference plane and at
%                  which the rays are read (default: the last element).
%     'anchor'     explicit 3-vector anchor overriding the element vertex.
%     'strip_ap'   true (DEFAULT) rewrites every `ApType=` to `None`.  A
%                  deck saved after realize_apertures carries that call's
%                  clip apertures, sized to whatever box it was handed;
%                  scoring through them aperture-limits the metric with
%                  stale state.
%     'min_rays'   fields returning fewer good rays stay NaN (10)
%     'model_size' engine model size for the init (256)
%     'init'       call macos.init first (default true)
%     'quiet'      (true)
%
%   Returns OUT with, per field: .wfe_m (the selected rung), .wfe (waves),
%   .wfe_m_chief/.wfe_m_boresight/.wfe_m_collimated (all three rungs),
%   .tilt_urad (the removed boresight), .divergence_urad and .R_curv_m
%   (the removed collimation error), .bore_split_urad (the AFOCAL_REFS
%   cross-check), .rho_max, .nrays, .anchor, .chief_dir, .bias_deg.
%
%   See also AFOCAL_LADDER_DECK, AFOCAL_RUNGS, AFOCAL_REFS,
%   STRICT_WFE_DECK.

    arguments
        deck (1,:) char
        Frel (:,2) double
        opts.reference (1,:) char {mustBeMember(opts.reference, ...
            {'chief','boresight','collimated'})} = 'chief'
        opts.ref_elt   (1,1) double = 0        % 0 = last
        opts.anchor    (1,3) double = [NaN NaN NaN]
        opts.strip_ap  (1,1) logical = true
        opts.min_rays  (1,1) double  = 10
        opts.model_size (1,1) double = 256
        opts.init      (1,1) logical = true
        opts.quiet     (1,1) logical = true
    end
    rung = find(strcmp(opts.reference, {'chief','boresight','collimated'}));

    txt = fileread(deck);
    if opts.strip_ap
        txt = regexprep(txt, '(ApType=\s*)\S+', '$1None');
    end
    [cdir0, cpos0, apst, lam] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));   by0 = asin(cdir0(2));
    Vs = grab_all3_(txt, 'VptElt');
    ie = opts.ref_elt;   if ie <= 0, ie = size(Vs,2); end
    anch = Vs(:,ie);
    if all(isfinite(opts.anchor)), anch = opts.anchor(:); end

    if opts.init, macos.init(opts.model_size); end

    K = size(Frel,1);
    out = struct('deck',deck, 'fields',Frel, 'lambda',lam, ...
                 'ref_elt',ie, 'anchor',anch.', 'reference',opts.reference, ...
                 'bias_deg',[bx0 by0]*180/pi, 'strip_ap',opts.strip_ap, ...
                 'wfe_m',nan(K,1), 'wfe',nan(K,1), ...
                 'wfe_m_chief',nan(K,1), 'wfe_m_boresight',nan(K,1), ...
                 'wfe_m_collimated',nan(K,1), ...
                 'tilt_urad',nan(K,1), 'tilt',nan(K,2), ...
                 'divergence_urad',nan(K,1), 'R_curv_m',nan(K,1), ...
                 'power_sag_nm',nan(K,1), 'bore_split_urad',nan(K,1), ...
                 'rho_max',nan(K,1), 'nrays',zeros(K,1), ...
                 'chief_dir',nan(3,K), 'chief_pos',nan(3,K));

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    for k = 1:K
        ri = trace_field_(txt, tmp, apst, stand, ...
                          bx0+Frel(k,1), by0+Frel(k,2), ie);
        ok = ri.ok_trace(:) & ri.ok_pass(:);
        ok(1) = false;                    % the chief is the reference, not a sample
        if nnz(ok) < opts.min_rays, continue; end
        p1 = ri.pos(:,1);   d1 = ri.dir(:,1);
        [v, ~, f] = afocal_rungs(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), ...
                                 p1, d1, anch);
        out.wfe_m_chief(k)      = v(1);
        out.wfe_m_boresight(k)  = v(2);
        out.wfe_m_collimated(k) = v(3);
        out.wfe_m(k)            = v(rung);
        out.wfe(k)              = v(rung)/lam;
        out.tilt(k,:)           = f.tilt;
        out.tilt_urad(k)        = f.tilt_urad;
        out.divergence_urad(k)  = f.divergence_urad;
        out.R_curv_m(k)         = f.R_curv_m;
        out.power_sag_nm(k)     = f.power_sag_m*1e9;
        out.bore_split_urad(k)    = f.bore_split_urad;
        out.rho_max(k)          = f.rho_max;
        out.nrays(k)            = nnz(ok);
        out.chief_dir(:,k)      = d1/norm(d1);
        out.chief_pos(:,k)      = p1;
        if ~opts.quiet
            fprintf(['   [%+7.3f %+7.3f]'' : %10.4g nm  ' ...
                     '(tilt %8.2f urad, div %8.2f urad, %d rays)\n'], ...
                Frel(k,1)*180/pi*60, Frel(k,2)*180/pi*60, v(rung)*1e9, ...
                f.tilt_urad, f.divergence_urad, nnz(ok));
        end
    end
end

% ---------------------------------------------------------------------
% Deck helpers.  Kept local (mirrors of AFOCAL_LADDER_DECK's) so this
% scorer is self-contained; TAFOCALKERNEL gates that the two agree.
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
        error('macos:design:afocal_wfe_deck:load', ...
              'deck failed to load: %s', tmp);
    end
    tr = macos.trace(ie);
    ri = macos.get_ray_info(tr.nRays);
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end

function [cdir, cpos, apst, lam] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');   cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    t = regexp(txt,'Wavelen=\s*([-\d.EeD+]+)','tokens','once');
    lam = str2double(strrep(t{1},'D','E'));
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
