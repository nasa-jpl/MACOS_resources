function T = afocal4_wfe_terms(P, deck, opts)
%AFOCAL4_WFE_TERMS  WHAT is the wavefront wall made of?
%
%   T = AFOCAL4_WFE_TERMS(P, DECK) decomposes the delivered wavefront of an
%   afocal deck into Zernike terms, per field, so that a solve which stalls
%   can be diagnosed instead of merely re-run with a bigger iteration cap.
%   The S4 brief's rule: if the solve stalls more than 2x above target, STOP
%   the polish and characterise WHY -- which aberration term, which DOF is
%   exhausted -- before spending more machine time.
%
%   HOW TO READ IT.  The wavefront is the AFOCAL rung-1 residual: optical
%   path to a flat reference normal to that field's own exit chief.  Rungs 2
%   and 3 of AFOCAL_RUNGS remove tilt and defocus, which is exactly removing
%   the first four columns of this fit, so the fit and the ladder are the
%   same statement at different resolutions.
%     tilt          boresight, removed at rung 2 -- not an error
%     defocus       residual divergence, removed at rung 3 -- a deliverable
%     spherical     a coaxial system's own aberration; a CONIC buys it,
%                   provided the surface has a beam footprint to act on
%     spherical_2   fifth order.  Conics buy third order first; when the
%                   residual is dominated by fifth order, adding conics to
%                   the SAME surfaces will not help and the layout is what
%                   has to change
%     astig / coma  field aberrations: present only off axis on a coaxial
%                   design, and the terms rigid bodies and the field mirror
%                   act on
%
%   Name-value:  'fields' (P.Fsolve), 'quiet' (false)
%
%   Returns T with .names, .coef (K x nterm, metres), .rms_nm (K x 1, the
%   rung-1 RMS), .dominant (per field), and .table (the printed block).
%
%   See also AFOCAL_RUNGS, AFOCAL_REFS, AFOCAL4_SCORE.

    arguments
        P (1,1) struct
        deck (1,:) char
        opts.fields (:,2) double = P.Fsolve
        opts.quiet  (1,1) logical = false
    end

    txt = regexprep(fileread(deck), '(ApType=\s*)\S+', '$1None');
    cdir = grab3_(txt,'ChfRayDir');   cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    stand = dot(apst - cpos, cdir);
    bx0 = asin(cdir(1));   by0 = asin(cdir(2));
    Vs = grab_all3_(txt,'VptElt');   ie = size(Vs,2);   anch = Vs(:,ie);

    nm = {'piston','tilt_x','tilt_y','defocus','astig0','astig45', ...
          'coma_x','coma_y','spherical','trefoil0','trefoil30', ...
          'astig5_0','astig5_45','coma5_x','coma5_y','spherical_2'};
    K = size(opts.fields,1);
    C = nan(K, numel(nm));   rms_nm = nan(K,1);
    tmp = [tempname '.in'];
    cu = onCleanup(@() del_(tmp)); %#ok<NASGU>

    for k = 1:K
        ri = trace_field_(txt, tmp, apst, stand, ...
                          bx0+opts.fields(k,1), by0+opts.fields(k,2), ie);
        ok = ri.ok_trace(:) & ri.ok_pass(:);   ok(1) = false;
        if nnz(ok) < 10, continue; end
        f = afocal_refs(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), ...
                        ri.pos(:,1), ri.dir(:,1), anch);
        % NORMALISE THE PUPIL FIRST.  afocal_refs returns px/py in METRES
        % (a 16.7 mm exit-pupil radius here), and a fifth-order polynomial in
        % metres is r^6 ~ 1e-11 -- so an un-normalised fit returns
        % coefficients of 1e10 nm that cancel to nanometres.  That is the
        % same pathology the rodgers1 Zernike-solve doctrine records: a fit
        % whose terms cancel at metre scale is a fit about nothing.  On the
        % unit disk the basis below is orthogonal and Noll-normalised, so
        % each coefficient IS that term's RMS contribution in metres.
        rho = max(hypot(f.px, f.py));
        B = zbasis_(f.px/rho, f.py/rho);
        C(k,:) = (B\f.W).';
        rms_nm(k) = std(f.W)*1e9;
    end

    % The dominant term ignores piston (no statistic contains it) and tilt
    % (rung 2 removes it as boresight): what is asked is what the SOLVE has
    % to beat, not what the reference convention already removed.
    amp = abs(C);   amp(:,1:3) = 0;
    [~, im] = max(amp, [], 2);
    T = struct('names',{nm}, 'coef',C, 'rms_nm',rms_nm, 'fields',opts.fields, ...
               'dominant',{nm(im)}, 'deck',deck);

    if ~opts.quiet
        fprintf('\n  WAVEFRONT DECOMPOSITION  %s\n', deck);
        fprintf('  rung-1 wavefront, nm, per field (piston and tilt shown but not ranked)\n');
        fprintf('  %8s %8s %9s |', 'XAN','YAN','rms');
        show = [4 5 6 7 8 9 16];
        fprintf(' %10s', nm{show});   fprintf(' | %s\n', 'dominant');
        for k = 1:K
            fprintf('  %8.3f %8.3f %9.1f |', opts.fields(k,1)*180/pi, ...
                    opts.fields(k,2)*180/pi, rms_nm(k));
            fprintf(' %10.1f', C(k,show)*1e9);
            fprintf(' | %s\n', T.dominant{k});
        end
    end
end

% =====================================================================
function B = zbasis_(u, v)
%ZBASIS_  Noll-normalised Zernikes through fifth order on the UNIT DISK.
%   Orthonormal there, so each fitted coefficient is that term's RMS
%   contribution to the wavefront and the terms do not trade against each
%   other.  Callers must hand in normalised coordinates.
    r2 = u.^2 + v.^2;   r4 = r2.^2;
    s3 = sqrt(3);   s5 = sqrt(5);   s6 = sqrt(6);
    s7 = sqrt(7);   s8 = sqrt(8);   s10 = sqrt(10);   s12 = sqrt(12);
    B = [ones(size(u)), ...
         2*u, 2*v, ...                                   % tilt
         s3*(2*r2-1), ...                                % defocus
         s6*(u.^2-v.^2), s6*(2*u.*v), ...                % astigmatism
         s8*(3*r2-2).*u, s8*(3*r2-2).*v, ...             % coma
         s5*(6*r4-6*r2+1), ...                           % spherical
         s8*(u.^3-3*u.*v.^2), s8*(3*u.^2.*v-v.^3), ...   % trefoil
         s10*(4*r2-3).*(u.^2-v.^2), s10*(4*r2-3).*(2*u.*v), ...  % 5th astig
         s12*(10*r4-12*r2+3).*u, s12*(10*r4-12*r2+3).*v, ...     % 5th coma
         s7*(20*r2.^3-30*r4+12*r2-1)];                   % 2nd spherical
end

function ri = trace_field_(txt, tmp, apst, stand, bx, by, ie)
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
    s = regexprep(s,   '(yGrid=\s*)[^\n]*', ['$1' v3_([0; cos(by); -sin(by)])]);
    fid = fopen(tmp,'w');   fprintf(fid,'%s',s);   fclose(fid);
    macos.load_rx(tmp);
    tr = macos.trace(ie);
    ri = macos.get_ray_info(tr.nRays);
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function M = grab_all3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    M = zeros(3, numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'),'%f',3); end
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
