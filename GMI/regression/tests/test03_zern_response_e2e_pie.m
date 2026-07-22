function result = test03_zern_response_optiix(opts)
% Zernike-channel apply: put a 1e-8 mm Z4 (defocus) on the first
% segment via param.zernSrf, compare the OPD difference against a
% committed reference column.
%
% Catches:
%   - the SrfType=8 forcing in the Zernike apply path getting broken
%   - the iNode = 4..nzern+3 indexing convention drifting
%   - the ELSE-branch reset failing to zero ZernCoef between calls

    name = 'test03_zern_response_optiix';
    result = make_result(name);

    [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_optiix();

    % Pick the first segment of optiix as the Zernike target.
    target_elt   = 4;                    % param.rbSrf row 1
    target_mode  = 4;                    % Z4 = defocus (Born&Wolf 4)
    amplitude_mm = 1.0d-8;
    param.zernSrf = [target_elt];

    % pzern layout: zernSrf-major, mode-minor.  Mode k (4..nzern+3)
    % lives at index (i-1)*nzern + (k-3) within each element's block.
    nzern = param.mzern;
    pzern_loc = zeros(length(param.zernSrf) * nzern, 1);
    pzern_loc((target_mode - 3)) = amplitude_mm;    % Z4 on first (only) zernSrf elt

    try
        clear mex;
        % Nominal
        [~, ~, OPDnom] = call_GMI(prb, zeros(size(pzern_loc)), pgrid, 0, 0, 0, ...
                                   param.pimg, InfFcnZern, InfFcnGrid, param);
        % Perturbed
        [~, ~, OPDpert] = call_GMI(prb, pzern_loc, pgrid, 0, 0, 0, ...
                                    param.pimg, InfFcnZern, InfFcnGrid, param);
    catch ME
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  call_GMI threw: %s', name, ME.message);
        return
    end

    dOPD = OPDpert - OPDnom;

    % Sanity: response must be non-trivial.
    if max(abs(dOPD(:))) == 0
        result.pass = false;
        result.msg = sprintf(['[%s] FAIL  perturbation produced zero ' ...
                              'response (Zernike apply path broken?)'], name);
        return
    end

    % Reference: dOPD must match the committed column.
    ref_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                        'reference', 'zern_response_optiix.mat');
    if ~isfile(ref_path)
        result.pass = false;
        result.msg = sprintf(['[%s] FAIL  reference not found: %s\n  ' ...
                              '(re-bootstrap with ./run_regression.sh --bootstrap)'], ...
                             name, ref_path);
        return
    end
    ref = load(ref_path);
    [pass_ref, max_ref, msg_ref] = compare_within(dOPD, ref.dOPD, ...
                                                   opts.tol, 'vs reference');
    if ~pass_ref
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  %s', name, msg_ref);
        return
    end

    result.pass = true;
    result.msg = sprintf(['[%s] PASS  Z%d=%.1e on Elt %d -> dOPD ' ...
                          'max|.|=%.3e, vs-ref=%.3e (tol %.3e)'], ...
                         name, target_mode, amplitude_mm, target_elt, ...
                         max(abs(dOPD(:))), max_ref, opts.tol);
end

function r = make_result(name)
    r.name = name;
    r.pass = false;
    r.msg = sprintf('[%s] not run', name);
end
