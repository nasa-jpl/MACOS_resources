function result = test06_zern_response_e5hex1(opts)
% Zernike-channel apply on Rx_e5hex1's Zernike-typed Elt 8 (the
% middle reflector).  e5hex1's segments 1..7 are FreeForm not
% Zernike, so the GMI Zernike-apply path (which forces SrfType=8)
% would change a FreeForm into a Zernike behind our back -- pointing
% it at Elt 8 (already Zernike) makes the test meaningful.
%
% For a parallel FreeForm-channel test, see test_pymacos's MonZern
% sensitivity work; GMI's monzernSrf channel is verified in the
% pymacos suite rather than here.

    name = 'test06_zern_response_e5hex1';
    result = make_result(name);

    [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_e5hex1();

    target_elt   = 8;                    % Reflector with Surface=Zernike
    target_mode  = 4;                    % Z4 = defocus (Born&Wolf 4)
    amplitude_mm = 1.0d-8;
    param.zernSrf = [target_elt];

    nzern = param.mzern;
    pzern_loc = zeros(length(param.zernSrf) * nzern, 1);
    pzern_loc((target_mode - 3)) = amplitude_mm;

    try
        clear mex;
        [~, ~, OPDnom] = call_GMI(prb, zeros(size(pzern_loc)), pgrid, 0, 0, 0, ...
                                   param.pimg, InfFcnZern, InfFcnGrid, param);
        [~, ~, OPDpert] = call_GMI(prb, pzern_loc, pgrid, 0, 0, 0, ...
                                    param.pimg, InfFcnZern, InfFcnGrid, param);
    catch ME
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  call_GMI threw: %s', name, ME.message);
        return
    end

    dOPD = OPDpert - OPDnom;

    if max(abs(dOPD(:))) == 0
        result.pass = false;
        result.msg = sprintf(['[%s] FAIL  perturbation produced zero ' ...
                              'response'], name);
        return
    end

    ref_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                        'reference', 'zern_response_e5hex1.mat');
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
