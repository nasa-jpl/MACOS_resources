function result = test02_nominal_repro_optiix(opts)
% Nominal-call repeatability: call_GMI twice with zero perturbations,
% the two OPDnom arrays must be bit-identical.  Catches state-drift
% bugs in SetToNominalSettings / ObtainNominalSettings (the class of
% bug CLAUDE.md documents: missing pFF/xFF/... in the FreeForm
% snapshot caused exactly this symptom).
%
% Also compares against the committed reference OPDnom to catch
% inter-build / inter-version drift.

    name = 'test02_nominal_repro_optiix';
    result = make_result(name);

    [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_optiix();

    try
        clear mex;
        [~, ~, OPDnom1] = call_GMI(prb, pzern, pgrid, 0, 0, 0, ...
                                    param.pimg, InfFcnZern, InfFcnGrid, param);
        [~, ~, OPDnom2] = call_GMI(prb, pzern, pgrid, 0, 0, 0, ...
                                    param.pimg, InfFcnZern, InfFcnGrid, param);
    catch ME
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  call_GMI threw: %s', name, ME.message);
        return
    end

    % Round-trip: two consecutive calls must match exactly.
    [pass_rt, max_rt, msg_rt] = compare_within(OPDnom1, OPDnom2, 0, ...
                                               'repro round-trip');
    if ~pass_rt
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  %s', name, msg_rt);
        return
    end

    % Reference: call 1 must match the committed reference.
    ref_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                        'reference', 'nominal_optiix.mat');
    if ~isfile(ref_path)
        result.pass = false;
        result.msg = sprintf(['[%s] FAIL  reference not found: %s\n  ' ...
                              '(re-bootstrap with ./run_regression.sh --bootstrap)'], ...
                             name, ref_path);
        return
    end
    ref = load(ref_path);
    [pass_ref, max_ref, msg_ref] = compare_within(OPDnom1, ref.OPDnom, ...
                                                   opts.tol, 'vs reference');
    if ~pass_ref
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  %s', name, msg_ref);
        return
    end

    result.pass = true;
    result.msg = sprintf('[%s] PASS  round-trip=%.3e, vs-ref=%.3e (tol %.3e)', ...
                         name, max_rt, max_ref, opts.tol);
end

function r = make_result(name)
    r.name = name;
    r.pass = false;
    r.msg = sprintf('[%s] not run', name);
end
