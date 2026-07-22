function result = test01_smoke_optiix(opts)
% Smoke test on the Optiix Rx: load + nominal call + sanity assertions
% on shape, finiteness, sum > 0.  Does NOT compare to reference yet --
% the harness's "first call after a `clear mex`" is what's exercised.

    name = 'test01_smoke_optiix';
    result = make_result(name);

    [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_optiix();

    try
        clear mex;
        [PIX, CEFnom, OPDnom, OPDnomMask, SPOT, WFE, c, metMeas, R] = ...
            call_GMI(prb, pzern, pgrid, 0, 0, 0, ...
                     param.pimg, InfFcnZern, InfFcnGrid, param);
    catch ME
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  call_GMI threw: %s', name, ME.message);
        return
    end

    % Shape: OPDnom is square and roughly mdttl-sized (the actual size
    % is mdttl-1 due to OPD-packing conventions).  Just sanity-check
    % squareness and rough magnitude.
    % macos enforces 2x padding on the diffraction grid; OPD is
    % returned at the (unpadded) pupil sampling: (mdttl/2 - 1) per
    % side.  For mdttl=512 -> 255x255; mdttl=256 -> 127x127.
    expected = param.mdttl / 2 - 1;
    if ~isequal(size(OPDnom), [expected, expected])
        result.pass = false;
        result.msg = sprintf(['[%s] FAIL  OPDnom shape %s != [%d %d] ' ...
                              '(expected mdttl/2-1 per side, mdttl=%d)'], ...
                             name, mat2str(size(OPDnom)), expected, expected, ...
                             param.mdttl);
        return
    end
    if ~all(isfinite(OPDnom(:)))
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  OPDnom contains non-finite values', name);
        return
    end
    if sum(abs(OPDnom(:))) == 0
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  OPDnom is identically zero', name);
        return
    end

    result.pass = true;
    result.msg = sprintf(['[%s] PASS  OPDnom shape=[%d %d], RMS=%.3e, ' ...
                          'WFE=%.3e'], ...
                         name, size(OPDnom,1), size(OPDnom,2), ...
                         sqrt(mean(OPDnom(OPDnom~=0).^2)), WFE);
end

function r = make_result(name)
    r.name = name;
    r.pass = false;
    r.msg = sprintf('[%s] not run', name);
end
