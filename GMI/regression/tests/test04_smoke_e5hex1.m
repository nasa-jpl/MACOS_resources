function result = test04_smoke_e5hex1(opts)
% Smoke test on Rx_e5hex1: load + nominal call + basic sanity.

    name = 'test04_smoke_e5hex1';
    result = make_result(name);

    [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_e5hex1();

    try
        clear mex;
        [PIX, CEFnom, OPDnom, OPDnomMask, SPOT, WFE] = ...
            call_GMI(prb, pzern, pgrid, 0, 0, 0, ...
                     param.pimg, InfFcnZern, InfFcnGrid, param);
    catch ME
        result.pass = false;
        result.msg = sprintf('[%s] FAIL  call_GMI threw: %s', name, ME.message);
        return
    end

    % macos enforces 2x padding on the diffraction grid; OPD is
    % returned at the (unpadded) pupil sampling: (mdttl/2 - 1) per
    % side.  For mdttl=256 -> 127x127.
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
