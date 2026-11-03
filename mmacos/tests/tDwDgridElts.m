classdef tDwDgridElts < matlab.unittest.TestCase
    % macos.dw_dgrid's 'elts' filter, and the grid<->MonZern convention it
    % has to preserve.
    %
    % dw_dgrid computed its filtered element list `g`, used it only to size
    % the default influence basis, and then called grid_channels WITHOUT
    % passing it -- so 'elts' was silently ignored and the Jacobian came back
    % with a column pair for every grid-bearing element in the Rx.  On a
    % 7-segment deck that is 8 elements' worth of columns when one was asked
    % for; the caller then picks a column by name and gets the right answer
    % for the wrong reason, having paid 8x the finite differences.
    %
    % Uses a REAL segmented deck (7 pie segments, all grid-bearing) so the
    % filter has something to filter; a single-grid-element Rx would pass
    % whether or not the fix is present.

    properties
        rx
        seg_elts = 2;      % Seg2
    end

    methods (TestClassSetup)
        function locate(tc)
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            tc.rx = fullfile(res_root, 'segmirmaker', 'test_in', 'e5pie.in');
            tc.assumeTrue(isfile(tc.rx), 'e5pie fixture missing');
        end
    end

    methods (Test)
        function test_elts_filter_is_honoured(tc)
            old = cd(fileparts(tc.rx));  restore = onCleanup(@() cd(old)); %#ok<NASGU>
            m = macos.Session(512);
            m.load_rx(tc.rx);
            all_grid = macos.find_grid_elts();
            tc.assumeGreaterThan(numel(all_grid), 2, ...
                'fixture must have several grid elements for this to mean anything');

            d1 = macos.dw_dgrid(m, tc.rx, 'elts', tc.seg_elts, 'zmodes', 5);
            tc.verifyEqual(unique(d1.iElt(:)).', tc.seg_elts, ...
                'dw_dgrid returned columns for elements that were not requested');
            tc.verifyEqual(size(d1.dwdg, 2), numel(tc.seg_elts), ...
                'one mode on one element must give exactly one column');

            % ... and the unfiltered call still covers everything
            d2 = macos.dw_dgrid(m, tc.rx, 'zmodes', 5);
            tc.verifyEqual(unique(d2.iElt(:)).', all_grid(:).', ...
                'unfiltered dw_dgrid must still cover every grid element');
        end

        function test_filtered_column_matches_unfiltered(tc)
        % The filter must select, not perturb: the requested element's column
        % has to be identical to the one the full sweep produces.
            old = cd(fileparts(tc.rx));  restore = onCleanup(@() cd(old)); %#ok<NASGU>
            m = macos.Session(512);
            d1 = macos.dw_dgrid(m, tc.rx, 'elts', tc.seg_elts, 'zmodes', 5);
            d2 = macos.dw_dgrid(m, tc.rx, 'zmodes', 5);
            k  = find(d2.iElt == tc.seg_elts, 1);
            tc.verifyNotEmpty(k);
            tc.verifyEqual(d1.dwdg(:,1), d2.dwdg(:,k), 'RelTol', 1e-12, ...
                'filtering changed the Jacobian column');
        end

        function test_grid_map_reproduces_monzern_poke(tc)
        % The convention this figure/driver pair rests on: a mode sampled onto
        % the grid in the segment's clocked face frame (zern_seg_eval) must
        % reproduce the MonZernCoef poke of the SAME mode.  Guards the pairing
        % used by the Luis OPD note's fig D.  Mode 5 = defocus.
            old = cd(fileparts(tc.rx));  restore = onCleanup(@() cd(old)); %#ok<NASGU>
            m = macos.Session(512);
            m.load_rx(tc.rx);
            elt = tc.seg_elts;  wf = m.num_elt() - 1;  mode = 5;  c = 1e-4;
            gi = macos.get_elt_grid(elt);
            cs = macos.get_elt_srf_csys(elt);
            % lMon from the deck text, NOT macos.get_elt_zrn_norm_radius:
            % that getter (and elt_srf_zrn_get) gate on
            % SrfType == SrfType_Zernike, so they return -1 for a FreeForm
            % segment even though the engine's Mon channel uses lMon(iElt)
            % for it.  Reported as an api gap; reading the deck keeps this
            % test independent of it.
            lmon = deck_lmon(tc.rx, elt);
            fr = struct('rpt', cs.pMon(:), 'xhat', cs.xMon(:), ...
                        'yhat', cs.yMon(:), 'lmon', lmon);
            N = gi.size;  c0 = (N+1)/2;  [I,J] = ndgrid(1:N,1:N);
            pts = fr.rpt + fr.xhat*((I(:).'-c0)*gi.dx) + fr.yhat*((J(:).'-c0)*gi.dx);
            map = reshape(macos.design.zern_seg_eval(fr, mode, pts), N, N);

            m.trace(wf);  W0 = m.opd();
            zc = macos.channels.MonZernChannel(m, elt, mode);
            zc.apply(c);  m.trace(wf);  Wz = m.opd();  zc.restore();
            gc = macos.channels.GridChannel(m, elt, map);
            gc.apply(c);  m.trace(wf);  Wg = m.opd();  gc.restore();

            dz = Wz(:) - W0(:);  dg = Wg(:) - W0(:);
            sel = abs(dz) > 0.1*max(abs(dz));
            tc.assumeGreaterThan(nnz(sel), 50, 'no segment support');
            scale = dg(sel) \ dz(sel);
            cc = corrcoef(dg(sel), dz(sel));
            tc.verifyEqual(scale, 1, 'AbsTol', 2e-2, ...
                'grid-sampled mode must reproduce the MonZern poke');
            tc.verifyGreaterThan(cc(1,2), 0.995);
        end
    end
end

function v = deck_lmon(rx, elt)
%DECK_LMON  lMon of element ELT, straight from the prescription text.
txt = strsplit(fileread(rx), newline);
ie = 0; v = NaN;
for k = 1:numel(txt)
    L = strtrim(txt{k});
    m1 = regexp(L, '^iElt\s*=\s*(\d+)', 'tokens', 'once');
    if ~isempty(m1), ie = str2double(m1{1}); continue, end
    if ie == elt
        m2 = regexp(L, '^lMon\s*=\s*(\S+)', 'tokens', 'once');
        if ~isempty(m2)
            v = str2double(strrep(strrep(m2{1},'D','E'),'d','e'));
            return
        end
    end
end
end
