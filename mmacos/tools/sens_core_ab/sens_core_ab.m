function sens_core_ab(setup_m, rx_path, outdir, fx, fy, model)
%SENS_CORE_AB  Deck-agnostic A/B battery for the sens-core supervisors.
%   sens_core_ab(SETUP_M, RX_PATH, OUTDIR, FX, FY, MODEL) runs a fixed
%   option-set battery for all four dw_d*_multi families on one deck and
%   saves each run-set's full output struct to OUTDIR/<set>.mat.  Run it
%   TWICE -- once with SETUP_M pointing at the PRE tree's
%   mmacos_setup.m (a worktree at the pre-sens-core commit) and once at
%   the sens-core tree -- into two OUTDIRs, then compare with
%   sens_core_ab_compare.  Expectation: BYTE-IDENTICAL (segment-class
%   corpus; Dave 2026-09-07).
%
%   Deck-agnostic by construction: element choices derive from the
%   discovery functions (identical on both trees -- both carry the
%   engine-truth discovery), capped small so any deck runs in minutes.
%   A set that errors is recorded as an error STRING in its .mat --
%   identical errors on both sides are error-parity (a pass); only a
%   pre/post DIFFERENCE is a finding.
%
%   FX/FY: field half-widths in rad.  MUST sit inside the deck's
%   vignetting margin ON THE PRE TREE (its dw_dx_multi still hard-errors
%   on a fully-vignetted field; note some decks' nominal chief is
%   off-axis -- the e5hex1 lesson).  Start small (1e-5) and grow only if
%   the deck's field behavior is known.  MODEL: engine model size
%   (default 256).
%
%   ONE MATLAB PROCESS PER CALL (the state-leak doctrine): drive the
%   pre/post x deck matrix from a shell loop, one matlab -batch per
%   invocation.  See README.md here + macos/BRIEF_ccmac_sens_core.md.
if nargin < 6, model = 256; end
run(setup_m);
macos.init(model);
m = macos.Session(model);
if ~exist(outdir, 'dir'), mkdir(outdir); end

% ---- deck-derived, capped channel choices (identical on both trees) --
m.load_rx(rx_path);
pe = macos.find_powered_elts(m, rx_path);  pe = pe(:);
try, ze = m.find_zern_elts(rx_path);  ze = ze(:);  catch, ze = []; end
try, g  = macos.find_grid_elts();      g  = g(:);   catch, g  = []; end
pick = @(v, n) v(1:min(n, numel(v)));
cfgs2 = [];
if ~isempty(pe)
    cfgs2 = [struct('name','cA', 'set', {{{'perturb', pe(end), ...
                'rotation', [1e-6;0;0], 'frame','local'}}}), ...
             struct('name','cB', 'set', {{{'perturb', pe(end), ...
                'rotation', [-1e-6;0;0], 'frame','local'}}})];
end
fprintf('[ab] %s: powered [%s], zern [%s], grid [%s]\n', rx_path, ...
    num2str(pe.'), num2str(ze.'), num2str(g.'));

% ---- the battery -----------------------------------------------------
runset_(outdir, 'surf_A', @() macos.dw_dsurf_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(pe,2)));
runset_(outdir, 'surf_B', @() macos.dw_dsurf_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(pe,2), ...
    'params',{'Kr'}, 'configs',cfgs2));
runset_(outdir, 'surf_C', @() macos.dw_dsurf_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(pe,2), ...
    'reset_xp',false, 'orient','xy', 'sign','wavefront', 'grid','3x1'));
runset_(outdir, 'dx_A', @() macos.dw_dx_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(pe,2), 'dofs',[0;5]));
runset_(outdir, 'dx_B', @() macos.dw_dx_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(pe,2), 'dofs',5, ...
    'configs',cfgs2, 'groups', grp_(pick(pe,3)), 'include_source',true));
runset_(outdir, 'dx_C', @() macos.dw_dx_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(pe,2), 'dofs',0, ...
    'reset_xp',false, 'orient','xy', 'grid','3x1'));
runset_(outdir, 'zern_A', @() macos.dw_dz_zernike_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(ze,1), 'n_zcoef',6));
runset_(outdir, 'zern_B', @() macos.dw_dz_zernike_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(ze,1), 'n_zcoef',6, ...
    'configs',cfgs2));
runset_(outdir, 'grid_A', @() macos.dw_dgrid_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(g,2)));
runset_(outdir, 'grid_B', @() macos.dw_dgrid_multi(m, rx_path, ...
    'field_x_rad',fx, 'field_y_rad',fy, 'elts',pick(g,2), ...
    'configs',cfgs2, 'reset_xp',false, 'orient','xy'));
fprintf('[ab] battery done -> %s\n', outdir);
end

function gm = grp_(elts)
if isempty(elts), gm = []; return; end
gm = containers.Map({'abgrp'}, {elts(:)});
end

function runset_(outdir, name, fn)
% A set that errors saves the error MESSAGE -- error parity across
% pre/post is a pass; a difference is the finding.
t0 = tic;
try
    S = fn();                                    %#ok<NASGU>
catch e
    S = struct('AB_ERROR', sprintf('%s: %s', e.identifier, e.message)); %#ok<NASGU>
end
save(fullfile(outdir, [name '.mat']), 'S', '-v7.3');
fprintf('[ab] %-8s %6.1f s\n', name, toc(t0));
end
