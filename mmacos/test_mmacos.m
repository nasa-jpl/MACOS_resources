function test_mmacos(rx_path)
%TEST_MMACOS  Smoke test for the mmacos mex bridge.
%   test_mmacos(RX_PATH) initializes mmacos, loads the Rx, runs a
%   handful of commands, and prints pass/fail per check.

if nargin < 1 || isempty(rx_path)
    rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
        'pymacos', 'tests', 'Rx', 'Rx_Cass_FarField.in');
end

n_pass = 0;
n_fail = 0;

    function check(name, cond)
        if cond
            fprintf('  [PASS] %s\n', name);
            n_pass = n_pass + 1;
        else
            fprintf('  [FAIL] %s\n', name);
            n_fail = n_fail + 1;
        end
    end

fprintf('=== mmacos smoke test ===\n');
fprintf('Rx: %s\n', rx_path);

if ~exist(rx_path, 'file')
    error('Rx not found: %s', rx_path);
end

mmacos('init', 128);
check('init returns', true);

% Strip the trailing .in extension if present -- macos's OLD command
% appends .in unconditionally and tries to load 'foo.in.in' otherwise
% (same workaround pymacos applies in macos.py).
[~, ~, ext] = fileparts(rx_path);
if strcmpi(ext, '.in')
    rx_for_load = rx_path(1:end-3);
else
    rx_for_load = rx_path;
end
nElt = mmacos('load_rx', rx_for_load);
check('load_rx returns >0', isnumeric(nElt) && nElt > 0);
fprintf('  nElt = %d\n', nElt);

n_query = mmacos('n_elt');
check('n_elt matches load_rx', n_query == nElt);

cbm = mmacos('base_unit_to_metres');
check('base_unit_to_metres > 0', isnumeric(cbm) && cbm > 0);
fprintf('  CBM = %g\n', cbm);

mmacos('modified_rx');
check('modified_rx returns', true);

% Perturbation: tiny Rx tilt of element 2 and back.
if nElt >= 2
    mmacos('perturb_elt', 2, [1e-6; 0; 0; 0; 0; 0], 1);
    check('perturb_elt(2, small Rx, global) returns', true);
    mmacos('perturb_elt', 2, [-1e-6; 0; 0; 0; 0; 0], 1);
end

% --- Trace-dependent commands ------------------------------------
% trace_rays runs a full ray trace; required before opd/intensity/etc.
% Returns [nRays, rms_WFE].
[nRays, rmsWFE] = mmacos('trace_rays', nElt);
check('trace_rays returns nRays > 0', nRays > 0);
fprintf('  nRays = %d, rms_WFE = %g\n', nRays, rmsWFE);

% OPD at image plane
opd = mmacos('opd');
check('opd is square matrix', isnumeric(opd) && ismatrix(opd) && ...
    size(opd,1) == size(opd,2));
opd_rms = sqrt(mean(opd(:).^2));
fprintf('  opd: %d x %d, rms = %g\n', size(opd,1), size(opd,2), opd_rms);

% Intensity at image plane
int_im = mmacos('intensity', nElt);
check('intensity has positive peak', max(int_im(:)) > 0);
fprintf('  intensity: %d x %d, peak = %g\n', size(int_im,1), ...
    size(int_im,2), max(int_im(:)));

% Complex field at image plane
cf = mmacos('complex_field', nElt);
check('complex_field is complex', ~isreal(cf));
fprintf('  cfield: %d x %d, |peak|^2 = %g\n', size(cf,1), size(cf,2), ...
    max(abs(cf(:)).^2));

% dx_at the image plane
dx = mmacos('dx_at', nElt);
check('dx_at returns positive scalar', isnumeric(dx) && dx > 0);
fprintf('  dx_at(%d) = %g m\n', nElt, dx);

fprintf('=== %d pass, %d fail ===\n', n_pass, n_fail);
if n_fail > 0
    error('mmacos smoke test failed');
end
end
