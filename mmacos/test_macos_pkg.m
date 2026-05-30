function test_macos_pkg(rx_path)
%TEST_MACOS_PKG  Smoke test for the +macos function package and the
%   macos.Session class.  Exercises both surfaces against the same Rx
%   to confirm they share state via the underlying libsmacos.a.
%
%   Layout:
%     Part A — function-style calls (macos.opd(), macos.trace(), ...)
%     Part B — class-style calls   (m = macos.Session(); m.opd())
%     Part C — cross-surface state check (function-style modifies, then
%              class-style observes; and vice versa)

if nargin < 1 || isempty(rx_path)
    rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
        'pymacos', 'tests', 'Rx', 'Rx_Cass_FarField.in');
end

n_pass = 0; n_fail = 0;
    function check(name, cond)
        if cond
            fprintf('  [PASS] %s\n', name); n_pass = n_pass + 1;
        else
            fprintf('  [FAIL] %s\n', name); n_fail = n_fail + 1;
        end
    end

fprintf('=== +macos package + Session smoke test ===\n');
fprintf('Rx: %s\n', rx_path);
if ~exist(rx_path, 'file')
    error('Rx not found: %s', rx_path);
end

% =============================================================
% Part A — function-style
% =============================================================
fprintf('\n--- Part A: macos.* function-style ---\n');

macos.init(128);
check('macos.init returns cleanly', true);

nElt_A = macos.load_rx(rx_path);
check('macos.load_rx returns >0', nElt_A > 0);
fprintf('  nElt = %d\n', nElt_A);

check('macos.has_rx true after load', macos.has_rx());
check('macos.num_elt matches load_rx', macos.num_elt() == nElt_A);

c = macos.cbm();
check('macos.cbm > 0', c > 0);
fprintf('  CBM = %g (Rx is in %g m units)\n', c, c);

wvl = macos.get_src_wvl();
check('macos.get_src_wvl > 0', wvl > 0);
fprintf('  wvl = %g WaveUnits\n', wvl);

vpt2_A = macos.get_elt_vpt(2);
check('macos.get_elt_vpt(2) returns 3-vec', numel(vpt2_A) == 3);

% Perturb the primary by a tiny tilt, trace, query OPD rms.
macos.perturb(2, 'rotation', [1e-6; 0; 0], 'frame', 'global');
check('macos.perturb(2, small Rx) returns', true);

s_trace_A = macos.trace();
check('macos.trace returns struct', isstruct(s_trace_A) && isfield(s_trace_A, 'nRays'));
fprintf('  trace.nRays = %d, trace.rmsWFE = %g\n', s_trace_A.nRays, s_trace_A.rmsWFE);
check('trace.nRays > 0', s_trace_A.nRays > 0);

W_A = macos.opd();
opd_rms_A = sqrt(mean(W_A(:).^2));
check('macos.opd is N x N', size(W_A,1) == size(W_A,2));
fprintf('  opd: %dx%d, rms = %g\n', size(W_A,1), size(W_A,2), opd_rms_A);

cf_A = macos.complex_field(nElt_A);
check('complex_field is complex', ~isreal(cf_A));

I_A = macos.intensity(nElt_A);
check('intensity peak > 0', max(I_A(:)) > 0);
fprintf('  intensity peak = %g\n', max(I_A(:)));

dx_m_A = macos.dx_at(nElt_A);
dx_mm_A = macos.dx_at(nElt_A, 'mm');
check('dx_at unit conversion (mm = m*1e3)', abs(dx_mm_A - dx_m_A * 1e3) < 1e-18);
fprintf('  dx_at(%d) = %g m = %g mm\n', nElt_A, dx_m_A, dx_mm_A);

% Reset the perturbation we applied so Part B starts clean.
macos.perturb(2, 'rotation', [-1e-6; 0; 0], 'frame', 'global');
macos.modify();

% =============================================================
% Part B — class-style via macos.Session
% =============================================================
fprintf('\n--- Part B: macos.Session class-style ---\n');

m = macos.Session(128);
check('macos.Session() constructs', isa(m, 'macos.Session'));

nElt_B = m.load_rx(rx_path);
check('m.load_rx returns >0', nElt_B > 0);
check('m.rx_path is set', strcmp(m.rx_path, rx_path));
check('class & function-style report same num_elt', nElt_B == nElt_A);

s_trace_B = m.trace();
check('m.trace returns nRays > 0', s_trace_B.nRays > 0);

W_B = m.opd();
opd_rms_B = sqrt(mean(W_B(:).^2));
fprintf('  m.opd: %dx%d, rms = %g\n', size(W_B,1), size(W_B,2), opd_rms_B);
% Part A traced the PERTURBED state (rms ~ 3.5e-8 from a 1 µrad tilt).
% Part B reloads Rx fresh, so its OPD is the CLEAN-state nominal
% (rms at the e-12 noise floor for a well-aligned Cass).  Their
% numerical inequality is therefore expected; same-shape is the only
% structural assertion that holds.  Cross-surface state coherence is
% proved in Part C below.
check('m.opd shape matches function-style', isequal(size(W_B), size(W_A)));
check('m.opd rms reasonable for clean state (< 1e-9)', opd_rms_B < 1e-9);
check('Part A perturbed rms > Part B clean rms (sanity)', ...
    opd_rms_A > opd_rms_B);

dx_native = m.dx_at(nElt_B, 'native');
fprintf('  m.dx_at(%d, native) = %g BaseUnits\n', nElt_B, dx_native);
check('dx_at native > 0', dx_native > 0);

% =============================================================
% Part C — cross-surface state check
% =============================================================
fprintf('\n--- Part C: cross-surface state check ---\n');

% Function-style sets sampling; class-style reads it back.
macos.set_src_sampling(64);
n_via_class = m.get_src_sampling();
check('function set + class get observe same sampling', n_via_class == 64);

% Class-style sets sampling; function-style reads it back.
m.set_src_sampling(128);
n_via_func = macos.get_src_sampling();
check('class set + function get observe same sampling', n_via_func == 128);

% =============================================================
% Summary
% =============================================================
fprintf('\n=== %d pass, %d fail ===\n', n_pass, n_fail);
if n_fail > 0
    error('+macos / Session smoke test failed');
end
end
