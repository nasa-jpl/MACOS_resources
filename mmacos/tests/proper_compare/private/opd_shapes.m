function opd = opd_shapes(name, N, dx_m, r_norm_m, rms_m, varargin)
%OPD_SHAPES  Generate one of three diagnostic OPD shapes (m).
%   Mirror of pymacos test_coro_dm_phase.py helpers
%   (opd_defocus, opd_sinusoid, opd_filtered_noise).
%
%   All shapes are N×N real, zero outside r_norm_m, and rescaled so
%   the in-support RMS equals rms_m.
%
%   name = 'defocus'         — Z4 quadratic on the unit disk
%        = 'sinusoid'        — sin(2*pi*k*x/D) along x; needs 'cycles' opt
%        = 'filtered_noise'  — low-pass noise capped at k_max cycles per
%                              pupil radius; needs 'max_cycles' + 'seed' opts
arguments
    name      (1,:) char
    N         (1,1) double {mustBeInteger, mustBePositive}
    dx_m      (1,1) double {mustBePositive}
    r_norm_m  (1,1) double {mustBePositive}
    rms_m     (1,1) double {mustBePositive}
end
arguments (Repeating)
    varargin
end

% Parse extra opts
opts = struct('cycles', [], 'max_cycles', [], 'seed', 12345);
for i = 1:2:numel(varargin)
    opts.(varargin{i}) = varargin{i+1};
end

% Centred mesh in metres, MATLAB ndgrid order matches numpy 'ij'.
ax = ((0:N-1) - (N - 1) / 2) * dx_m;
[xx, yy] = ndgrid(ax, ax);
rr = hypot(xx, yy);

switch lower(name)
    case 'defocus'
        rho = rr / r_norm_m;
        opd = 2 * rho.^2 - 1;
        inside = rho <= 1;
        opd(~inside) = 0;
    case 'sinusoid'
        if isempty(opts.cycles)
            error('opd_shapes:missingCycles', ...
                "'sinusoid' shape needs 'cycles', N option");
        end
        pupil_D = 2 * r_norm_m;
        opd = sin(2 * pi * opts.cycles * xx / pupil_D);
        inside = rr <= r_norm_m;
        opd(~inside) = 0;
    case 'filtered_noise'
        if isempty(opts.max_cycles)
            error('opd_shapes:missingMaxCycles', ...
                "'filtered_noise' needs 'max_cycles', K option");
        end
        rng_state = rng();
        cleanup = onCleanup(@() rng(rng_state));
        rng(opts.seed, 'twister');
        noise = randn(N, N);

        % FFT-space low-pass at k_cutoff = max_cycles / r_norm_m
        fx = (mod((0:N-1) + floor(N/2), N) - floor(N/2)) / (N * dx_m);
        fy = fx;
        [kxx, kyy] = ndgrid(fx, fy);
        k_mag    = hypot(kxx, kyy);
        k_cutoff = opts.max_cycles / r_norm_m;

        F = fft2(noise);
        F(k_mag > k_cutoff) = 0;
        opd = real(ifft2(F));
        inside = rr <= r_norm_m;
        opd(~inside) = 0;
    otherwise
        error('opd_shapes:unknownShape', 'unknown shape: %s', name);
end

% Rescale to requested in-support RMS.
inside = rr <= r_norm_m;
rms_actual = sqrt(mean(opd(inside).^2));
if rms_actual > 0
    opd = opd * (rms_m / rms_actual);
end
end
