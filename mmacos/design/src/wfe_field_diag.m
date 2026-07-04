function diag = wfe_field_diag(t, F, opts)
%WFE_FIELD_DIAG  Per-field wavefront decomposition -- WHAT is the field wall?
%   diag = wfe_field_diag(t, F) traces the design T (a macos.design.Telescope,
%   already built/optimized) at each field offset in F (Kx2 rad, as from
%   macos.design.field_ring/field_grid) and decomposes each field's OPD:
%
%     .rms_raw    raw RMS WFE                      (waves)
%     .rms_tilt   after removing piston+tip/tilt   (waves)
%     .rms_focus  after also removing defocus      (waves)
%     .rms_astig  after also removing astigmatism  (waves)
%     .z4         fitted defocus coefficient       (m)
%     .z56        fitted astig coefficients [Z5 Z6](m)
%
%   Reading the ladder tells you which correction can fix the field:
%     rms_focus << rms_tilt      -> field curvature (curved focal surface /
%                                   relay conjugates fix it)
%     rms_astig << rms_focus     -> field-dependent astigmatism (a FIXED
%                                   mirror Zernike cannot fix it if the
%                                   magnitude/orientation vary over the
%                                   field -- e.g. the binodal astigmatism
%                                   an eccentric-pupil off-axis section
%                                   induces; needs a 4th powered mirror,
%                                   field-conjugate freeform, or a smaller
%                                   field)
%     rms_astig still high       -> coma/higher order: revisit the conic /
%                                   rigid-body balance
%
%   Fits are least-squares on the lit-footprint-normalized pupil (unit
%   circle over the actual illuminated samples).  Name-value:
%     'lambda'  wavelength for the waves scaling (default t.spec.wavelength)
%     'quiet'   suppress the printed table (default false)
%
%   Uses Telescope.trace_at_field (restores the nominal field on exit).
%
%   See also macos.design.field_ring, macos.design.Telescope/trace_at_field.
    arguments
        t
        F (:,2) double
        opts.lambda (1,1) double = 0
        opts.quiet  (1,1) logical = false
    end
    lam = opts.lambda;
    if lam <= 0, lam = t.spec.wavelength; end
    nF = size(F,1);
    z = zeros(nF,4);  z4 = zeros(nF,1);  z56 = zeros(nF,2);
    cleanup = onCleanup(@() t.trace_at_field([]));
    for j = 1:nF
        t.trace_at_field(F(j,:));
        W = macos.opd();
        [ny,nx] = size(W);
        [X,Y] = meshgrid(linspace(-1,1,nx), linspace(-1,1,ny));
        m = isfinite(W) & (W ~= 0);
        x = X(m); y = Y(m); w = W(m);
        x = x - mean(x);  y = y - mean(y);
        s = max(hypot(x,y));  x = x/s;  y = y/s;
        B_t = [ones(size(x)), x, y];                     % piston + tilt
        B_f = [B_t, (2*(x.^2+y.^2)-1)];                  % + defocus
        B_a = [B_f, (x.^2-y.^2), (2*x.*y)];              % + astig
        c_t = B_t\w;  c_f = B_f\w;  c_a = B_a\w;
        z(j,:)   = [std(w), std(w-B_t*c_t), std(w-B_f*c_f), std(w-B_a*c_a)];
        z4(j)    = c_f(4);
        z56(j,:) = c_a(5:6).';
    end
    diag = struct('fields',F, 'lambda',lam, ...
                  'rms_raw',  z(:,1)/lam, 'rms_tilt', z(:,2)/lam, ...
                  'rms_focus',z(:,3)/lam, 'rms_astig',z(:,4)/lam, ...
                  'z4',z4, 'z56',z56);
    if ~opts.quiet
        fprintf('\n field(arcmin x,y)     raw    -tilt   -focus  -astig   Z4(nm)  |Z5,6|(nm)\n');
        for j = 1:nF
            fprintf(' (%+6.2f,%+6.2f)    %7.4f %7.4f %7.4f %7.4f  %+7.1f   %7.1f\n', ...
                rad2deg(F(j,1))*60, rad2deg(F(j,2))*60, ...
                z(j,1)/lam, z(j,2)/lam, z(j,3)/lam, z(j,4)/lam, ...
                z4(j)*1e9, hypot(z56(j,1),z56(j,2))*1e9);
        end
        fprintf(' worst:              %7.4f %7.4f %7.4f %7.4f  (waves @ %.3g um)\n', ...
            max(z(:,1))/lam, max(z(:,2))/lam, max(z(:,3))/lam, max(z(:,4))/lam, lam*1e6);
    end
end
