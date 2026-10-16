% example_telescope_align.m
% -------------------------------------------------------------------
% Sprint 2A-ii worked example: the de-novo BUILDER feeding the same
% analysis core the importer (Sprint 2A-i) uses.  Declare design intent
% for a classical Cassegrain, build the prescription from closed-form
% optics, then align an M2 despace error back to minimum WFE.
%
%   Telescope(...)  ->  build()  ->  System.from_rx  ->  vary
%                    ->  evaluate  ->  optimize
%
% The builder owns the geometry (layout + conics, Schroeder (m,β),
% PLAN_DESIGN_LAYER §5); everything downstream is the shared
% front-end-agnostic analysis surface (§1.0).  No CodeV/Zemax step.
%
% Run interactively:  >> run('.../example_telescope_align.m')
% (Batch check: matlab -batch "run('.../example_telescope_align.m'); exit(0)")

addpath('~/dev/MACOS_resources/mmacos/src');   % +macos on path
exdir = pwd;                                    % save artifacts here

%% 1. Declare design intent (no engine calls) — a 1 m f/8 classical Cass.
t = macos.design.Telescope('family','Cassegrain', ...
        'aperture_diameter_m', 1.0, 'system_fnum', 8.0, ...
        'primary_fnum', 2.0, 'BFD_m', 0.25, ...
        'model_size', 256, 'grid_npts', 41);
fprintf('=== built design ===\n');
t.describe();

%% 2. Build: derive -> emit .in -> validate by loading through SMACOS.
rx = t.build();
fprintf('\n  emitted + loaded: %s\n', rx);

%% 3. Hand the emitted Rx to the analysis core (same surface as import).
s = macos.design.System.from_rx(rx, 'model_size', 256, 'init', false);

%% 4. Align: M2 (element 2) despace is the free DOF.
s.vary(2, 'despace', 'bounds', [-2 2], 'unit', 'mm');
m_nom = s.evaluate(0).merit;        % aligned baseline (spherical-free)
m_off = s.evaluate(0.5).merit;      % +0.5 mm M2 despace
fprintf('\n  WFE aligned baseline   : %.6e m\n', m_nom);
fprintf('  WFE +0.5 mm M2 despace : %.6e m  (%.0fx worse)\n', m_off, m_off/max(m_nom,eps));

%% 5. Optimize the despace back to alignment.
res = s.optimize('x0', 0.5, 'MaxIter', 40);
fprintf('\n  optimize() from the 0.5 mm disturbance:\n');
fprintf('   merit0 (start)        : %.6e m\n', res.merit0);
fprintf('   merit_opt (end)       : %.6e m\n', res.merit_opt);
fprintf('   recovered despace     : % .4e mm   (target ~0)\n', res.x_opt);
fprintf('   exitflag              : %d\n', res.exitflag);

%% 6. Add an accessible exit pupil + export the (nominal) design.
%  The exit pupil is the deliverable handle for downstream instruments;
%  add_pupil keeps the focal plane and inserts the image + pupil refs
%  before it.  (The alignment above ran on the imported copy `s`; the
%  builder `t` holds the clean nominal design we export.)
fprintf('\n=== export the design ===\n');
t.add_pupil();
p = t.spec.pupil;
fprintf('  exit pupil @ z = %.4f m,  reference radius = %.4f m\n', ...
        p.ep_vpt(3), p.ep_radius);
out_rx   = fullfile(exdir, 'cass_1m_f8.in');
out_spec = fullfile(exdir, 'cass_1m_f8.mat');
t.save(out_rx);  t.save_spec(out_spec);
fprintf('  saved: %s\n         %s\n', out_rx, out_spec);

fprintf('\n  done.  (design + exit pupil on disk; MATLAB stays open)\n');
