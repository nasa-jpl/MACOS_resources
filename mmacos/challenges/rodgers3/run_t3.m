function OUT = run_t3()
%RUN_T3  The challenge run: OUR template at Mike's parameters.
%
%   Runs templates/10_telescopes/offset_imager end-to-end (S1-S5) at the
%   rodgers3 parameter set with HIS constraint set pinned:
%     - exit beam horizontal = box-centre exit chief along [0 0 -1]
%       (MEASURED from his decks: r3/r4/r5 hold it to <= 2e-5 rad, r2 --
%       FPA freedom only -- sits 0.24 deg off; scratch probe r3_exit),
%     - clearances > 50 / > 35 mm reported per leg (the PACKET states
%       the pairing interpretation).
%   Artifacts land in t3/ (decks, figures, oi_REPORT.md, run .mat).
%   PACKET.md quotes this run's ladder against his and Stage-0's.
%
%   Then the two BOUNDED counter-design looks (timeboxed; results are
%   flagged, not iterated):
%     (a) sphere+Zernike from the start (sz doctrine): S3's radii with
%         K = 0, aspheres dropped, straight to the S5 Zernike solve.
%     (b) term-set probe: the S5 solve rerun with Mike's holdouts
%         RELEASED into the basis (power mode 5 + y-tilt mode 3) -- if
%         the ladder drops materially, his 53 nm is term-set-limited.
%
%   See also OFFSET_IMAGER, PACKET.md, rodgers3.m.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','mmacos_setup.m'));
    addpath(here);
    addpath(fullfile(here,'..','..','templates','10_telescopes','offset_imager'));

    outdir = fullfile(here,'t3');

    % ---- DECK-TRUTH parameters, read from the .seq truth at runtime ------
    % Nothing hand-copied: the packaging (z_m1, spacings) and the focal
    % ratio come from rodgers3_seq itself.  The r1 deck measures EFL
    % 300.003 mm = F/4.00004 -- the slide's F/4 CONFIRMED (and a guard
    % against exactly the station-vs-spacing transcription slip that
    % briefly suggested otherwise: the .seq THICKNESSES are th4 = -722.9,
    % th6 = +740.8 mm; the m1/stop/m3 STATIONS are +665/-58/+683 mm).
    S = rodgers3_seq();
    r1 = S.r1.s([4 6 7]);
    tnet = [S.r1.s(4).th, S.r1.s(6).th]*1e-3;
    fo_deck = oi_paraxial([r1.R]*1e-3, tnet);
    Fno_deck = fo_deck.EFL_m / 0.075;
    z_m1  = (S.r1.s(2).th + S.r1.s(3).th)*1e-3;
    spac  = [S.r1.s(4).th, S.r1.s(5).th, S.r1.s(6).th]*1e-3;
    fprintf('deck-truth EFL %.6f m -> F/%.5f; z_m1 %.6f; spacings [%g %g %g] m\n', ...
            fo_deck.EFL_m, Fno_deck, z_m1, spac);

    % ================= the main five-stage run ==============================
    OUT = offset_imager(struct( ...
        'name','rodgers3-T3', 'tag','r3t', 'outdir',outdir, ...
        'Fno', Fno_deck, 'z_m1_m', z_m1, 'spacings_m', spac, ...
        'exit_dir',[0 0 -1], 'exit_tol_deg',0.1, ...
        'gn_iters',30));

    P = OUT.P;

    % ================= counter-design (a): sphere + Zernike =================
    fprintf('\n===== counter-design (a): sphere+Zernike from the start =====\n');
    Xa = OUT.s3.X;                    % S3's radii and packaging...
    Xa.K = [0 0 0];  Xa.asph = zeros(3,3);   % ...but PURE SPHERES
    Xa.fpa_refit = [0 0];
    Xa = oi_zern_seed(Xa, P);         % zero-coefficient Zernike surfaces
    [Xa, ha] = oi_solve(Xa, P, 'S5', 'walls', @(Xc,Gc) false);
    [Xa, Ga] = oi_close(Xa, P);  Xa.fpa = oi_apply_fpa(Xa);  Ga.fpa = Xa.fpa;
    [~, mpa] = oi_map_fig(Xa, Ga, P, P.offset_deg, ...
        'counter (a): sphere+Zernike from the start', ...
        fullfile(outdir,'r3t_ca_map.png'));
    fprintf('  counter (a): %.1f -> %.1f nm (solve), map max %.1f nm\n', ...
            ha.rms0, ha.rms, mpa.max_nm);
    OUT.counter_a = struct('X',Xa,'map',mpa,'hist',ha);

    % ================= counter-design (b): released term set =================
    fprintf('\n===== counter-design (b): S5 with power + y-tilt released =====\n');
    Xb = OUT.s4.X;
    Xb.fpa_refit = [0 0];
    Xb = oi_zern_seed(Xb, P, 'modes', sort([3 5 P.zern_modes]));
    Pb = P;  Pb.zern_modes = sort([3 5 P.zern_modes]);
    [Xb, hb] = oi_solve(Xb, Pb, 'S5', 'walls', @(Xc,Gc) false);
    [Xb, Gb] = oi_close(Xb, Pb);  Xb.fpa = oi_apply_fpa(Xb);  Gb.fpa = Xb.fpa;
    [~, mpb] = oi_map_fig(Xb, Gb, Pb, Pb.offset_deg, ...
        'counter (b): S5 with power + y-tilt in the basis', ...
        fullfile(outdir,'r3t_cb_map.png'));
    fprintf('  counter (b): %.1f -> %.1f nm (solve), map max %.1f nm  (main S5 %.1f)\n', ...
            hb.rms0, hb.rms, mpb.max_nm, OUT.s5.map.max_nm);
    OUT.counter_b = struct('X',Xb,'map',mpb,'hist',hb);

    save(fullfile(outdir,'r3t_run.mat'), 'OUT');
    fprintf('\nsaved %s\n', fullfile(outdir,'r3t_run.mat'));
end
