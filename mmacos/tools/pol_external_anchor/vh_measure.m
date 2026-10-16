function m = vh_measure(work, lam_mm, aoi_deg, layers_mm, ngridpts)
%VH_MEASURE  Per-ray r_s/r_p of a coated flat, measured FRAME-FREE.
%
%   M = VH_MEASURE(WORK, LAM_MM, AOI_DEG, LAYERS_MM, NGRIDPTS)
%
%   Emits a Bench fold rig whose mirror sits at AOI_DEG, applies the
%   coating stack LAYERS_MM (col 1 complex index N = n - 1i*k, col 2
%   PHYSICAL thickness in mm = the rig's BaseUnits), traces, and returns
%   the per-ray amplitude ratio r_s/r_p in the engine's RAY-FOLLOWING
%   p-hat frame, plus each ray's own incidence cosine.
%
%   WHY TWO TRACES.  The obvious single-trace construction (tJonesPupil's
%   Fresnel gate) divides out the input state's projection on the s and p
%   axes, which requires knowing the engine's per-ray LAUNCH FRAME -- in
%   practice a hard-coded xGrid.  That is safe at 45 deg and quietly wrong
%   elsewhere: an input-frame error contaminates the ratio by a factor that
%   happens to be ~1 when the geometry is symmetric.  Measured cost of
%   getting this wrong: the diattenuation came out nearly FLAT in AOI
%   (-3.1e-3 at 2 deg vs -4.7e-3 at 45 deg) where an isotropic surface must
%   give D ~ theta^2 -- the same "flat where physics demands a power law"
%   signature that exposed the 2022 r_p sign defect.
%
%   Instead, trace TWO orthogonal input states and build the 2x2 map M from
%   the (unknown) input frame to the (s, p_r) output frame.  For an
%   isotropic surface the physics is diag(r_s, r_p) and the unknown frame
%   is a rotation R(phi), so
%
%       M = diag(r_s, r_p) * R(phi)
%         = [ r_s cos(phi)   -r_s sin(phi) ;
%             r_p sin(phi)    r_p cos(phi) ]
%
%   and therefore
%
%       r_s/r_p = M11/M22 = -M12/M21
%
%   with phi cancelling identically.  No launch-frame knowledge is used,
%   and the TWO independent estimates cross-check each other -- M.consistency
%   reports their relative disagreement, which is a built-in validity guard
%   rather than an assumption.

    b = macos.design.Bench('vhrig', 'aperture', 0.06, ...
                           'ngridpts', ngridpts, 'wavelen', lam_mm);
    % Fold in the x-z plane (source runs along +x).  The mirror DEVIATION
    % is 180 - 2*AOI, NOT 2*AOI: normal incidence (AOI 0) sends the beam
    % back on itself (deviation 180), grazing (AOI 90) leaves it undeviated
    % (deviation 0).  Getting this backwards sweeps the COMPLEMENT of the
    % intended angles and is self-cancelling at exactly 45 deg -- which is
    % the one angle the pre-existing Fresnel gate runs at, so nothing in
    % the suite would have caught it.
    dev = 180 - 2*aoi_deg;
    fold = b.add_fold(50, [cosd(dev); 0; sind(dev)]);
    b.add_detector(60);
    rx = fullfile(work, 'vhrig.in');
    b.emit(rx);

    macos.load_rx(rx);
    % An EMPTY stack leaves the element uncoated -- the perfect-conductor
    % idiom the Bench emits.  Used to measure the p-hat convention against
    % a case whose answer is known exactly, instead of arguing it.
    if ~isempty(layers_mm)
        macos.coating(fold, 'index',     real(layers_mm(:,1)).', ...
                            'extinc',    -imag(layers_mm(:,1)).', ...
                            'thickness', layers_mm(:,2).');
    end

    % ---- trace 1: input along the launch frame's x -----------------------
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.trace(fold);
    r1 = macos.ray_field(fold);

    % ---- trace 2: input along the launch frame's y -----------------------
    macos.polarization('on', 'Ex', [0 0], 'Ey', [1 0]);
    macos.trace(fold);
    r2 = macos.ray_field(fold);

    g = (r1.status == 0) & (r2.status == 0);

    kox = r1.kx(g); koy = r1.ky(g); koz = r1.kz(g);
    nx  = r1.nx(g); ny  = r1.ny(g); nz  = r1.nz(g);

    % incident direction: reflect the exit direction back through the flat
    kd  = kox.*nx + koy.*ny + koz.*nz;
    kix = kox - 2*kd.*nx;  kiy = koy - 2*kd.*ny;  kiz = koz - 2*kd.*nz;

    % output s/p frame: s = ki x n (normal to the plane of incidence),
    % p_r = s x ko (ray-following p-hat -- the engine's assembly basis)
    sx = kiy.*nz - kiz.*ny;  sy = kiz.*nx - kix.*nz;  sz = kix.*ny - kiy.*nx;
    sm = sqrt(sx.^2 + sy.^2 + sz.^2);
    sx = sx./sm; sy = sy./sm; sz = sz./sm;
    prx = sy.*koz - sz.*koy;  pry = sz.*kox - sx.*koz;  prz = sx.*koy - sy.*kox;

    M11 = r1.Ex(g).*sx  + r1.Ey(g).*sy  + r1.Ez(g).*sz;
    M21 = r1.Ex(g).*prx + r1.Ey(g).*pry + r1.Ez(g).*prz;
    M12 = r2.Ex(g).*sx  + r2.Ey(g).*sy  + r2.Ez(g).*sz;
    M22 = r2.Ex(g).*prx + r2.Ey(g).*pry + r2.Ez(g).*prz;

    rho_a = M11 ./ M22;
    rho_b = -M12 ./ M21;

    m.rho         = rho_a;
    m.rho_alt     = rho_b;
    m.consistency = max(abs(rho_a - rho_b) ./ abs(rho_a));
    m.cthi        = abs(kix.*nx + kiy.*ny + kiz.*nz);
    m.n           = nnz(g);
    m.elt         = fold;
end
