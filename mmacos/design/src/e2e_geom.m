function G = e2e_geom(GT, GD)
%E2E_GEOM  Telescope + spectrometer as ONE chain, the grating its stop.
%   G = e2e_geom(GT, GD) prepends the telescope chain GT (telescope_geom,
%   already PLACED in GD's frame: its terminal plane is GD's slit plane) to
%   the spectrometer chain GD (spectrometer_geom, its groove period and
%   focus as solved or held): sky -> TelM1 -> TelM2 -> TelM3 [-> TelFold]
%   -> Slit (a recorded 'pass' station: the slit plane, no refraction) ->
%   the Dyson's surfaces -> FPA.  The source is the telescope's collimated
%   field bundle (chain_bundle), aimed per field through the GRATING's
%   vertex (G.iStop = G.iG): the grating is the stop of the instrument, as
%   Mouroulis & Green prescribe and as the engine's macos.stop(iG) does on
%   the emitted deck.  Everything the Dyson's consumers read (G.fpa, G.n,
%   G.grating, G.slit, G.P, G.aim for the slit-side aim) is GD's; the
%   detector-frame scorer, the clearance gate (form 'dyson': the slit mask
%   and the FPA package are bodies, the telescope mirrors join the body
%   list, the leg into the slit is exempt from the mask it passes through)
%   and the emitter read the combined chain unchanged.
    nT = numel(GT.surf);
    ST = GT.surf;
    ST(nT).act = 'pass';  ST(nT).name = 'Slit';          % the slit plane: recorded, not an image terminal here
    S = [ST, GD.surf];
    G = GD;
    G.surf = S;  G.iG = GD.iG + nT;  G.iSlit = nT;  G.iStop = G.iG;
    G.form = 'dyson';  G.e2e = true;  G.tel = GT;
    G.launch = GT.launch;  G.src = GT.src;  G.src.u = GD.src.u;  G.src.zsrc_gap = GD.src.zsrc_gap;
    G.src.lambda_c = GD.src.lambda_c;
    G.field_dir = GT.field_dir;  G.station0 = 'Sky';  G.axis = GT.axis;  G.pupil = GT.pupil;
    G.name = [GT.name '_' GD.form];
    % the per-field launch the bundle and the engine scorer need: [launch
    % point, direction, ok] for a field angle along the slit at a wavelength,
    % the chief through the grating vertex.  Defined BEFORE the other handles:
    % a handle captures the struct as it is at assignment, so chain_bundle
    % only sees launch_field if it exists by then (cost one cycle, 2026-10-01)
    G.launch_field = @(thx, lam) launch_(S, GT, G, thx, lam);
    G.trace = @(p0, d, lam) chain_trace(S, p0, d, lam, G);
    G.aim_pt = @(d, lam) chain_aim(S, GT.launch, d, G.iStop, lam, G);
    G.bundle = @(varargin) chain_bundle(G, varargin{:});
    G.footprints = @(varargin) chain_footprints(G, chain_bundle(G, varargin{:}));
end

function [p0, d, ok] = launch_(S, GT, G, thx, lam)
    % seed the grating aim with the telescope's own aim (the chief through
    % M2's vertex lands on the slit and heads near the grating vertex); the
    % straight line from the launch plane to the grating is no guess at all
    d = GT.field_dir(thx);
    pg = GT.aim_pt(d, lam);
    [p0, ok] = chain_aim(S, GT.launch, d, G.iStop, lam, G, pg);
end
