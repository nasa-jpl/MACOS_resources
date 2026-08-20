function X = oi_seed(P)
%OI_SEED  First-order seed design for the offset_imager template.
%
%   X = OI_SEED(P) builds the starting design struct from the parameter
%   set alone: radii from the first-order seed solve (OI_PARAXIAL --
%   EFL = P.EFL_m exactly, Petzval = 0 for the flat wide field, M1
%   radius = P.seed_R1_m as the free curvature budget; P.bfd_m instead
%   swaps the M1 seed for a back-focus requirement), spherical surfaces
%   (K = 0, no aspheres -- S1 opens them), no tilts/decenters, stop on
%   axis.  The FP pose comes from the first OI_CLOSE call.
%
%   See also OI_PARAXIAL, OI_CLOSE, OFFSET_IMAGER.

    tnet = [P.spacings_m(1) + P.spacings_m(2), P.spacings_m(3)];
    req  = struct('EFL_m', P.EFL_m);
    if ~isempty(P.bfd_m), req.BFD_m = P.bfd_m; end
    fo = oi_paraxial(P.seed_R1_m, tnet, req);

    X = struct();
    X.R        = fo.R;
    X.K        = [0 0 0];
    X.asph     = zeros(3,3);
    X.zern     = {[],[],[]};
    X.yde      = [0 0 0];
    X.ade      = [0 0 0];
    X.z_m1     = P.z_m1_m;
    X.spacings = P.spacings_m;
    X.stopC    = [0; 0; P.z_m1_m + P.spacings_m(1)];
    X.fpa      = struct('Vpt',[0;0;0], 'psi',[0;0;1]);   % posed by OI_CLOSE
    X.fpa_refit = [0 0];
    X.EPD_m    = P.EPD_m;
    X.WL_m     = P.lambda_m;
    X.sampling = P.sampling;
    X.name     = P.name;
end
