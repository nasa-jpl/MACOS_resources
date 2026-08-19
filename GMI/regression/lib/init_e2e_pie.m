function [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_e2e_pie()
% INIT_E2E_PIE  Reference param struct for the e2e_pie Rx (regression).
%
% e2e_pie = the 7-segment pie-PM three-mirror telescope emitted by the
% mmacos design pipeline (templates/80_end_to_end/e2e).  17 elements:
%   Elts  1..7  : FreeForm PM Segments (grid + Mon-Zernike figure)
%   Elts  8..14 : downstream Reflectors (8,10,11,12,14 = Zernike; 9 Flat;
%                 13 Conic)
%   Elts 15..16 : Return (Flat, Conic)  -- Elt 16 = the exit-pupil plane
%   Elt   17    : FocalPlane
% Object-space aperture stop (ApStop= 0 0 0).  4 m, 500 nm, model 512.
% The _met variant carries per-segment metrology, so ifMetCalc=1 here
% exercises the METcalc / MetMeas channel.
%
% Slimmed-down reference initializer -- only the fields call_GMI reads.
% The vectors (prb, pzern, pgrid) are zero-initialized; each test
% overrides whatever channels it exercises.

    numseg            = 7;                 % 7 pie PM segments
    numSAF            = 0;
    mgrid             = 99;
    mgrid2            = mgrid * mgrid;
    param.mzern       = 12;
    mpdm              = 90 * numseg;

    mrbSrf            = numseg + 7;         % 7 segments + 7 downstream mirrors
    mprb              = mrbSrf * 6;
    mpgrid            = mgrid2 * numseg;
    mpzern            = (numseg + numSAF) * param.mzern;

    param.Rx          = 'e2e_pie_met';
    param.mdttl       = 512;
    param.mgrid       = mgrid;

    % rbSrf: the 14 real optics (7 segments + 7 reflectors) in the
    % global frame (column 2 = 0 global / 1 local).  Returns (15,16)
    % and the FocalPlane (17) are not rigid-body channels.
    param.rbSrf       = [(1:14)', zeros(14,1)];
    param.gridSrf     = (1:7)';    % FreeForm PM segments carry the grid figure
    param.zernSrf     = [];        % populated by tests that exercise the Zernike channel
    param.dmSrf       = [];
    param.RptSrf      = [];
    param.RptElt      = [];

    % STOP at object space (0 0 0)
    param.STOP        = [0 0 0 0];
    param.iFSM        = [];
    param.TFSM        = [];

    % pflg scalars
    param.ifFEX               = 0;
    param.ifPupilImg          = 0;
    param.cGrid               = 256;
    param.cPix                = param.mdttl;
    param.DMlim               = 10.0;
    param.ifOPD               = 16;       % OPD reported at Elt 16 (exit-pupil Return)
    param.ifPIX               = 0;
    param.ifPIXElt            = 17;       % FocalPlane
    param.ifMetCalc           = 1;        % _met Rx: exercise the METcalc/MetMeas channel
    param.ifShotNoise         = 0;
    param.sigReadNoise        = 0;
    param.sigJitterX          = 0;
    param.sigJitterY          = 0;
    param.sigCrosstalk        = 0;
    param.StartSeed           = 0;
    param.transMaskThreshold  = 1d22;
    param.rotMaskThreshold    = 1d22;
    param.pixelSize           = 1.672d-2;
    param.QE                  = 1.0;
    param.DBias               = 0.0;
    param.pfa                 = 0;
    param.pimg                = [5d-7, 1d0];   % wavelength (m), flux
    param.nProc               = 1;

    % Influence functions -- defaults matching legacy test_gmi.m
    InfFcnZern        = zeros(45, 1);
    InfFcnZern(1:15)  = 1d-3 * [0;0;0;0;1;0;0;0;0.1;0;0;0;0;0;0];
    InfFcnGrid        = zeros(mgrid, mgrid);

    % Zero perturbation -- scalar 0 disables each channel (call_GMI's
    % "channel disabled" sentinel).  Tests that exercise a channel
    % build their own sized vector and pass it in.
    prb               = zeros(mprb, 1);
    pzern             = 0;
    pgrid             = 0;

end
