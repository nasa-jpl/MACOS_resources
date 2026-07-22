function [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_optiix()
% INIT_OPTIIX  Reference param struct for the Optiix Rx (regression).
%
% Slimmed-down version of optiixInit_jzlou.m -- only the fields
% call_GMI actually reads, no interactive prompts, no plotting setup.
% The vectors (prb, pzern, pgrid) are zero-initialized -- each test
% overrides whatever channels it exercises.

    numseg            = 6;
    numSAF            = 0;
    mgrid             = 99;
    mgrid2            = mgrid * mgrid;
    param.mzern       = 12;
    mpdm              = 90 * numseg;

    mrbSrf            = 6 + 5 + 1;
    mprb              = mrbSrf * 6;
    mpgrid            = mgrid2 * numseg;
    mpzern            = (numseg + numSAF) * param.mzern;

    param.Rx          = 'optiixonaxisz1_v4_pmsm_met';
    param.mdttl       = 512;
    param.mgrid       = mgrid;

    % rbSrf: column 2 = 0 (global coords) / 1 (local coords).
    param.rbSrf       = [(4:9)', zeros(6,1); ...
                         (11:16)', zeros(6,1)];
    param.gridSrf     = [param.rbSrf(1,1)]';
    param.zernSrf     = [];     % populated by tests that exercise Zernike channel
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
    param.ifOPD               = 17;       % OPD reported at Elt 17
    param.ifPIX               = 0;
    param.ifPIXElt            = 18;
    param.ifMetCalc           = 0;
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
    param.pimg                = [5d-4, 1d0];   % wavelength, flux
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
