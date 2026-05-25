function [param, prb, pzern, pgrid, InfFcnZern, InfFcnGrid] = init_e5hex1()
% INIT_E5HEX1  Reference param struct for Rx_e5hex1 (regression).
%
% e5hex1 = 7 FreeForm hex segments + Zernike reflector + FreeForm
% lens + Conic lens + 2 Return surfaces + FocalPlane.  Object-space
% stop at (0,0,0).  Same Rx the pymacos sensitivity work has been
% hammering on.
%
% wf_elt = nElt-1 = Elt 12 (the EP / Return:Conic surface).
% MonZern channel can exercise the FreeForm segments (Elts 1..7) or
% the FreeForm refractor (Elt 9).

    numseg            = 7;     % 7 hex segments
    numSAF            = 0;
    mgrid             = 99;
    mgrid2            = mgrid * mgrid;
    param.mzern       = 12;
    mpdm              = 90 * numseg;

    mrbSrf            = numseg + 6;       % segments + lens/EP/FP elements
    mprb              = mrbSrf * 6;
    mpgrid            = mgrid2 * numseg;
    mpzern            = (numseg + numSAF) * param.mzern;

    param.Rx          = 'Rx_e5hex1';
    param.mdttl       = 256;
    param.mgrid       = mgrid;

    % rbSrf: actual optics (segments + lens + EP + FP) in global frame.
    param.rbSrf       = [(1:13)', zeros(13,1)];
    param.gridSrf     = [];
    param.zernSrf     = [];     % populated by tests that exercise the Zernike-typed Elt 8
    param.dmSrf       = [];
    param.RptSrf      = [];
    param.RptElt      = [];

    % e5hex1 declares an object-space stop at (0,0,0) via ApStop.
    param.STOP        = [0 0 0 0];
    param.iFSM        = [];
    param.TFSM        = [];

    % pflg scalars
    param.ifFEX               = 0;
    param.ifPupilImg          = 0;
    param.cGrid               = 128;
    param.cPix                = param.mdttl;
    param.DMlim               = 10.0;
    param.ifOPD               = 12;       % OPD at Elt 12 (the EP Return:Conic)
    param.ifPIX               = 0;
    param.ifPIXElt            = 13;
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

    % Influence functions -- same defaults as optiix init.
    InfFcnZern        = zeros(45, 1);
    InfFcnZern(1:15)  = 1d-3 * [0;0;0;0;1;0;0;0;0.1;0;0;0;0;0;0];
    InfFcnGrid        = zeros(mgrid, mgrid);

    prb               = zeros(mprb, 1);
    pzern             = 0;
    pgrid             = 0;

end
