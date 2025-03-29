%***********************************************************************
%    +----------------------------------------------------------------+
%    |  Copyright (C) 1998-2008, California Institute of Technology.  |
%    |  U.S. Government Sponsorship is acknowledged.		      |
%    +----------------------------------------------------------------+
%***********************************************************************
%
%  Set array sizes
%
    numseg             = 6;
    numSAF             = 0;
    mgrid              = 99; %256;
    mgrid2             = mgrid*mgrid;
    param.mzern        = 12;
    mpdm               = 90*numseg;
    mpdm2              = 350;
  
    mrbSrf             = 6+5+1; 
    mprb               = mrbSrf*6; 
	                          
    mpgrid             = mgrid2*numseg;
    mpzern             = (numseg+numSAF)*param.mzern;

    %param.Rx = 'KentD_v2';   % prescription file name, with local frames, could be overritten
	 	              % in main script
    param.Rx = 'KentD_v2_pmsm_met';

    %param.mdttl = 256;   % Macos size
    param.mdttl = 512;   % Macos size

    param.STOP = [0 0 0 0];  % 1st param=0 -> OBJECT, followed by StopVec
		% 1st param=1 -> ELEMENT, followed by iSTOP, dxoffset, dyoffset

    param.iFSM = []; % Fast steering mirror element
%   param.TFSM=[ 0.000000000000505   0.000000000001223   1d0                  0d0 0d0 0d0
%               -0.713105808382103  -0.701056421432790   0.000000000001218    0d0 0d0 0d0]';
%   % make sure components of TFSM are consistent with what is in Rx!
    param.TFSM=[];

    % gridSrf describes all the surfaces that will have gridded data
    % applied to them.  Also, pdm may be applied to these surfaces.
    % Negative numbers are placeholders for surfs in Rx that have GridMat
    % defined, but don't want the surf changed to 8 or 9 (because of NS problem)
    % *** not updated to current Rx yet !!! -jzlou
    %param.gridSrf = [9, 11:13, 19,  20
    %                30, 26:28, -1,  -1]';

    % rbSrf describes all the surfs that will have element rb perts applied to.
    % Negative numbers check for MaskThreshold, otherwise no check.
    % Last row defines Global (0) or Element (1) coordinates for perturbations.

    %param.rbSrf = [2   3    5    6    7    8   9   10   
    %              ones(1,mrbSrf)]';  % 1 when using local frames
    %              zeros(1,mrbSrf)]'; % 0 when using global frames
    %              1   1    1    1    0    0   0    0]';

     param.rbSrf = [   4:1:9, 11, 12 13, 14, 15, 16
                   ones(1,6),  0,  0  0,  0,  0,  0]';  % note that Elt 12 is for metrology 
						        % sensitivity only; -jzlou

    param.gridSrf = [param.rbSrf(1,1)]';

    % zernSrf describes all the surfaces that will have Zernikes applied to them
    % *** not updated to current Rx yet !!! -jzlou
    %param.zernSrf = [11:13,  9,  19,  20
    %                26:28, 30,   0,   0]';
    param.zernSrf=[];

    % dmSrf describes all the surfaces that will have pdm applied to them
    % dmSrf must overlap either gridSrf or zernSrf.  gridSrf supercedes zernSrf.
     param.dmSrf = []';


if 0, % if activated, this would require all 0's or 1's in last row of param.rbSrf
 if isfield(param,'rbSrf'),
    if length(unique(param.rbSrf(:,size(param.rbSrf,2)))) == 1 & ...
              unique(param.rbSrf(:,size(param.rbSrf,2)))  == 1
        frm_label = 'local';
    elseif length(unique(param.rbSrf(:,size(param.rbSrf,2)))) == 1 & ...
                  unique(param.rbSrf(:,size(param.rbSrf,2)))  == 0
        frm_label = 'global';
    else
        fprintf('Error! - local or global frame label not specified! \n');
        pause
    end
 end;
end;

    % RptSrf describes surfaces for which the RptElt will change.
    % RptElt contains the changes
    param.RptSrf = []';
    param.RptElt = [];

% These are the traditional pflg values
    param.ifFEX              = 0;
    param.ifPupilImg         = 0;

    param.cGrid              = 256;
    param.cPix               = param.mdttl;

    param.DMlim             = 10d0;
    param.ifOPD             = 17; % for ecco3.in
    param.ifPIXElt          = 18;

  if 0,
    idx_tmp = find(param.rbSrf(:,1) == param.ifOPD);
    if isempty(idx_tmp)
        fprintf('OPD is computed at an element not found in purturbed elements! \n');
        pause
    else
        fprintf(['Computing OPD at element ',elmts_rbSrf{idx_tmp},' ...\n']);
        opd_elmt_flg = input('Is this correct? (y or n)', 's');
        if ~strcmp(opd_elmt_flg, 'y')
            fprintf('Check the element where opd is computed! \n');
            pause
        end
    end
  end;

    param.pimg		     = 0;
    param.ifShotNoise        = 0;
    param.sigReadNoise       = 0;
    param.sigJitterX         = 0;
    param.sigJitterY         = 0;
    param.sigCrosstalk       = 0;
    param.StartSeed          = 0;
    param.transMaskThreshold = 1d22;
    param.rotMaskThreshold   = 1d22;
    param.pixelSize          = 1.672D-02;
    param.QE                 = 1.d0;
    param.DBias              = 0.d0;


% ------------------------------------------------------------------------------
%  set influence functions
% ------------------------------------------------------------------------------

    InfFcnZern = 1e-3*[0;0;0;0;1;0;0;0;0.1;0;0;0;0;0;0];

    InfFcnGrid = zeros(99);

% ------------------------------------------------------------------------------
%  Miscellaneous flags
% ------------------------------------------------------------------------------

    IF_PLOT = 1;
    OPD_Fig  = 1;
    dOPD_Fig = 2;

    FileName_WFmat     = sprintf('%s_%i_mat',param.Rx,param.mdttl);
    FileName_Control   = sprintf('%s_%i_Control',param.Rx,param.mdttl);
