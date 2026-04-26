%% MACOS/GMI Parameter Initialization File
%
% Defines all parameters required to successfully run a script that uses
% the GMI interface to MACOS. Parameters are entered in a parameter
% structure.
%--------------------------------------------------------------------------
%%% Set GMI Parameters
%
% * * param.mdttl:* MACOS size
% * * param.STOP:* 1st param=0 -> OBJECT followed by StopVec, 1st param=1
%                  -> ELEMENT folowed by iSTOP, dxoffset, and dyoffset
% * *param.iFSM:* Fast steering mirror element.
% * *param.TFSM:* [*_Transpose_*(FSMTElt(1:3,1)) 0dd 0d0 0d0
%                 [*_Transpose_*(FMSTElt(1:3,2)) 0d0 0d0 0d0]';
% * *param.gridSrf:* List of surfaces which accept shape deformation maps,
%                    Negative numbers are placeholders for surfs in Rx that
%                    have GridMat defined, but don't want the surf changed
%                    to 8 or 9 (because of NS problem).
% * *pram.rbSrf:* rbSrf describes all the surfaces that will have element
%                 rb perturbations applied. Negative numbers check for
%                 MaskThreshold, otherwise no check. Las row defines Global
%                 (0) or Element (1) coordinates for perturbations.
% * *param.Rx:* MACOS optical prescription
% * *param.zernSrf:* describes all the surfaces that will have zernikes
%                    applied to them. ZernSrf and gridSrf cannot be both
%                    defined, doing so will drive GMI to crash. Thus user
%                    may either apply zernikes via gridSrf or zernSrf not
%                    both.
% * *param.dmSrf:* describes all the surfaces that will have pdm applied to
%                  them, surface deformed mirror by poking surface
%                  actuators.
% * *param.RptSrf:* describes surfaces for which the RptElt will change
% * *param.RptElt:* contains the RptSrf changes (perturbation)
% * *param.ifSysCalib:* system optimizer flag, if set to 1 GMI will run
%                       optimization as set up. Optimization parameters are
%                       defined at the MACOS Rx level.
% * *param.ifFEX:* FEX (find exit pupil command) flag, if set to 1 the FEX                     
%                  command will be executed by GMI.
%
%--------------------------------------------------------------------------
% Changes for FreeForm surfaces in GMI.F (SUBROUTINE GMI_DVR):
%  -New args pmonzern, ifpmonzern on the GMI_DVR signature
%  -New locals nmonzern, imonzernSrf, jmonzernSrf, monzernSrf(mprb), 
%   lmonzernSrf(mElt)
%  -ExtractFlagParameters now parses a monzernSrf block from pflg 
%   (immediately after the ifFEX Exit Pupil block) using the same 
%   9999-sentinel convention as gridSrf/zernSrf/dmSrf
%  -ApplyPerturbationToOpticalSystem has a new MonZernCoef block right 
%   after the Zernike block (lines 940-979), mirroring its structure: 
%   writes MonZernCoef(iNode, iElt) = pmonzern(k) for iNode = 4..nmonzern+3 
%   (skipping piston/tip/tilt) on each monzernSrf(iSeg) element. Does not 
%   change SrfType — caller must already have the target element configured 
%   as FreeForm (14)
% GMIG.F (mex gateway):
%  -NRHS check raised to 14
%  -New pmonzern(mpzern) REAL*8 buffer, Npmonzern, pmonzernP, ifpmonzern
%  -New PRHS(14) validation, pointer fetch, and mxCopyPtrToReal8 copy
%   pmonzern, ifpmonzern passed through to GMI_DVR
% call_GMI.m:
%  -Encodes param.monzernSrf into pflg (with 9999 sentinel when absent), 
%   placed right after the ifFEX block to match the GMI.F parser's position
%  -Reads optional param.pmonzern (default scalar 0) and passes it as the 
%   14th argument to the mex GMI call
% To use: set param.monzernSrf = [iElt; ...] (matching the param.zernSrf 
%   shape) listing the FreeForm elements to perturb, and param.pmonzern 
%   as the per-mode coefficient vector. The first 3 modes (piston/tip/tilt) 
%   are skipped by the apply loop, matching the existing pzern convention.
%--------------------------------------------------------------------------

param.Rx              = 'ff_pie';       %without .in
param.pimg(1)         = 1e-3;           % wavelength as defined in Rx file
param.pimg(2)         = 1e0;            %Flux as define in Rx file
param.STOP            = [-1.1d20 0 0 0]; % use this number to disable stop cmd
param.iFSM            = [];
param.TFSM            = [];
param.mdttl           = 256;
param.pfa             =[1 0 0];

%% Define Element Array Parameters for Perturbations
%
% Here we define parameter arrays containing elements to be perturbed via
% pgrid, prb, pzern, and/or pdm'
 
% Here we define param.zernSrf, an array parameter that contains elements
% to be perturbed (surface perturbation) via pmonzern. This requires that
% the elements refered to ar FreeForm surfaces. Typically the design
% FFZerns will not be perturbed, but the MonZerns will be perturbed, as
% they are used to express aberrations:
param.monzernSrf      =[1:7;]';

%%% Define param.gridSrf
% 
% Here we define param.gridSrf, an array that contains elements to be
% perturbed (surface perturbation) via pgrid:
%
%             iElt1 iElt2 iElt3 iElt4 iElt5 iElt6 iElt7 
% param.gridSrf=[ 1:7]';
param.gridSrf=[];

%%% Define param.zernSrf
%
% Here we define param.zernSrf, an array parameter that contains elements
% to be perturbed (surface perturbation) via pzern. This may help on
% perturbing freeform or Zernike surfaces (I tried it before but it did not
% work) so this will need to be tested again because we were able to do so
% for AMD in the past.
param.zernSrf= []';

%%% Define param.rbSrf
%
% Here we define param.rbSrf, an array parameter that contains elements to
% be perturbed (rigidbody perturbation) via prb.The last column will be set
% to 0 or 1, 0 means element will be perturbed in global coordinate frame,
% 1 means element will be perturbed in local coordinate frame (TElt is
% coordinate frame of element).
%     Linked: 35 36 37 38 39 40 41     42             43
%             A1 A2 A3 A4 A5 A6 m2 fold_ota_pass2   m3_pass2   ACF  ACF_CF1 ACF_CF2 ACF_CF3 ACF_CF4 ACF_CF5 ACF_CF6     dm    bs_cam_Newport_30Q20BS1_pass2   filter_cam Edmund_doublet_50107 fold1_cam_Newport_40Z40ER2 Detector fold1_inj_Newport_30Z40ER2  TRIPLET_LP bs_inj_Newport_30Q20BS1 bs_cam_Newport_30Q20BS1
% Two columns: col 1 = element numbers, col 2 = frame flag
% (0 = global frame, 1 = local/element frame).  GMI.F's rb-apply loop
% runs DO j=1,jrbSrf-1, so a single-column rbSrf disables all rb perts.
param.rbSrf=[(1:10)'  zeros(10,1)];


%% Set Array Sizes
%
% Here we set up the size of the perturbation arras. There are perturbation
% arrays for rigibody, shape in Zernike modes, shape in grid, shape in
% actuator, conic constant, and radius of curvature. Here we set up the
% most commonly used.
numseg=7;
numSAF=0;
mgrid=256;
mgrid2=mgrid*mgrid;
param.mzern=15;
mpdm= numseg;
mrbSrf= size(param.rbSrf,1);
mprb= mrbSrf*6;
mpgrid=mgrid2*size(param.gridSrf,1);
mpzern=size(param.zernSrf,1)*param.mzern;
mprad=0;

%% Define Other Parameters
param.nProc = 1;
pram.pfa = 0;
param.dmSrf = [];
param.RptSrf = [];
param.RptElt =[];
param.ifSysCalib = 0;
param.ifFEX = 0;
param.ifPupilImg = 0;
param.cGrid = 256;
param.cPix = param.mdttl;
param.DMlim = 10d0;
%param.ifSPOT = 13;
param.ifOPD =12;
param.ifPIXElt = 13;
param.ifPIX = 0;
param.ifMetCalc =0;
param.ifShotNoise = 0;
param.sigReadNoise = 0;
param.sigJitterX =0;
param.sigJitterY = 0;
param.sigCrosstalk =0;
param.StartSeed =0;
param.transMaskThreshold = 1e22;
param.rotMaskThreshold =1e22;
param.pixelSize =1.098471E-02;
param.QE = 1.d0;
param.DBias = 0.d0;

InfFcnZern= 1e-3*[0;0;0;0;1;0;0;0;0.1;0;0;0;0;0;0];
InfFcnGrid = zeros(256);

%% MACOS Rx
%
% MACOS Rx does not include the .in file type.
Rx = strvcat({'ff_pie'});


%% None GMI parameters
%
%
ifPlot = 'On';
rbEltList={'Seg1','Seg2','Seg3','Seg4','Seg5','Seg6','Seg7','SM','Lens1','Lens2'};

pgridEltList={}; 

%%% Segments and M2 Zernike Sensitivities
pgridSegM2EltSens= {'Seg1','Seg2','Seg3','Seg4','Seg5','Seg6','Seg7','SM'};

%%% Parameters for Zernike Sensitivities
%
%
%pgridBackEndEltSens= {'fold_ota','m3','dm'};

%%% Parameters for rigidbody sensitivities
%
%
rbEltSens= {'Seg1','Seg2','Seg3','Seg4','Seg5','Seg6','Seg7','SM','Lens1','Lens2'};


%%% Parameters for ROC and KC Sensitivities
%
%
rocEltSens = {'Seg1','Seg2','Seg3','Seg4','Seg5','Seg6','Seg7'};
kcEltSens = {};

%%% List of Freeform Optics
%
%
FFEltSens = {'ielt1','ielt2','ielt3','ielt4','ielt5','ielt6','ielt7'};

%%% Parameters for asbult
%
%
psurfElt = {};

























