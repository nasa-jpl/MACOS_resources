%% MACOS/GMI Parameter Initialization File
%
% Defines all parameters required to successfully run a script that uses
% the GMI interface to MACOS. Parameters are entered in a parameter
% structure.

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

param.Rx              = 'iris_dp_v14';%without .in
param.pimg(1)         = 8.500000000000000D-4; % wavelength as defined in Rx file
param.pimg(2)         = 1e0; %Flux as define in Rx file
param.STOP            = [-1.1d20 0 0 0]; % use this number to disable stop cmd
param.iFSM            = [];
param.TFSM            = [];
param.mdttl           = 256;
param.pfa             =[1 0 0];
%% Define Element Array Parameters for Perturbations
%
% Here we define parameter arrays containing elements to be perturbed via
% pgrid, prb, pzern, and/or pdm'
% 

%%% Define param.gridSrf
% 
% Here we define param.gridSrf, an array that contains elements to be
% perturbed (surface perturbation) via pgrid. Some elements know as
% freeform or Zernike elements can not be perturbed via pgrid so they will
% be perturbed separate via pzern if possible, otherwise they will be
% perturbed via Matlab RxEditor. We can define for single pass or double
% pass system, on double pass an element may be seen twice one each pass
% and gridSrf will be set as illustrated int he following example:
%
%             iElt1 iElt2 iElt3 iElt4 iElt5 iElt6 iElt7 iElt8 iElt9 iElt10
% param.gridSrf=[ 9    10    11    12    13    14    15     6     5     4
%                30    31    32    33    34    35    36    38    39    40]';

%               A1 A2 A3 A4 A5 A6 m2 fold_ota  m3  dm  
% param.gridSrf=[ 17 18 19 20 21 22 14    13     12  11   
%                 35 36 37 38 39 40 42    43     44  45 ]';           


%               ACF_CF1 ACF_CF2 ACF_CF3 ACF_CF4 ACF_CF5 ACF_CF6   bs_cam_Newport_30Q20BS1 filter_cam_s1 filter_cam_s2  Edmund_doublet_50107_s1  Edmund_doublet_50107_s2 doublet_s3 doublet_s4  fold1_cam_Newport_40Z40ER2  fold1_inj_Newport_30Z40ER2 TRIPLET_LP_3_REFOCUSED_s1  TRIPLET_LP_3_REFOCUSED_s2 TRIPLET_LP_2_s1  TRIPLET_LP_2_s2  TRIPLET_LP_1_s1 TRIPLET_LP_1_s2  bs_inj_Newport_30Q20BS1 bs_cam_Newport_30Q20BS1_pass1_s2  bs_cam_Newport_30Q20BS1_pass1_s1
param.gridSrf=[    25      26      27      28      29      30                  46               47            48               49                        50                 51         52                    53                      1                         2                          3                         4                5                6               7                8                     9                                    10]';           


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
param.rbSrf=[ 17 18 19 20 21 22 14     13             12        25     26      27      28      29      20      31       11             46                         47             49                     53                 54             1                       2              8                            9                 
               1  1  1  1  1  1  1      1              1         0      1       1       1       1       1       1        1              1                          1              1                      1                  1             1                       1              1                            1]';



%% Set Array Sizes
%
% Here we set up the size of the perturbation arras. There are perturbation
% arrays for rigibody, shape in Zernike modes, shape in grid, shape in
% actuator, conic constant, and radius of curvature. Here we set up the
% most commonly used.
numseg=6;
numSAF=6;
mgrid=99;
mgrid2=mgrid*mgrid;
param.mzern=45;
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
%param.ifSPOT = 54;
param.ifOPD =55;
param.ifPIXElt = 55;
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
param.pixelSize =2*1.672D-02;
param.QE = 1.d0;
param.DBias = 0.d0;

InfFcnZern= 1e-3*[0;0;0;0;1;0;0;0;0.1;0;0;0;0;0;0];
InfFcnGrid = zeros(99);

%% MACOS Rx
%
% MACOS Rx does not include the .in file type.
Rx = strvcat({'iris_dp_v14'});


%% None GMI parameters
%
%
ifPlot = 'On';
rbEltList={'A1','A2','A3','A4','A5','A6','m2','fold_ota','m3','ACF','ACF_CF1','ACF_CF2','ACF_CF3','ACF_CF4','ACF_CF5','ACF_CF6','dm','bs_cam_Newport_30Q20BS1','filter_cam Edmund_doublet_50107','fold1_cam_Newport_40Z40ER2','Detector','fold1_inj_Newport_30Z40ER2','TRIPLET_LP bs_inj_Newport_30Q20BS1','bs_cam_Newport_30Q20BS1'};

pgridEltList={'ACF_CF1','ACF_CF2','ACF_CF3','ACF_CF4','ACF_CF5','ACF_CF6','bs_cam_Newport_30Q20BS1','filter_cam_s1','filter_cam_s2','Edmund_doublet_50107_s1','Edmund_doublet_50107_s2','doublet_s3','doublet_s4','fold1_cam_Newport_40Z40ER2','fold1_inj_Newport_30Z40ER2','TRIPLET_LP_3_REFOCUSED_s1','TRIPLET_LP_3_REFOCUSED_s2','TRIPLET_LP_2_s1','TRIPLET_LP_2_s2','TRIPLET_LP_1_s1','TRIPLET_LP_1_s2','bs_inj_Newport_30Q20BS1','bs_cam_Newport_30Q20BS1_pass1_s2','bs_cam_Newport_30Q20BS1_pass1_s1'}; 

%%% Segments and M2 Zernike Sensitivities
pgridSegM2EltSens= {'A1','A2','A3','A4','A5','A6','m2'};

%%% Parameters for Zernike Sensitivities
%
%
%pgridBackEndEltSens= {'fold_ota','m3','dm'};
pgridBackEndEltSens= {'ACF_CF1','ACF_CF2','ACF_CF3','ACF_CF4','ACF_CF5','ACF_CF6','bs_cam_Newport_30Q20BS1','filter_cam_s1','filter_cam_s2','Edmund_doublet_50107_s1','Edmund_doublet_50107_s2','doublet_s3','doublet_s4','fold1_cam_Newport_40Z40ER2','fold1_inj_Newport_30Z40ER2','TRIPLET_LP_3_REFOCUSED_s1','TRIPLET_LP_3_REFOCUSED_s2','TRIPLET_LP_2_s1','TRIPLET_LP_2_s2','TRIPLET_LP_1_s1','TRIPLET_LP_1_s2','bs_inj_Newport_30Q20BS1','bs_cam_Newport_30Q20BS1_pass1_s2','bs_cam_Newport_30Q20BS1_pass1_s1'};

%%% Parameters for rigidbody sensitivities
%
%
rbEltSens= {'A1','A2','A3','A4','A5','A6','m2','fold_ota','m3','ACF','ACF_CF1','ACF_CF2','ACF_CF3','ACF_CF4','ACF_CF5','ACF_CF6','dm','bs_cam_Newport_30Q20BS1','filter_cam Edmund_doublet_50107','fold1_cam_Newport_40Z40ER2','Detector','fold1_inj_Newport_30Z40ER2','TRIPLET_LP bs_inj_Newport_30Q20BS1','bs_cam_Newport_30Q20BS1'};


%%% Parameters for ROC and KC Sensitivities
%
%
rocEltSens = {'A1','A2','A3','A4','A5','A6'};
kcEltSens = {'A1','A2','A3','A4','A5','A6'};

%%% List of Freeform Optics
%
%
zEltSens = {'ielt1','ielt2','ielt3','ielt4','ielt5'};

%%% Parameters for asbult
%
%
psurfElt = {'ielt1','ielt2','ielt3','ielt4','ielt5'};

























