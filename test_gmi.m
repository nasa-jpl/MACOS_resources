% 
% A script running GMI interface to macos_f90
% for Optiix optical prescription. 
% John Z. Lou, Jet Propulsion Laboratory
% Last modified: Feb 21, 2012
% 

% Calc dwdx and Optiix 

clear mex; clear all;

% ------------------------------------------------------------------------------
% Initialize parameters
% ------------------------------------------------------------------------------
optiixInit_jzlou;  % specify Rx and other parameters 
param.Rx='optiixonaxisz1_v4_pmsm_met';
 
prb = zeros(mprb,1);
pzern = zeros(mpzern,1);
pgrid = zeros(mpgrid,1); %1e-4*rand(mpgrid,1); %zeros(mpgrid,1);

%pdm = 0; %zeros(mpdm,1);
%param.pfa = [1 0 -0.72*pi/180];
param.pfa=0; 
param.pimg(1)=5d-04; % WL
param.pimg(2)=1d0;
 

% ------------------------------------------------------------------------------
% Apply perturbations - example
% ------------------------------------------------------------------------------

%prb(2*6+1)=5d-4;  % x-tilt (global frame) M5 by 1 urad

% Grid surface(s) 
pgrid=zeros(mgrid2,1);

% ------------------------------------------------------------------------------
% Call model and plot results
% ------------------------------------------------------------------------------

% Call GMI interface
param.ifOPD=17; % OPD element
param.ifPIX=0;  % this turns on or off PIX operation
param.ifPIXElt=18; % this specifies PIX element
param.ifMetCalc=1;
param.nProc=1;

% Nominal case 
if 1,
 %[PIX,OPDnom,OPDnomMask,SPOT,WFE,c,R] ...
  [PIX,CEFnom,OPDnom,OPDnomMask,SPOT,WFE,c,metMeasNom,R] ...
       = call_GMI(prb,0,0,0,0,0, ...
                  param.pimg,InfFcnZern,InfFcnGrid,param);
  [k2ij(:,1),k2ij(:,2)]=find(OPDnom); k2ij=k2ij';
  wnom=OPD2w(OPDnom,k2ij);
  figure; imagesc(OPDnom); colorbar;
  %save wnom_optiix_v4 wnom OPDnom OPDnomMask metMeasNom k2ij;
  return;
else,
  load wnom_optiix_v4;
end;

nseg=6; 

% Calc 1-D met beam sensitivity and dwdx
% 6D RB perturbations used in generating sensitivities
da = 2d-6;  % differential tip/tilt angle
dc = 1d-4;  % differential clocking
dt = 1d-5;  % differential x and y translations
dp = 5d-6;  % differential piston

if 1,
 % Generating global WF mask
 OPDMask_g=OPDnomMask;
 for irb=1:mrbSrf,
  for idof=1:6
    rr=mod(idof-1,6);
    if rr<2, dx=da;
    elseif rr==2, dx=dc;
    elseif rr<5, dx=dt;
    else, dx=dp; end;
    prb=zeros(mprb,1);
    prb((irb-1)*6+idof)=dx;
    [PIX,CEF,OPD,OPDMask,SPOT,WFE,c,R]=call_GMI(prb,0,0,0,0,0, ...
                          param.pimg,InfFcnZern,InfFcnGrid,param); 
    OPDMask_g=OPDMask_g.*OPDMask;
    if 0, %0 & idof==6, 
      dOPD=OPD-OPDnom;
      imagesc(dOPD); colorbar; pause; 
    end;
  end;
 end;
 %
  save OPDMask_g OPDMask_g;
  return;
else,
  load OPDMask_g;
end;

% Filter OPDnom and wnom with global OPD mask 'OPDMask_g' 
OPDnom=OPDnom.*OPDMask_g;
wnom=OPD2w(OPDnom,k2ij);

%
% Now compute dwdx, and mask all OPD with OPDMask_g 
%
%dwdx=zeros(size(wnom,1),nseg*6);
dwdx=zeros(size(wnom,1),mrbSrf*6); % mrbSrf=(6+5)
%for irb=11:11,
% for idof=2:2,
for irb=1:mrbSrf,
  for idof=1:6,
    rr=mod(idof-1,6);
    if rr<2, dx=da;
    elseif rr==2, dx=dc;
    elseif rr<5, dx=dt;
    else, dx=dp; end;
    prb=zeros(mprb,1);
    prb((irb-1)*6+idof)=dx;
    [PIX,CEF,OPD,OPDMask,SPOT,WFE,c,R] = call_GMI(prb,0,0,0,0,0, ...
                                    param.pimg,InfFcnZern,InfFcnGrid,param);
    OPD=OPD.*OPDMask_g;
    dOPD=OPD-OPDnom;
    %figure; imagesc(dOPD); colorbar;
    w1=OPD2w(OPD,k2ij);
    dwdx(:,(irb-1)*6+idof)=(w1-wnom)/dx; 
    %opdw=w2OPD(dwdx(:,(irb-1)*6+idof),k2ij); imagesc(opdw); pause; 
  end;
end;

if 1,
  % Convert to urad and um for wavefront sensitivity
  dwdx(:,1:6:6*mrbSrf-5)=dwdx(:,1:6:6*mrbSrf-5)/1d3;
  dwdx(:,2:6:6*mrbSrf-4)=dwdx(:,2:6:6*mrbSrf-4)/1d3;
  dwdx(:,3:6:6*mrbSrf-3)=dwdx(:,3:6:6*mrbSrf-3)/1d3;
  wnom=wnom*1d3; % mm to um
end;
elt_names=[' A1-A6  ', '  SM    ', '  TM    ', '  FSM   ', ' Dummy  ' 'Detector'];

save optiix_v4_dwdx_11elts_urad_um dwdx wnom k2ij elt_names OPDMask_g;

%save optiix_dwdx wnom k2ij dwdx;
%save optiix_dwdx_11elts_urad_um dwdx wnom k2ij elt_names;
return;

if 0,
 figure,imagesc(OPD); colorbar; title('Optiix nominal OPD');
 rmswf=rms2(nonzeros(OPD(:)));
 xlabel(['RMS = ' num2str(rmswf*1d6) ' nm']);
 %figure,imagesc(PIX); colorbar; title('ecco PIX');
end;

