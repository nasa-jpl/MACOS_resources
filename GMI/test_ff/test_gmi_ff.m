%

% GMI test script incorporating FreeForm optics
% Uses ff_pie.in, gmi_Init_ff.m, call_GMI.m, GMI.mexa64, etc.

% ------------------------------------------------------------------------------
% Initialize parameters

clear all

gmi_Init_ff;  % specify Rx and other parameters

prb = zeros(mprb,1);
pzern = zeros(mpzern,1);
pmonzern = zeros(mpzern,1);
pgrid = zeros(mpgrid,1); %1e-4*rand(mpgrid,1); %zeros(mpgrid,1);

%pdm = 0; %zeros(mpdm,1);
%param.pfa = [1 0 -0.72*pi/180];
param.pfa=0;
param.pimg(1)= 1.0e-3; % WL
param.pimg(2)=1d0;

pgrid=zeros(mgrid2,1);

%% ------------------------------------------------------------------------
% Nominal OPD

fprintf('Nominal OPD ');
[PIX,CEFnom,OPDnom,OPDnomMask,SPOT,WFE,c,metMeasNom,R] ...
    = call_GMI(0,0,0,0,0,0, ...
    param.pimg,InfFcnZern,InfFcnGrid,param);
fprintf('\n');
[vec,indx]=m2v(OPDnom);

figure(1), clf
dimage(OPDnom.*1e3,1);   % mm to um
title('Nominal OPD (um)')

nseg=7;

%% ------------------------------------------------------------------------
% Calculate a mask that contains rays from all sensitivities

da = 2d-6;  % differential tip/tilt angle
dc = 1d-4;  % differential clocking
dt = 1d-5;  % differential x and y translations
dp = 5d-6;  % differential piston

% Generate global WF mask valid for all perturbations
if ~isfile('OPDMask_g.mat')
    fprintf('Building global OPD mask ');
    OPDMask_g=OPDnomMask;
    for irb=1:mrbSrf
        for idof=1:6
            rr=mod(idof-1,6);
            if rr<2, dx=da;
            elseif rr==2, dx=dc;
            elseif rr<5, dx=dt;
            else, dx=dp; end
            prb=zeros(mprb,1);
            prb((irb-1)*6+idof)=dx;
            [PIX,CEF,OPD,OPDMask,SPOT,WFE,c,R]=call_GMI(prb,0,0,0,0,0, ...
                param.pimg,InfFcnZern,InfFcnGrid,param);
            OPDMask_g=OPDMask_g.*OPDMask;
            if 0, %0 & idof==6,
                dOPD=OPD-OPDnom;
                imagesc(dOPD); colorbar; pause;
            end
        end
    end
else
    fprintf('Loading global OPD mask (OPDMask_g.mat)');
    load OPDMask_g;
end
fprintf('\n');
prb=zeros(mprb,1);
%
save OPDMask_g OPDMask_g OPDnom;

% Filter OPDnom and wnom with global OPD mask 'OPDMask_g'

OPDnom=OPDnom.*OPDMask_g;
[wnom,ix] = m2v(OPDnom);

figure(2), clf
dimage(OPDMask_g,1);
title('OPD Mask')

%% ------------------------------------------------------------------------
% Compute dwdx, mask all OPD with OPDMask_g, and convert to um and urad

fprintf('Computing dwdx ');
dwdx=zeros(size(wnom,1),mrbSrf*6); % mrbSrf=(6+5)
figure(3), clf
jdof=0;
for irb=1:mrbSrf
    for idof=1:6
        jdof=jdof+1;
        rr=mod(idof-1,6);
        if rr<2, dx=da;
        elseif rr==2, dx=dc;
        elseif rr<5, dx=dt;
        else, dx=dp; end
        prb=zeros(mprb,1);
        prb((irb-1)*6+idof)=dx;
        [PIX,CEF,OPD,OPDMask,SPOT,WFE,c,R] = call_GMI(prb,0,0,0,0,0, ...
            param.pimg,InfFcnZern,InfFcnGrid,param);
        OPD=OPD.*OPDMask_g;
        dOPD=OPD-OPDnom;
        w1=m2v(OPD,ix);
        dwdx(:,(irb-1)*6+idof)=(w1-wnom)/dx;
        figure(3)
        dimage(dOPD,1)
        title(['dOPD, DOF=' num2str(jdof)])
        drawnow
    end
end
fprintf('\n');

% Keep an mm-domain copy of wnom for the dwdz loop below, since the
% next line scales wnom to um in place.
wnom_mm = wnom;

% Convert to urad and um for wavefront sensitivity
dwdx(:,1:6:6*mrbSrf-5)=dwdx(:,1:6:6*mrbSrf-5)/1d3;
dwdx(:,2:6:6*mrbSrf-4)=dwdx(:,2:6:6*mrbSrf-4)/1d3;
dwdx(:,3:6:6*mrbSrf-3)=dwdx(:,3:6:6*mrbSrf-3)/1d3;
wnom=wnom*1d3; % mm to um
elt_names = rbEltSens;

save ff_pie_dwdx_urad_um dwdx wnom ix elt_names OPDMask_g;

%% ------------------------------------------------------------------------
% Now compute surface deformation sensitivities (MonZernCoef on FreeForm
% segments).  pmonzern flows to GMI.F via param.pmonzern (read inside
% call_GMI.m and passed as the 14th mex argument).
%
% GMI.F's apply loop is:
%   DO iNode = 4, nmonzern+3       ! skip piston/tip/tilt
%     MonZernCoef(iNode, iElt) = pmonzern(k); k = k + 1
% so each segment in param.monzernSrf consumes nmonzern values of pmonzern,
% starting at Born&Wolf mode index 4.  nmonzern defaults to param.mzern.

nz      = param.mzern;            % modes per segment
nmzsrf  = length(param.monzernSrf);
mpmzsrf = nz*nmzsrf;

prb_zero = 0;   %zeros(mprb,1);         % no rigid-body perturbations here
dwdz     = zeros(size(wnom,1), mpmzsrf);

fprintf('Computing dwdz ');
figure(4), clf
jdof = 0;
dz   = 1d-4;
for irb = 1:nmzsrf
    for idof = 1:nz
        jdof = jdof + 1;
        pmonzern_pert         = zeros(mpmzsrf,1);
        pmonzern_pert(jdof)   = dz;
        param.pmonzern        = pmonzern_pert;   % consumed by call_GMI
        [PIX,CEF,OPD,OPDMask,SPOT,WFE,c,R] = call_GMI( ...
            prb_zero, 0, 0, 0, 0, 0, ...
            param.pimg, InfFcnZern, InfFcnGrid, param);
        OPD  = OPD .* OPDMask_g;
        dOPD = OPD - OPDnom;
        w1   = m2v(OPD,ix);
        dwdz(:, jdof) = (w1 - wnom_mm) / dz;   % both sides in mm
        figure(4)
        dimage(dOPD,1)
        title(['dOPD, MonZern DOF=' num2str(jdof) ...
               ' (seg ' num2str(irb) ' mode ' num2str(idof+3) ')'])
        drawnow
    end
end
fprintf('\n');
param = rmfield(param, 'pmonzern');   % leave param clean for later calls

% Scale dwdz to um for consistency with wnom (already converted above)
% and with the _urad_um filename convention.
dwdz = dwdz * 1d3;   % mm/dz -> um/dz

elt_names = rbEltSens;

save ff_pie_dwdz_urad_um dwdz wnom ix elt_names OPDMask_g;

     
