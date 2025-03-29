%function [PIX,OPD,OPDMask,SPOT,WFE,c,USER] = ...
function [PIX,CE,OPD,OPDMask,SPOT,WFE,c,metMeas,USER] = ...
   call_GMI(prb,pzern,pgrid,pdm,pfa,prad,pimg,InfFcnZern,InfFcnGrid,param,winfil)

if nargin < 11, winfil = 0; end

fname = param.Rx;

if isfield(param,'mgrid'),
  mgrid=param.mgrid;
else,
  mgrid=99;  % default value
end;

pflg(1)  = param.ifFEX(1);
pflg(2)  = param.ifPupilImg;
pflg(3)  = param.cGrid;
pflg(4)  = param.cPix;
pflg(5)  = param.DMlim;
pflg(6)  = param.ifOPD;
pflg(7)  = param.ifShotNoise;
pflg(8)  = param.sigReadNoise;
pflg(9)  = param.sigJitterX;
pflg(10) = param.sigJitterY;
pflg(11) = param.sigCrosstalk;
pflg(12) = param.StartSeed;
pflg(13) = param.transMaskThreshold;
pflg(14) = param.rotMaskThreshold;
pflg(15) = param.pixelSize;
pflg(16) = param.mzern;
pflg(17) = param.QE;
pflg(18) = param.DBias;
if isfield(param,'wlens'), pflg(19) = param.wlens;
else		           pflg(19) = 0;
end
if isfield(param,'ifPIX'), pflg(20) = param.ifPIX;
else		           pflg(20) = 1;
end;

if isfield(param,'ifRetRefSrf'), pflg(21) = param.ifRetRefSrf;
else		                 pflg(21) = 1;
end;
if isfield(param,'ifSPOT'), pflg(22) = param.ifSPOT;
else		            pflg(22) = 0; % default to no SPOT cmd
end;
if isfield(param,'ifPIXflip'), pflg(23) = param.ifPIXflip;
else		               pflg(23) = 0;
end;
if isfield(param,'ifPIXSpotDetCheck'), pflg(24) = param.ifPIXSpotDetCheck;
else		                       pflg(24) = 0;
end;

% Added by jzlou
if isfield(param,'ifSysCalib'), pflg(25) = param.ifSysCalib;
else,                           pflg(25) = 0; % no optimization;
end;

pflg(26) = param.ifPIXElt;

if isfield(param,'ifMetCalc'),
  pflg(27) = param.ifMetCalc;
else,
  pflg(27) = 0;
end;

if isfield(param,'ifSpfCalc'),
  pflg(28) = param.ifSpfCalc;
else,
  pflg(28) = 0;
end;

if isfield(param,'ifRetUserSrf'),
  pflg(29)=param.ifRetUserSrf;
else,
  pflg(29)=0;  % default, no user surface info returned
end;

ipflg = 30;

% ----------------------------------------------------
% Set up STOP
% ----------------------------------------------------
for i=1:4,
  ipflg       = ipflg+1;
  if isfield(param,'STOP'),
    pflg(ipflg) = param.STOP(i);
  else,
    if i==1,
     %pflg(ipflg) = -1d22;
      pflg(ipflg) = -9999;
    else,
      pflg(ipflg) = 0d0;  % STOP undefined
    end;
    %pflg(ipflg) = 0d0;  % STOP undefined
  end;
end

fprintf(1,'**call_GMI: after STOP: ipflg = %i\n', ipflg);

% ----------------------------------------------------
% Set up iFSM
% ----------------------------------------------------
if isfield(param,'iFSM'),
  ipflg = ipflg+1;  pflg(ipflg) = length(param.iFSM);
  for i=1:length(param.iFSM), 
    ipflg = ipflg+1;  pflg(ipflg) = param.iFSM(i); 
  end
  for i=1:length(param.TFSM),  
    ipflg = ipflg+1;  pflg(ipflg) = param.TFSM(i); 
  end
else
  ipflg = ipflg+1; pflg(ipflg) = 9999;
end;

% ----------------------------------------------------
% Set up iFDP
% ----------------------------------------------------
if isfield(param,'iFDP'),
  ipflg = ipflg+1; pflg(ipflg) = param.iFDP
else
  ipflg = ipflg+1; pflg(ipflg) = 9999;
end;

% ----------------------------------------------------
% Set up gridSrf information in pflg
% ----------------------------------------------------
if isfield(param,'gridSrf'),
 ipflg = ipflg+1;  pflg(ipflg) = size(param.gridSrf,1); % # of gridSrfs
 ipflg = ipflg+1;  pflg(ipflg) = size(param.gridSrf,2); % # of passes of gridSrfs
 for i=1:size(param.gridSrf,1)*size(param.gridSrf,2)
   ipflg = ipflg+1;  pflg(ipflg) = param.gridSrf(i);
 end;
 if (length(pgrid) ~= size(param.gridSrf,1)*size(InfFcnGrid,1)*size(InfFcnGrid,2)) & (pgrid~=0)
   disp('gridSrf error'), return
 end;
else,
  ipflg = ipflg+1;  pflg(ipflg) = 9999; 
end;

% ----------------------------------------------------
% Set up zernSrf information in pflg
% ----------------------------------------------------
if isfield(param,'zernSrf'),
 ipflg = ipflg+1;  pflg(ipflg) = size(param.zernSrf,1); % # of zernSrfs
 ipflg = ipflg+1;  pflg(ipflg) = size(param.zernSrf,2); % # of passes of zernSrfs
 for i=1:size(param.zernSrf,1)*size(param.zernSrf,2)
   ipflg = ipflg+1;  pflg(ipflg) = param.zernSrf(i);
 end
 if ~((length(pzern) == size(param.zernSrf,1)*param.mzern) | (length(pzern)==1 & (pzern==0)))
    disp('zernSrf error'), return
 end
else,
 ipflg = ipflg+1;  pflg(ipflg) = 9999;
end;

% ----------------------------------------------------
% Set up dmSrf information in pflg
% ----------------------------------------------------
if isfield(param,'dmSrf'),
 ipflg = ipflg+1;  pflg(ipflg) = size(param.dmSrf,1); % # of dmSrfs
 ipflg = ipflg+1;  pflg(ipflg) = size(param.dmSrf,2); % # of passes of dmSrfs
 for i=1:size(param.dmSrf,1)*size(param.dmSrf,2)
   ipflg = ipflg+1;  pflg(ipflg) = param.dmSrf(i);
 end
 %if ~((length(pdm) == size(param.dmSrf,1)) | (length(pdm)==1 & (pdm==0)))
 %   disp('dmSrf error'), return
 %end
else,
  ipflg = ipflg+1;  pflg(ipflg) = 9999;
end;

% ----------------------------------------------------
% Set up rbSrf information in pflg
% ----------------------------------------------------
if isfield(param,'rbSrf'),
 ipflg = ipflg+1;  pflg(ipflg) = size(param.rbSrf,1); % # of rbSrfs
 ipflg = ipflg+1;  pflg(ipflg) = size(param.rbSrf,2); % # of passes of rbSrfs
 for i=1:size(param.rbSrf,1)*size(param.rbSrf,2)
   ipflg = ipflg+1;  pflg(ipflg) = param.rbSrf(i);
 end
 %if (length(prb) ~= size(param.rbSrf,1)*size(InfFcnGrid,1)*size(InfFcnGrid,2)) & (prb~=0)
 %   disp('rbSrf error'), return
 %end
else,
  ipflg = ipflg+1;  pflg(ipflg) = 9999;
end;

% ----------------------------------------------------
% Set up RptElt for surfaces requiring different Rpt's
% ----------------------------------------------------
if isfield(param,'RptSrf'),
 ipflg = ipflg+1;  pflg(ipflg) = size(param.RptSrf,1); % # of passes of rbSrfs
 ipflg = ipflg+1;  pflg(ipflg) = size(param.RptSrf,2); % # of RptSrfs
 for i=1:size(param.RptSrf,1)*size(param.RptSrf,2)
   ipflg = ipflg+1;  pflg(ipflg) = param.RptSrf(i);
 end
 if size(param.RptSrf,1)*3 ~= length(param.RptElt)
   disp('RptElt is not sized properly'), return
 end
 for i=1:size(param.RptSrf,1)*3
   ipflg = ipflg+1;  pflg(ipflg) = param.RptElt(i);
 end
else,
  ipflg = ipflg+1;  pflg(ipflg) = 9999;
end;

if isfield(param,'ifFEX'),
 if (param.ifFEX(1)==2)
   for i=2:8
      ipflg = ipflg+1;  pflg(ipflg) = param.ifFEX(i);
   end
 end
end;

if isfield(param,'RefSurfs')
   ipflg = ipflg+1;  pflg(ipflg) = length(param.RefSurfs); % # of Ref surfaces
   for i=1:length(param.RefSurfs)
      ipflg = ipflg+1;  pflg(ipflg) = param.RefSurfs(i);
   end
else
   ipflg = ipflg+1;  pflg(ipflg) = 0;
end

if isfield(param,'INTsrf')
   ipflg = ipflg+1;  pflg(ipflg) = length(param.INTsrf); % # of INTsrf's
   for i=1:length(param.INTsrf)
      ipflg = ipflg+1;  pflg(ipflg) = param.INTsrf(i);
   end
else
   ipflg = ipflg+1;  pflg(ipflg) = 0;
end

if isfield(param,'nProc')==0,
  param.nProc=1;  
end;

if (length(pflg)>2000), disp('!!!!!  PFLG TOO BIG!!!!!'); return, end

if winfil
   fid = fopen('infil','w');
   fprintf(fid,'%s\n',param.Rx);
   fprintf(fid,'%d\n',length(pflg));
   for i=1:length(pflg)
      fprintf(fid,'%g\n',pflg(i));
   end
   fclose(fid);
end

if 1,
  [PIX,ER,EI,OPD,OPDMask,SPOT,WFE,c,metMeas,USER] = ...
     GMI(prb,pzern,pgrid,pdm,pfa,prad,pimg,pflg,InfFcnZern,InfFcnGrid,...
         fname,param.mdttl,param.nProc);
     CE = complex(ER,EI);
     %GMI_mex(prb,pzern,pgrid,pdm,pfa,prad,pimg,pflg,InfFcnZern,InfFcnGrid,...
else
  % old version
  [OPD,WFE,USER] = ...
     GMI(prb,pzern,pgrid,pdm,pfa,prad,pimg,pflg,InfFcnZern,InfFcnGrid,...
         fname,param.mdttl);
     !GMI_mex(prb,pzern,pgrid,pdm,pfa,prad,pimg,pflg,InfFcnZern,InfFcnGrid,...
  PIX=[];
  SPOT=[];
  c=[];
end;

return
