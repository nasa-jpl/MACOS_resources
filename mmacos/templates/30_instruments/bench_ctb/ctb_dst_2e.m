function OUT = ctb_dst_2e(lsb_nm, nrep, region)
%CTB_DST_2E  DST Lane 2e: DM command quantization (LSB electronics noise).
%   The controller solves in CONTINUOUS strokes against the deep N=1024
%   mono baseline Jacobian; the PLANT is commanded with fresh U(+/-LSB/2)
%   dither per actuator per evaluation (ctb_efc quant_nm) -- DM LSB noise,
%   attributed by Ruane 2020 to the ~3e-11 modulated floor.  No new
%   Jacobian: the controller model is unchanged, only the plant is
%   perturbed.  control == truth otherwise, perfect sensing.
%
%   Sweeps LSB over lsb_nm (default [0.001 0.01 0.1] nm) with nrep=4
%   repeated EFC runs each (different rng seeds -> the DST 4-run idiom),
%   plus an LSB=0 clean run.  GATE: run-to-run RANDOM speckle at a stable
%   floor per LSB; the floor rises with LSB; LSB->0 recovers the baseline.
%   The LSB matching a ~3e-11-class floor is FLAGGED as inferred (we lack
%   Ruane 2020's hardware LSB).
%
%   Region default 'hp' (DST half-plane 3-8, the primary metric).
%
%   See also: ctb_efc (quant_nm plant hook), ctb_dst_s1 (baseline).
    if nargin < 1 || isempty(lsb_nm), lsb_nm = [0.001 0.01 0.1]; end
    if nargin < 2 || isempty(nrep),   nrep = 4; end
    if nargin < 3 || isempty(region), region = 'hp'; end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);
    jac = fullfile(here, sprintf('ctb_dm_jacobian_N1024_c6L080_N1024_%s.mat', region));
    assert(isfile(jac), 'ctb_dst_2e: baseline Jacobian %s missing (run ctb_dst_s1 first)', jac);

    rep = fullfile(here,'ctb_dst_2e_report.txt');
    logf_(rep,'==== DST Lane 2e -- DM LSB quantization | region %s | N=1024 | Lyot 0.80 | control==truth | %s', ...
          region, datestr(now,31)); %#ok<DATST>

    % clean baseline (LSB = 0) -- must recover the S1 r0 floor
    o0 = ctb_efc('jac',jac,'niter',20,'quant_nm',0,'save',false);
    logf_(rep,'LSB = 0 (clean): floor %.3e  [S1 r0 baseline recovered]', o0.c_after);

    OUT = struct('region',region,'lsb_nm',lsb_nm,'clean',o0.c_after,'rows',[]);
    logf_(rep,'  LSB nm |   mean floor |   std      |    min      |    max     | (nrep=%d seeds)', nrep);
    rows = struct('lsb',{},'mean',{},'std',{},'min',{},'max',{},'floors',{});
    for lsb = lsb_nm(:).'
        floors = zeros(1,nrep);
        for s = 1:nrep
            o = ctb_efc('jac',jac,'niter',20,'quant_nm',lsb,'quant_seed',s,'save',false);
            floors(s) = o.c_after;
        end
        logf_(rep,' %7.4f | %12.3e | %10.3e | %10.3e | %10.3e', ...
              lsb, mean(floors), std(floors), min(floors), max(floors));
        rows(end+1) = struct('lsb',lsb,'mean',mean(floors),'std',std(floors), ...
              'min',min(floors),'max',max(floors),'floors',floors); %#ok<AGROW>
    end
    OUT.rows = rows;
    save(fullfile(here,sprintf('ctb_dst_2e_%s.mat',region)),'-struct','OUT');
    logf_(rep,'(controller continuous on the deep mono Jacobian; plant U(+/-LSB/2) dither; LSB=inferred, no Ruane-2020 hardware value.)');
end

function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep,'a'); fprintf(fid,'%s\n',s); fclose(fid);
    fprintf('%s\n', s);
end
