function OUT = ctb_dst_s1_nb(centers_nm, N)
%CTB_DST_S1_NB  DST Lane S1 narrowband leg: charge-6 idealized floors vs
%   center wavelength (the DST 1% narrowband sweep), N=1024, matched config
%   (Lyot 0.80), both scoring regions.  control == truth, perfect sensing.
%
%   APPROXIMATION (Dave 2026-09-03): the idealized charge-6 vortex is
%   ACHROMATIC (phase exp(i6theta) is wavelength-independent), so a 1%
%   band ~= mono at each center for the BASELINE.  Each center is run as
%   MONO (1-color) at that wavelength via ctb_chain's lam_m override -- the
%   true 1% 3-color band is reserved for the chromatic defect lanes where
%   it matters.  Expectation: the floor is FLAT across 610-660 and decades
%   below DST's measured 4-11e-10 narrowband.
%
%   Per center x region: fixed-G EFC (r0, niter 20) + la@50 diagnostic.
%   (The mono baseline's relin ladder reaches deeper still, 3e-14; r0 here
%   is consistent across centers and sufficient to establish the flat
%   reference the defect lanes compare against.)  Rows appended to
%   ctb_dst_s1_report.txt; Jacobian .fp.json committed, heavy .mat
%   gitignored.
%
%   See also: ctb_dst_s1 (mono baseline + ladder), ctb_efc_physics (bands),
%   ctb_dm_jacobian, ctb_efc, ctb_linfloor.
    if nargin < 1 || isempty(centers_nm), centers_nm = [615 625 635 645 655]; end
    if nargin < 2 || isempty(N),          N = 1024; end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);
    base = {'fpm_kind','vortex','charge',6,'apodizer',false,'r_lyot_frac',0.80};
    R(1) = struct('key','ann','region','annulus',  'inner',3,'outer',15);
    R(2) = struct('key','hp', 'region','halfplane','inner',3,'outer',8);

    rep = fullfile(here,'ctb_dst_s1_report.txt');
    logf_(rep,'==== DST Lane S1 -- charge-6 NARROWBAND sweep (mono-at-center approx) | N=%d | Lyot 0.80 | %s', ...
          N, datestr(now,31)); %#ok<DATST>
    logf_(rep,'center nm  region     | dz px | static     | EFC(r0)    | la@50nm    | stroke nm');
    OUT = struct('N',N,'centers_nm',centers_nm);
    rows = struct('center',{},'region',{},'static',{},'efc',{},'la',{});
    for C = centers_nm(:).'
        for i = 1:numel(R)
            r = R(i);
            try
                cargs = [base, {'lam_m', C*1e-9}];
                tag = sprintf('c6L080_N%d_nb%d_%s', N, C, r.key);
                fprintf('\n===== S1 NB: %d nm, %s (%s %g-%g) =====\n', C, r.key, r.region, r.inner, r.outer);
                J = ctb_dm_jacobian('model_size',N,'chain',cargs,'inner_lamD',r.inner, ...
                        'outer_lamD',r.outer,'region',r.region,'tag',tag);
                o  = ctb_efc('jac',J,'niter',20,'save',false);
                la = ctb_linfloor(J,50);
                logf_(rep,' %4d nm    %-9s | %5d | %10.3e | %10.3e | %10.3e | [%s]', ...
                      C, r.region, numel(J.dz_idx), o.c_before, o.c_after, la.floor, ...
                      num2str(o.stroke_rms_nm,'%.1f '));
                rows(end+1) = struct('center',C,'region',r.region,'static',o.c_before, ...
                      'efc',o.c_after,'la',la.floor); %#ok<AGROW>
            catch ME
                logf_(rep,' %4d nm    %-9s FAILED: %s', C, r.region, ME.message);
                fprintf(2,'S1 NB %d %s FAILED: %s\n', C, r.region, ME.message);
            end
        end
    end
    OUT.rows = rows;
    save(fullfile(here,sprintf('ctb_dst_s1_nb_N%d.mat',N)),'-struct','OUT');
    logf_(rep,'(mono-at-center approx for the achromatic idealized mask; r0 fixed-G; N=%d.)', N);
end

function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep,'a'); fprintf(fid,'%s\n',s); fclose(fid);
    fprintf('%s\n', s);
end
