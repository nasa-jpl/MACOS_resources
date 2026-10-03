function OUT = ctb_dst_s1(N, nrelin)
%CTB_DST_S1  DST Lane S1 baseline: charge-6 idealized vortex floors on the
%   DST-matched configuration (Lyot 0.80), scored on BOTH regions -- the
%   DST half-plane 3-8 lambda/D dark hole and the campaign annulus 3-15.
%   control == truth, perfect sensing.
%
%   Runs at grid N (default 1024).  The N=512 grid was shown (attribution
%   probes, 2026-09-02) to floor the idealized charge-6/Lyot-0.80 static
%   ~57x too high -- the pixelated vortex core at 2 px/lambda/D, NOT an
%   open-Lyot wall (K-sweep flat, N=1024 drops it 57x; the Session-11/12
%   sampling pattern).  So the valid baseline is measured at N=1024.
%
%   Per region: fixed-G EFC (r0) then a RELINEARIZATION LADDER (re-measure
%   G at the dug state, warm-start there -- Session-12 idiom) up to nrelin
%   rounds, stopping on <3%-over-two-rounds convergence.  Records static /
%   per-round EFC floors + the linear-achievable la@50nm (ctb_linfloor) at
%   each Jacobian.  ctb_efc save=false (skip its figure step -- headless
%   colorbar bug; numbers captured here); Jacobians save .mat+.fp.json.
%
%   OUT + ctb_dst_s1_N<N>.mat carry every floor; rows appended to
%   ctb_dst_s1_report.txt.  Heavy G .mat are gitignored (fingerprints are
%   the tracked truth).
%
%   See also: ctb_dm_jacobian, ctb_efc, ctb_linfloor, ctb_push, ctb_chain.
    if nargin < 1 || isempty(N),      N = 1024; end
    if nargin < 2 || isempty(nrelin), nrelin = 3; end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);
    cargs = {'fpm_kind','vortex','charge',6,'apodizer',false,'r_lyot_frac',0.80};

    R(1) = struct('key','ann','region','annulus',  'inner',3,'outer',15);
    R(2) = struct('key','hp', 'region','halfplane','inner',3,'outer',8);

    rep = fullfile(here,'ctb_dst_s1_report.txt');
    logf_(rep,'==== DST Lane S1 -- charge-6 idealized baseline | N=%d | Lyot 0.80 | control==truth, perfect sensing | %s', ...
          N, datestr(now,31)); %#ok<DATST>
    logf_(rep,'region     scoring   round | dz px | contrast   | la@50nm    | stroke nm');
    OUT = struct('N',N);
    for i = 1:numel(R)
        r = R(i);
        try
            cbase = sprintf('c6L080_N%d_%s', N, r.key);
            fprintf('\n===== S1 N=%d: %s (%s %g-%g) =====\n', N, r.key, r.region, r.inner, r.outer);
            % r0: fresh Jacobian at flat state + fixed-G EFC
            J = ctb_dm_jacobian('model_size',N,'chain',cargs,'inner_lamD',r.inner, ...
                    'outer_lamD',r.outer,'region',r.region,'tag',cbase);
            o = ctb_efc('jac',J,'niter',20,'save',false);
            la = ctb_linfloor(J,50);
            logf_(rep,' %-9s  %-9s r0    | %5d | %10.3e | %10.3e | (static %.3e)', ...
                  r.key,r.region,numel(J.dz_idx),o.c_after,la.floor,o.c_before);
            floors = o.c_after;  a = o.a;  lafl = la.floor;
            % relin ladder
            for rr = 1:nrelin
                Jr = ctb_dm_jacobian('model_size',N,'chain',cargs,'inner_lamD',r.inner, ...
                        'outer_lamD',r.outer,'region',r.region,'a0',a, ...
                        'tag',sprintf('%s_r%d',cbase,rr));
                or = ctb_efc('jac',Jr,'a0',a,'niter',20,'save',false);
                lar = ctb_linfloor(Jr,50);
                logf_(rep,' %-9s  %-9s r%-4d | %5d | %10.3e | %10.3e | [%s]', ...
                      r.key,r.region,rr,numel(Jr.dz_idx),or.c_after,lar.floor, ...
                      num2str(or.stroke_rms_nm,'%.1f '));
                floors(end+1) = or.c_after; %#ok<AGROW>
                lafl(end+1) = lar.floor; %#ok<AGROW>
                improved = or.c_after < floors(end-1);
                if improved, a = or.a; end
                if ~improved || floors(end-1)/max(floors(end),realmin) < 1.03
                    logf_(rep,' %-9s  %-9s CONVERGED at %.3e (round %d)', r.key,r.region,min(floors),rr);
                    break
                end
            end
            OUT.(r.key) = struct('region',r.region,'inner',r.inner,'outer',r.outer, ...
                'dz_px',numel(J.dz_idx),'static',o.c_before,'floors',floors, ...
                'la',lafl,'best',min(floors),'a',{a});
        catch ME
            logf_(rep,' %-9s  FAILED: %s', r.key, ME.message);
            fprintf(2,'S1 region %s FAILED: %s\n', r.key, ME.message);
        end
    end
    save(fullfile(here,sprintf('ctb_dst_s1_N%d.mat',N)),'-struct','OUT');
    logf_(rep,'(la@50 = linear-achievable at 50 nm; relin re-measures G at the dug state; N=%d.)', N);
end

function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep,'a'); fprintf(fid,'%s\n',s); fclose(fid);
    fprintf('%s\n', s);
end
