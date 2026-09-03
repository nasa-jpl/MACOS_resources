function OUT = ctb_dst_s1_mono()
%CTB_DST_S1_MONO  DST Lane S1, MONO leg: charge-6 idealized vortex floors
%   on the DST-matched configuration (Lyot 0.80 diameter fraction),
%   scored on BOTH regions -- the DST half-plane 3-8 lambda/D dark hole
%   and the campaign annulus 3-15 reported alongside.  control == truth,
%   perfect sensing.
%
%   Per region: fixed-G EFC (the Session-10 loop) -> relinearize (re-
%   measure G at the dug state, warm-start there -- Session-12 idiom) ->
%   record static / EFC / relin floors + the linear-achievable (la@50nm)
%   diagnostic at the static and dug states (ctb_linfloor).  The Jacobians
%   (ctb_dm_jacobian_N512_c6L080_{ann,hp38}.mat) must exist first.
%
%   Idealized baseline: expected DECADES below DST's measured floors --
%   that headroom is what the defect lanes (2c) and quantization (2e)
%   will spend.
%
%   See also: ctb_dm_jacobian, ctb_efc, ctb_linfloor, ctb_chain.
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);
    cargs = {'fpm_kind','vortex','charge',6,'apodizer',false,'r_lyot_frac',0.80};

    R(1) = struct('key','ann','region','annulus',  'inner',3,'outer',15, ...
                  'jac',fullfile(here,'ctb_dm_jacobian_N512_c6L080_ann.mat'));
    R(2) = struct('key','hp', 'region','halfplane','inner',3,'outer',8, ...
                  'jac',fullfile(here,'ctb_dm_jacobian_N512_c6L080_hp38.mat'));

    rep = fullfile(here,'ctb_dst_s1_report.txt');
    logf_(rep,'==== DST Lane S1 -- charge-6 idealized baseline (MONO) | Lyot 0.80 | control==truth, perfect sensing | %s', ...
          datestr(now,31)); %#ok<DATST>
    logf_(rep,'region     scoring        | dz px | static     | EFC floor  | la@50 stat | relin floor| la@50 dug  | stroke nm');
    OUT = struct();
    for i = 1:numel(R)
        r = R(i);
        tb = sprintf('s1_c6L080_%s', r.key);
        fprintf('\n===== S1 mono: %s (%s %g-%g) =====\n', r.key, r.region, r.inner, r.outer);
        o1  = ctb_efc('jac', r.jac, 'niter', 15, 'tag', tb, 'save', true);
        J1  = load(r.jac);
        la1 = ctb_linfloor(J1, 50);
        % relinearize: re-measure G at the dug state, warm-start EFC there
        Jr  = ctb_dm_jacobian('chain', cargs, 'inner_lamD', r.inner, ...
                  'outer_lamD', r.outer, 'region', r.region, 'a0', o1.a, ...
                  'tag', sprintf('c6L080_%s_r1', r.key));
        o2  = ctb_efc('jac', Jr, 'a0', o1.a, 'niter', 15, 'tag', [tb '_r1'], 'save', true);
        lar = ctb_linfloor(Jr, 50);
        logf_(rep,' %-9s  %-9s %4g-%2g | %5d | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | [%s]', ...
              r.key, r.region, r.inner, r.outer, numel(J1.dz_idx), ...
              o1.c_before, o1.c_after, la1.floor, o2.c_after, lar.floor, ...
              num2str(o2.stroke_rms_nm, '%.1f '));
        OUT.(r.key) = struct('region',r.region,'inner',r.inner,'outer',r.outer, ...
            'dz_px',numel(J1.dz_idx),'static',o1.c_before,'efc',o1.c_after, ...
            'la_static',la1.floor,'relin',o2.c_after,'la_dug',lar.floor, ...
            'stroke_nm',o2.stroke_rms_nm);
    end
    logf_(rep,'(la@50 = linear-achievable floor at 50 nm stroke bound; relin re-measures G at the dug state.)');
    save(fullfile(here,'ctb_dst_s1_mono.mat'),'-struct','OUT');
end

function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep,'a'); fprintf(fid,'%s\n',s); fclose(fid);
    fprintf('%s\n', s);
end
