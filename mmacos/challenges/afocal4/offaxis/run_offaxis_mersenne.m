function run_offaxis_mersenne(varargin)
%RUN_OFFAXIS_MERSENNE  The off-axis family's BEST CASE, with the pupil
%   requirement dropped -- an upper bound on what going off axis can buy.
%
%   A two-mirror confocal parabola pair is EXACTLY afocal and EXACTLY P.M x
%   for a beam entering anywhere on it.  It has no free parameter left: both
%   powers are consumed by the magnification and the collimation, so its exit
%   pupil lands where the geometry puts it and cannot be moved.  It therefore
%   CANNOT meet the interface-pupil requirement -- and that is exactly why it
%   is worth measuring.  It separates two questions the full requirement set
%   fuses together:
%
%     Q1  Is the OFF-AXIS FAMILY capable of the 71 nm wavefront at all?
%     Q2  Is it capable of it WHILE ALSO putting the exit pupil 140 mm past
%         the last mirror?
%
%   If the bare Mersenne is already far above 71 nm, the answer to Q1 is no
%   and the pupil ladder is irrelevant -- no amount of back-end cleverness
%   recovers a front end that cannot form the wavefront.  If it is below,
%   then the gap between it and a full off-axis solve is the PRICE OF THE
%   PUPIL in this family, directly comparable with the coaxial family's 2.7x
%   at 343 mm.
%
%   WHY THE CLOSURE CANNOT BE USED HERE, measured.  DESCENT_CLOSE solves the
%   last two powers from the marginal state after the free mirrors:
%       u2 = um - ym*phi;   b = (yout - ym)/u2;   phiN = u2/yout
%   A Mersenne front end arrives at mirror 2 with y = 0.016667 m and
%   u = 0.000000 -- i.e. ALREADY at the exit height and ALREADY collimated,
%   so the numerator (yout - ym) is zero and the parameterization is
%   singular.  Handed a five-mirror spec it does not fail; it puts a strong
%   third mirror in (phi3 = -1.2308 /m) that BREAKS the Mersenne -- y to
%   0.0372, u to 0.0205 -- and then re-closes it with a 2.74 m lever.  The
%   result is paraxially exact (residuals 1e-16, mag 30.000000) and traces at
%   M = 26.73 with 25445 urad of collimation error, because the intermediate
%   beam grows to 1.6 m across on mirrors of 1.8 m radius.  That is not a bug
%   in the closure: it is the closure being asked to solve a problem that was
%   already solved, and it is why this driver builds the pair directly.
%
%   Env: OM_FORMS, OM_F1 (comma list), OM_H (comma list, m; 'auto' = the
%   clearing decenter), OM_OUT.

    ap = fileparts(fileparts(mfilename('fullpath')));
    addpath(ap); addpath(fullfile(ap,'clearing')); addpath(fullfile(ap,'descent'));
    addpath(fullfile(ap,'offaxis'));

    forms = strsplit(getenv_d('OM_FORMS','cass,greg'), ',');
    f1s   = str2double(strsplit(getenv_d('OM_F1','1.25,2.5,5.0'), ','));
    hstr  = getenv_d('OM_H','auto');
    outd  = getenv_d('OM_OUT', fullfile(ap,'offaxis','decks'));
    if ~exist(outd,'dir'), mkdir(outd); end

    P = afocal4_params();
    macos.init(P.model_size);

    fprintf('\n==== OFF-AXIS MERSENNE: the family with the pupil dropped ====\n');
    fprintf('  target %d nm; coaxial wavefront-only floor 3841.8 nm (N=7, all DOFs)\n', 71);
    fprintf('  %-5s %6s %7s %8s %11s %11s %10s %9s %9s\n', 'form','f1','h','sep', ...
            'traced M','coll urad','WFE nm','x target','union mm');

    rows = struct('form',{},'f1',{},'h',{},'sep',{},'M',{},'coll',{}, ...
                  'wfe',{},'union',{},'deck',{},'lost',{});
    for i = 1:numel(forms)
      fm = strtrim(forms{i});
      for f1 = f1s
        f2 = f1/P.M;
        switch fm
        case 'cass', sep = f1 - f2;   cvx = true;
        case 'greg', sep = f1 + f2;   cvx = false;
        otherwise, error('macos:design:offaxis_mersenne:form','bad form %s',fm);
        end
        if strcmpi(hstr,'auto')
            hs = auto_h_(P);
        else
            hs = str2double(strsplit(hstr,','));
        end
        for h = hs
            deck = fullfile(outd, sprintf('om_%s_f%g_h%g.in', fm, f1, h));
            t = macos.design.Telescope('family','tma', ...
                    'aperture_diameter_m',P.D, 'wavelength_m',P.lambda, ...
                    'grid_npts',P.ngrid, 'model_size',P.model_size);
            t.add_mirror('M1','radius_m',2*f1,'spacing_after_m',sep, ...
                         'convex',false,'conic',-1);
            t.add_mirror('M2','radius_m',2*f2,'spacing_after_m',P.iface, ...
                         'convex',cvx,'conic',-1);
            t.add_exit_reference('ColdStop','dist_m',P.iface);
            if P.bias_deg ~= 0, t.set_field_bias(P.bias_deg*60); end
            t.build(deck);

            oa = offaxis_decenter(deck, h, 'fields',P.Fsolve, 'quiet',true);

            % score WAVEFRONT ONLY -- the pupil ladder is not applicable to a
            % design that has no freedom left to place a pupil with.
            S = afocal4_score(P, deck, 'fields',P.Fsolve, ...
                              'nodes',P.solve.nodes_score, 'pupil',false);
            K = afocal4_union(deck, 'fields',P.Fsolve, ...
                    'body_k',  getf_(P.pack,'union_body_k',1.15), ...
                    'body_pad',getf_(P.pack,'union_body_pad',0.015), ...
                    'quiet',true);

            fprintf('  %-5s %6.2f %7.3f %8.3f %11.5f %11.1f %10.1f %9.0f %9.1f %s\n', ...
                    fm, f1, h, sep, oa.traced.mag, oa.traced.collimation_urad, ...
                    S.wfe_max_nm, S.wfe_max_nm/71, K.floor_m*1e3, ...
                    tern_(oa.nlost>0, sprintf('(%d LOST)',oa.nlost), ''));
            rows(end+1) = struct('form',fm,'f1',f1,'h',h,'sep',sep, ...
                'M',oa.traced.mag,'coll',oa.traced.collimation_urad, ...
                'wfe',S.wfe_max_nm,'union',K.floor_m,'deck',deck, ...
                'lost',oa.nlost); %#ok<AGROW>
        end
      end
    end
    save(fullfile(outd,'offaxis_mersenne.mat'), 'rows','P','-v7.3');
    fprintf('\n  wrote %s\n\n', fullfile(outd,'offaxis_mersenne.mat'));
end

function hs = auto_h_(P)
%AUTO_H_  Decenters that clear the secondary body, plus a coaxial control at
%   h = 0.  The control is the point of the sweep: without it "the off-axis
%   number" has nothing to be off-axis FROM.
    r2   = (P.D/P.M)/2;
    bk   = getf_(getf_(P,'pack',struct()),'union_body_k',1.15);
    bp   = getf_(getf_(P,'pack',struct()),'union_body_pad',0.015);
    hmin = P.D/2 + bk*r2 + 2*bp;
    hs   = [0, hmin, 0.75, 1.00, 1.50];
end

function v = getenv_d(k,d), v = getenv(k); if isempty(v), v = d; end, end
function v = getf_(s,f,d), if isstruct(s)&&isfield(s,f), v=s.(f); else, v=d; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
