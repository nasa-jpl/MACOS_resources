% RUN_OFFAXIS  Is the wavefront limit the COAXIAL CONSTRAINT?
%
%   The descent measured that no mirror count in this family reaches the 71 nm
%   target -- 48x at N = 7 with the pupil requirement abandoned.  Every design
%   in that ladder is COAXIAL: all mirrors on one axis, the aberration field
%   centred on it, and the 0.5 deg field box used in an annular zone offset
%   0.6 deg off that centre.  S4 measured the residual as FIELD-VARYING
%   astigmatism (1108 -> 3312 -> 6809 nm across the box) and found rigid
%   bodies bought 0.4 %, because a rigid body adds a field-CONSTANT term.
%
%   THAT MEASUREMENT WAS MADE INSIDE A PERTURBATION BOUND.  P.bounds.tilt is
%   +-0.05 rad = +-2.86 deg and P.bounds.dec is +-50 mm -- alignment
%   tolerances, not design freedoms -- and the rigid bodies were never in the
%   DOF set that produced the committed deck.  Tilting and decentring MOVES
%   THE ABERRATION FIELD CENTRE, which is the standard tool for exactly the
%   field-varying astigmatism S4 identified.  So the question is whether the
%   limit is the OPTICS or the COAXIAL CONSTRAINT, and it is answered by
%   opening those bounds to design scale and re-solving.
%
%   TWO ARMS, because a wavefront-only solve at large tilt can cheat:
%     'wfe'  wavefront ONLY.  The most optimistic number -- but with the
%            pupil ladder unscored the MAGNIFICATION term is unscored too, so
%            a big tilt can buy wavefront by ceasing to be a 30x telescope.
%            M and collimation are therefore REPORTED on every result and a
%            floor reached by breaking them is not a design.
%     'full' the whole requirement set, so M is defended by the merit.
%
%   OFFAX_ARM    'wfe' | 'full'   OFFAX_TILT  bound, deg (15)
%   OFFAX_DEC    bound, mm (300)  OFFAX_EVALS / OFFAX_ROUNDS / OFFAX_TAG
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));  up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing')); addpath(fullfile(up,'wall'));

arm  = getenv('OFFAX_ARM');   if isempty(arm), arm = 'wfe'; end
tlim = str2double(getenv('OFFAX_TILT'));  if isnan(tlim), tlim = 15;  end
dlim = str2double(getenv('OFFAX_DEC'));   if isnan(dlim), dlim = 300; end
ev   = str2double(getenv('OFFAX_EVALS')); if isnan(ev),   ev = 500;   end
rd   = str2double(getenv('OFFAX_ROUNDS'));if isnan(rd),   rd = 2;     end
tag  = getenv('OFFAX_TAG');   if isempty(tag), tag = sprintf('OFFAX_%s',arm); end

macos.init(256);
P = afocal4_params();
P.pack.enforce = true;
P.solve.fd_type='central'; P.solve.fd_step=1e-4;
P.solve.tol_fun=1e-8; P.solve.tol_x=1e-9; P.solve.tol_opt=1e-8;
P.solve.max_fev = ev;
% THE COAXIAL CONSTRAINT, RELEASED.  Alignment tolerances become design
% freedoms; the scales go with them or one solver unit means nothing.
P.bounds.tilt = deg2rad(tlim)*[-1 1];
P.bounds.dec  = (dlim/1e3)*[-1 1];
P.dof_scale.tilt = deg2rad(2);      % one unit = 2 deg
P.dof_scale.dec  = 0.020;           % one unit = 20 mm

src = fullfile(up,'afocal4_b2long_343mm.in');
D0  = wall_recover(P, src);
D0.tilt_deg = 0;
fprintf(['\n==== OFF-AXIS: is the limit the OPTICS or the COAXIAL ' ...
         'CONSTRAINT? ====\n  arm %s, rigid bodies on elements [%s], ' ...
         'tilt +-%.1f deg, decenter +-%.0f mm\n'], upper(arm), ...
        num2str(P.rb_elts), tlim, dlim);

want_pupil = strcmp(arm,'full');
S0 = afocal4_score(P, src, 'fields',P.Fsolve, 'nodes',P.solve.nodes_score, ...
                   'pupil',want_pupil);
fprintf('  coaxial start: WFE %.1f nm\n', S0.wfe_max_nm);
fprintf('  coaxial wavefront-only FLOOR for reference: 3841.8 nm (54x target)\n\n');

deck = fullfile(here, sprintf('afocal4_%s.in', tag));
Dc = D0;  mprev = Inf;  rounds = struct('k',{},'nfev',{},'xfl',{},'wfe',{});
for r = 1:rd
    try
        Rr = clear_solve(P, Dc, 'dofs',{'conic','standoff','front','rb'}, ...
                 'deck',deck, 'pupil',want_pupil, 'max_iter',400, ...
                 'label',sprintf('%s r%d',tag,r), 'quiet',true);
    catch ME
        fprintf('  round %d FAILED: %s\n', r, ME.message);  break;
    end
    fprintf('  round %d: %4d evals, %5.1f min, xfl %d, WFE %9.1f nm\n', ...
            r, Rr.nfev, Rr.seconds/60, Rr.exitflag, Rr.S.wfe_max_nm);
    rounds(end+1) = struct('k',r,'nfev',Rr.nfev,'xfl',Rr.exitflag, ...
                           'wfe',Rr.S.wfe_max_nm); %#ok<SAGROW>
    m = Rr.S.merit;  Dc = Rr.D;
    if isfinite(mprev) && (mprev-m)/max(abs(mprev),eps) < 1e-6, break; end
    mprev = m;
end

Pr = P;  Pr.pack.enforce = false;
clear_build(Pr, Dc, deck, 'verify',false);
Q = descent_require(P, deck, 'quiet',true);
S1 = afocal4_score(P, deck, 'fields',P.Fsolve, 'nodes',P.solve.nodes_score);
fprintf('\n  RESULT (%s arm)\n', arm);
fprintf('    WFE  %9.1f nm   (%.0fx the 71 nm target; coaxial floor 3841.8)\n', ...
        S1.wfe_max_nm, S1.wfe_max_nm/71);
fprintf('    blur %9.1f um   breathing %7.4f %%   wander %8.1f um\n', ...
        S1.blur_um, S1.breathe_pct, S1.wander_um);
fprintf('    M    %9.4f      (error %.4f %% against a 0.1 %% target)  <-- a floor\n', ...
        S1.mag_centre_chief, abs(S1.mag_centre_chief/30-1)*100);
fprintf('    reached by breaking M is NOT a design\n');
fprintf('    union floor %+.2f mm, anchoring residual %.4f um, targets %s\n', ...
        Q.floor_mm, S1.anchor_resid_um, tern(Q.ok,'MET','missed'));
fprintf('    rigid bodies: dec [%s] mm, tilt [%s] deg\n', ...
        strjoin(arrayfun(@(x)sprintf('%+.1f',x*1e3),Dc.rb(:,1).','UniformOutput',false),' '), ...
        strjoin(arrayfun(@(x)sprintf('%+.2f',rad2deg(x)),Dc.rb(:,2).','UniformOutput',false),' '));
R = struct('arm',arm,'tilt_lim',tlim,'dec_lim',dlim,'D',Dc,'Q',Q,'S',S1, ...
           'rounds',rounds,'deck',deck,'tag',tag); %#ok<NASGU>
save(fullfile(here, sprintf('descent_%s.mat', tag)), 'R', '-v7.3');
fprintf('\n  wrote %s\n', fullfile(here, sprintf('descent_%s.mat', tag)));
exit(0);

function s = tern(c,a,b), if c, s=a; else, s=b; end, end
