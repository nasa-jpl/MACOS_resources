function run_offaxis_seed(varargin)
%RUN_OFFAXIS_SEED  Build the off-axis seeds and PROVE they are what they claim.
%
%   Nothing is scored here.  This driver answers the three questions that
%   have to be settled before a wavefront number from an off-axis deck means
%   anything at all, and it answers them in the order in which a failure of
%   one makes the next meaningless:
%
%     1  IS IT AFOCAL AND IS IT 30x?   Traced, not paraxial.  The descent's
%        standing guard: one cold closure traced M = 40.45 against a
%        paraxial 30 and would have been scored as a design.
%     2  IS EVERY RAY THERE?           A clipped beam reports as a different
%        telescope, not as a clipped beam -- the coaxial apertures lose 777
%        of 1185 rays at h = 0.6 m and call the survivors 37.65x.
%     3  IS IT ACTUALLY OFF AXIS?      The pupil must MISS the secondary's
%        body.  An "off-axis" design that is still obscured is a coaxial
%        design with extra parts.
%
%   Env: OA_FORMS (cass,greg), OA_N, OA_F1, OA_H, OA_OUT.

    ap = fileparts(fileparts(mfilename('fullpath')));
    addpath(ap); addpath(fullfile(ap,'clearing')); addpath(fullfile(ap,'descent'));
    addpath(fullfile(ap,'offaxis'));

    forms = strsplit(getenv_d('OA_FORMS','cass,greg'), ',');
    N     = str2double(getenv_d('OA_N','5'));
    f1    = str2double(getenv_d('OA_F1','1.25'));
    hset  = str2double(getenv_d('OA_H','NaN'));
    outd  = getenv_d('OA_OUT', fullfile(ap,'offaxis','decks'));
    if ~exist(outd,'dir'), mkdir(outd); end

    P = afocal4_params();
    macos.init(P.model_size);

    for i = 1:numel(forms)
        fm = strtrim(forms{i});
        fprintf('\n================ SEED %s (N = %d) ================\n', ...
                upper(fm), N);
        S = offaxis_seed(P, fm, 'N',N, 'f1',f1, 'h',hset);
        fprintf('  %s\n', S.why);

        deck = fullfile(outd, sprintf('oa_%s_N%d.in', fm, N));
        try
            out = descent_build(P, S, deck, 'defer_union',true, ...
                                'oa_fields',P.Fsolve, 'quiet',false);
        catch ME
            fprintf('  BUILD FAILED: %s\n    %s\n', ME.identifier, ME.message);
            continue;
        end

        % ---- 1 + 2: the identities, traced ------------------------------
        oa = out.offaxis;
        fprintf(['\n  IDENTITIES   traced M %.6f (want %.4f, err %.4f %%), ' ...
                 'collimation %.4f urad\n'], out.traced.mag, P.M, ...
                (out.traced.mag/P.M - 1)*100, out.traced.collimation_urad);
        fprintf('               rays %d, lost %d over the %d-field box\n', ...
                oa.nrays, oa.nlost, size(P.Fsolve,1));
        fprintf('  APERTURES    %-6s %11s %11s %11s\n', ...
                'elt','r mm','xc mm','yc mm');
        for k = 1:numel(oa.ap)
            fprintf('               %-6s %11.3f %11.3f %11.3f\n', ...
                    oa.ap(k).name, oa.ap(k).r_m*1e3, oa.ap(k).xc_m*1e3, ...
                    oa.ap(k).yc_m*1e3);
        end

        % ---- 3: is it off axis?  the union gate, its own allowance -------
        K = afocal4_union(deck, 'fields',P.Fsolve, ...
                'body_k',  getf_(P.pack,'union_body_k',1.15), ...
                'body_pad',getf_(P.pack,'union_body_pad',0.015), ...
                'quiet',true);
        fprintf(['\n  UNION        floor %+.2f mm (declared body); ' ...
                 '%d ray(s) lost\n'], K.floor_m*1e3, K.nLost);
        fprintf('               %s\n', tern_(K.floor_m >= 0, ...
                'CLEAR -- no body stands in a beam', ...
                'OBSCURED -- this is not an off-axis design'));

        save(fullfile(outd, sprintf('oa_%s_N%d.mat', fm, N)), ...
             'S','out','K','P','-v7.3');
        fprintf('  wrote %s\n', deck);
    end
    fprintf('\n');
end

function v = getenv_d(k,d), v = getenv(k); if isempty(v), v = d; end, end
function v = getf_(s,f,d), if isfield(s,f), v = s.(f); else, v = d; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
