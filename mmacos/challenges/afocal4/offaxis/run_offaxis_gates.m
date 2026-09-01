function run_offaxis_gates(varargin)
%RUN_OFFAXIS_GATES  Realizability on every off-axis deck this slice would quote.
%
%   Task 3 of the brief, and the reason it is a separate driver: an off-axis
%   train trades OBSCURATION for TILT.  The coaxial family's failure mode is a
%   body standing in a beam; the off-axis family's is angle-of-incidence
%   growth and package girth.  A study that reported only the wavefront would
%   be quoting the half of the trade that improved.
%
%   Everything scored here is DESCENT_REQUIRE's committed requirement set, on
%   one footing, so an off-axis row and a coaxial row are subtractable:
%   targets (WFE rung 2, pupil blur, wander, breathing, interface surface
%   rim-anchored, M error), walls (union floor, last powered mirror behind
%   M1, minimum spacing, max chief AOI against the 15 deg standing rule), and
%   gates (rays lost, anchoring residual).
%
%   THE AOI COLUMN IS THE ONE TO READ FIRST on these decks, and it is
%   reported per mirror rather than as a maximum alone.  Decentering the pupil
%   by h moves the chief off every parent axis, so incidence grows on EVERY
%   surface at once -- unlike a fold tilt, which spends its angle at one
%   station.  Where the rule is broken, the design is REPORTED broken and
%   named; it is not quietly dropped from the table.  A design that fails a
%   realizability wall is a finding about the family, and hiding it would
%   leave the wavefront number looking free.
%
%   THE REQUIRED APERTURE IS THE OFF-AXIS FAMILY'S OWN PRICE, and it has no
%   counterpart in the coaxial tables: an off-axis section of radius r taken
%   at height h needs a PARENT of radius |h| + r, and that parent is the part
%   somebody has to figure and test.  Reported per element from the measured
%   footprints, beside the section's own radius, so the two are never
%   confused.
%
%   Env: OG_DECKS (comma list of .in paths, or a glob), OG_TAG.

    ap = fileparts(fileparts(mfilename('fullpath')));
    addpath(ap); addpath(fullfile(ap,'clearing')); addpath(fullfile(ap,'descent'));
    addpath(fullfile(ap,'offaxis')); addpath(fullfile(ap,'wall'));

    spec = getenv_d('OG_DECKS', fullfile(ap,'offaxis','afocal4_OAW*_N*_h*.in'));
    tag  = getenv_d('OG_TAG','OAG');
    decks = resolve_(spec);
    if isempty(decks)
        fprintf('  no decks matched %s\n', spec);   return;
    end

    P = afocal4_params();
    macos.init(P.model_size);

    fprintf('\n==== OFF-AXIS REALIZABILITY ====\n');
    fprintf('  %d deck(s); AOI rule 15 deg; union body = %.2f x footprint + %.0f mm\n\n', ...
            numel(decks), getf_(P.pack,'union_body_k',1.15), ...
            getf_(P.pack,'union_body_pad',0.015)*1e3);

    rows = struct('deck',{},'ok',{},'walls',{},'gates',{},'wfe',{},'aoi',{}, ...
                  'union',{},'behind',{},'lost',{},'parent_r',{},'sect_r',{});
    for i = 1:numel(decks)
        d = decks{i};
        [~,nm] = fileparts(d);
        fprintf('  ---- %s ----\n', nm);
        try
            Q = descent_require(P, d, 'fields',P.Fsolve, 'union',true, 'quiet',false);
        catch ME
            fprintf('    REQUIRE FAILED: %s\n    %s\n\n', ME.identifier, ME.message);
            continue;
        end

        % the parents the sections come out of -- the off-axis family's bill
        [pr, sr, nmv] = parents_(d, P);
        fprintf('\n    %-6s %12s %12s %12s\n', 'elt','section r mm', ...
                'offset mm','PARENT r mm');
        for k = 1:numel(pr)
            fprintf('    %-6s %12.2f %12.2f %12.2f\n', nmv{k}, sr(k)*1e3, ...
                    (pr(k)-sr(k))*1e3, pr(k)*1e3);
        end
        aoi = Q.aoi.per_elt_deg;
        fprintf('\n    chief AOI per mirror (deg): %s   max %.2f  %s\n\n', ...
                strjoin(arrayfun(@(x) sprintf('%.2f',x), aoi(~isnan(aoi)), ...
                        'UniformOutput',false), ' '), Q.aoi.max_deg, ...
                tern_(Q.aoi.max_deg <= 15, '', '<-- BREAKS the 15 deg rule'));

        % ---- the layout render: a review gate BEFORE the numbers --------
        % view_rx reads the loaded Rx back from the engine and draws each
        % optic as a solid body on its declared ApVec, so on these decks it
        % renders the fitted OFF-AXIS SECTIONS rather than centred discs --
        % which is the whole point of looking at one.
        png = fullfile(fileparts(d), sprintf('%s_layout.png', nm));
        try
            macos.load_rx(d);
            f = macos.view_rx('visible',false);
            set(f,'Position',[100 100 1400 520]);
            exportgraphics(f, png, 'Resolution',140);   close(f);
            fprintf('    layout -> %s\n', png);
        catch ME
            fprintf('    layout render failed: %s\n', ME.message);
        end

        rows(end+1) = struct('deck',d, 'ok',Q.ok, 'walls',Q.walls_ok, ...
            'gates',Q.gates_ok, 'wfe',Q.S.wfe_max_nm, 'aoi',Q.aoi.max_deg, ...
            'union',Q.floor_mm, 'behind',Q.z.behind_m1, ...
            'lost',lost_(Q), 'parent_r',pr, 'sect_r',sr); %#ok<AGROW>
    end

    fprintf('==== SUMMARY ====\n');
    fprintf('  %-30s %11s %8s %10s %10s %7s\n', 'deck','WFE nm','AOI deg', ...
            'union mm','behind mm','all ok');
    for i = 1:numel(rows)
        r = rows(i);   [~,nm] = fileparts(r.deck);
        fprintf('  %-30s %11.1f %8.2f %+10.1f %+10.0f %7s\n', nm, r.wfe, ...
                r.aoi, r.union, r.behind*1e3, ...
                tern_(r.ok && r.walls && r.gates,'yes','NO'));
    end
    save(fullfile(fileparts(mfilename('fullpath')), ...
         sprintf('offaxis_gates_%s.mat',tag)), 'rows','P','-v7.3');
    fprintf('\n');
end

% =====================================================================
function [pr, sr, nm] = parents_(deck, P)
%PARENTS_  Section radius, its offset from the parent vertex, and hence the
%   PARENT radius each element is cut from -- measured from the traced
%   footprints over the field box, in each element's own aperture frame.
    I = probe_apertures_(deck, P.Fsolve);
    n = numel(I);   pr = zeros(1,n);   sr = zeros(1,n);   nm = cell(1,n);
    for k = 1:n
        sr(k) = I(k).r_m;
        pr(k) = hypot(I(k).xc_m, I(k).yc_m) + I(k).r_m;
        nm{k} = I(k).name;
    end
end

function ap = probe_apertures_(deck, F)
%PROBE_APERTURES_  OFFAXIS_DECENTER's measurement, with no edit: copy the
%   deck, decenter by zero, and read back what it measured.  Reusing that
%   routine rather than re-deriving the footprint keeps ONE definition of an
%   off-axis clear aperture in the study.
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    copyfile(deck, tmp);
    I = offaxis_decenter(tmp, 0, 'fields',F, 'fit',false, 'quiet',true);
    ap = I.ap;
end

function c = resolve_(spec)
    if any(spec == '*')
        d = dir(spec);
        c = arrayfun(@(x) fullfile(x.folder,x.name), d, 'UniformOutput',false);
    else
        c = strtrim(strsplit(spec,','));
        c = c(cellfun(@(x) isfile(x), c));
    end
    c = c(:).';
end

function v = getenv_d(k,d), v = getenv(k); if isempty(v), v = d; end, end
function v = getf_(s,f,d), if isstruct(s)&&isfield(s,f), v=s.(f); else, v=d; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
function n = lost_(Q)
%LOST_  The union gate's ray count -- DESCENT_REQUIRE's own 'rays lost' row
%   is taken from there, so this reads the same number rather than a second
%   opinion about it.
    n = 0;
    if isstruct(Q) && isfield(Q,'K') && isstruct(Q.K) && isfield(Q.K,'nLost')
        n = Q.K.nLost;
    end
end

function del_(f), if exist(f,'file'), delete(f); end, end
