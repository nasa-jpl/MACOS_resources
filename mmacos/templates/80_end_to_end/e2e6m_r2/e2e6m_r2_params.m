function P = e2e6m_r2_params(over)
%E2E6M_R2_PARAMS  Round-2 knobs; everything unnamed inherits round 1.
%
%   Wraps ../e2e6m/e2e6m_params (round 1, FROZEN, read-only) and adds
%   the round-2 back end: the CTB coronagraph topology instanced at
%   observatory scale -- 8 OAPs + DM1/DM2 -- per macos/BRIEF_e2e6m_redo
%   R1 and Dave's round-2 item 1.
%
%   TOPOLOGY (light order, from the telescope focus):
%
%     OAP1 (collimate) -> DM1 (pupil) -> DM2 -> OAP2 (focus)
%       -> [intermediate focus, no station] -> OAP3 (collimate)
%       -> Apodizer pupil -> OAP4 (focus) -> FPM focus
%       -> OAP5 (collimate) -> Lyot pupil -> OAP6 (focus)
%       -> FieldStop focus -> OAP7 (collimate) -> Backend pupil
%       -> OAP8 (focus) -> Science
%
%   Every OAP after the collimator is an f_relay 1:1 relay, so the
%   47 mm collimated pupil is preserved at DM1/DM2, the Apodizer, the
%   Lyot and the Backend marker -- which keeps the mask scales and the
%   lambda/D bookkeeping directly comparable with round 1's committed
%   numbers.  The DIFFRACTION SEED moves to the DM1->DM2 leg (the CTB
%   convention): the field then EXISTS at the DM planes, which is what
%   the EFC layer probes, and ctb_aplc's DM1/DM2 stations become the
%   real planes instead of round 1's apodizer-leg stand-ins.
%
%   P = E2E6M_R2_PARAMS()      defaults
%   P = E2E6M_R2_PARAMS(OVER)  with top-level field overrides
%
%   See also R1_BACKEND, R1_CORO, ../e2e6m/e2e6m_params.

    arguments
        over struct = struct()
    end
    here  = fileparts(mfilename('fullpath'));
    r1dir = fullfile(here, '..', 'e2e6m');
    addpath(r1dir);
    P = e2e6m_params();
    P.outdir = here;               % round-2 artifacts land HERE
    P.r1dir  = r1dir;              % round-1 artifacts, READ-ONLY

    % ---- the DM-bearing back end (R1) -----------------------------------
    P.b2.tag      = 'seg';         % artifact suffix; 'mono' for the twin
    P.b2.base_in  = 's2_segmented.in';   % base deck, resolved in r1dir
    P.b2.fno_in   = NaN;           % NaN = read the telescope f/# from
                                   % round 1's s1_run.mat
    P.b2.back_m   = 0.30;          % bench start before the telescope focus
    P.b2.f_oap1   = 1.20;          % collimator -> 47 mm pupil at f/25.4
    P.b2.f_relay  = 0.90;          % OAP2..OAP8, 1:1 pupil relays
    P.b2.aoi_deg  = 6;             % every fold (OAPs AND DMs), sides
                                   % alternating -- the round-1 accordion
                                   % that stays inside the shroud annulus
    P.b2.d_dm1    = 0.25;          % OAP1 -> DM1 (the collimated pupil)
    P.b2.d_dm2    = 0.15;          % DM1 -> DM2 (the 2-DM Talbot spacing;
                                   % what buys the two-sided dark zone)
    P.b2.d_oap2   = 0.20;          % DM2 -> OAP2
    P.b2.d_mark   = 0.25;          % collimating OAP -> marker, and
                                   % marker -> next OAP
    P.b2.aprad_dm = 0.030;         % DM clear-aperture radius, m (beam
                                   % radius at the pupil is 0.0236)
    P.b2.drop_tail = 3;            % telescope terminal quartet dropped at
                                   % the splice (as round 1)

    % ---- DM augmentation (grid-data surfaces on the prop deck) ----------
    P.dm.ng   = 256;               % nGridMat (model >= ng)
    P.dm.names = {'DM1','DM2'};

    % ---- overrides ------------------------------------------------------
    fn = fieldnames(over);
    for i = 1:numel(fn)
        if ~isfield(P, fn{i})
            error('e2e6m_r2_params:unknown','unknown parameter "%s"', fn{i});
        end
        P.(fn{i}) = over.(fn{i});
    end
end
