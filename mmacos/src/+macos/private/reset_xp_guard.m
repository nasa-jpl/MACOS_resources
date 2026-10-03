function varargout = reset_xp_guard(action, varargin)
%RESET_XP_GUARD  Shared no-effect/clobber guard for the reset_xp path.
%   The four dw_d*_multi supervisors re-reference the exit pupil per field
%   by calling macos.fex(1), which the engine honors ONLY when nElt-1 is a
%   Return(EltID 8) or Reference(EltID 3) surface -- for any other type the
%   FEX handler declines to write (macos_cmd_loop.inc: "Invalid element
%   type").  xp_fnd used to return PASS anyway, making reset_xp a silent
%   no-op; it now prechecks the element type and returns FAIL, which the
%   'fex' action below absorbs into the same no-effect verdict.
%   By design (Dave 2026-08-04): the write needs a dedicated exit-pupil
%   element at nElt-1 -- the structure add_pupil builds (2-pass: FP_return
%   + ExitPupil inserted before the FocalPlane).  A bare focal deck (a
%   powered optic at nElt-1, no pupil) has nothing to update.  This helper
%   detects that no-pupil case (and the dangerous case where a write lands
%   on a powered optic) so the supervisors report the truth.
%
%   reset_xp_guard('fex', session[, axis])
%       the per-field macos.fex call itself.  AXIS 'chief' (default) |
%       'centroid' selects the FEX pupil-sphere AXIS (the engine
%       CHIEFRAY/CENTROID toggle; api xp_fnd mode 1 | 0).  The radius
%       and vertex are axis-invariant -- only psi moves -- and on an
%       obscured or segmented beam the centroid axis leaves pure
%       tip/tilt (+piston) frame terms in the OPD (Luis 2026-08-27,
%       the FEX-axis ruling), so 'centroid' is a diagnostic opt-in.
%       Raises
%       macos:dw_dx_multi:noStop when no aperture stop is set; ABSORBS the
%       no-pupil-element FAIL (macos:fex:noPupilElt) so the loop runs to
%       completion and 'finalize' issues the single noPupil verdict.
%
%   ep_pow = reset_xp_guard('is_powered', session)
%       true if nElt-1 is a POWERED optic (curved, |Kr|<<1e22, and not a
%       Reference/Return surface) -- a reset write there would clobber it.
%
%   moved = reset_xp_guard('check', session, xp0, moved, ep_pow, wf_elt)
%       call AFTER each field's macos.fex(1).  Compares the EP element
%       state to the pre-loop snapshot xp0; returns the running 'moved'
%       flag OR'd with this field's result.  ERRORS
%       (macos:dw_dx_multi:resetClobbersOptic) if the EP moved AND nElt-1
%       is a powered optic (restores xp0 first so the session is clean).
%
%   stamp = reset_xp_guard('finalize', reset_xp, moved, wf_elt)
%       call AFTER the field loop (with reset_xp true).  If the EP never
%       moved, WARNS once (macos:dw_dx_multi:resetNoEffect) and returns the
%       stamp 'no-effect'; otherwise returns reset_xp unchanged.  The stamp
%       goes into out.reset_xp so run_compare's convention check sees the
%       true state (true | false | 'no-effect').
%
%   The error/warning identifiers use the dw_dx_multi: prefix for all four
%   supervisors -- one identifier family for the whole reset_xp behavior.

switch action
    case 'pupil_find'
        % pf = reset_xp_guard('pupil_find', session, Ffield, stop_elt,
        %                     xp_elt, pf_opts, xp0)
        % ONCE-PER-CONFIGURATION placement of the cone-convergence
        % best-fit exit-pupil sphere (design/src/pupil_find), for
        % reset_xp_method='pupil_find'.  A configuration lives in ENGINE
        % STATE, not the deck file, so the CONFIGURED state is saved to a
        % temp deck first; pupil_find then loads THAT deck, places the
        % sphere, and leaves the session on the configured geometry with
        % the sphere at xp_elt and the stop re-set at stop_elt (its own
        % macos.stop call) -- exactly the state the field loop needs.
        % The caller runs its field loop with the per-field 'fex' reset
        % OFF: the placed sphere is a FROZEN, field-set-wide reference
        % (an upgrade of the frozen-EP mode, NOT a per-field re-reference).
        session  = varargin{1};
        Ffield   = varargin{2};
        stop_elt = varargin{3};
        xp_elt   = varargin{4};
        pf_opts  = varargin{5};
        xp0      = varargin{6};   % PRISTINE deck EP (vpt/psi/rad), from the
                                  % supervisor's pre-loop macos.get_xp()
        has_selt = ~isempty(stop_elt) && isscalar(stop_elt) && stop_elt >= 1;
        if size(Ffield, 1) < 3
            error('macos:dw_dx_multi:pfNeedsFields', ...
                  ['reset_xp_method=''pupil_find'' needs >= 3 field ' ...
                   'points (the cone aperture IS the field set; %d given).'], ...
                  size(Ffield, 1));
        end
        if exist('pupil_find', 'file') ~= 2
            error('macos:dw_dx_multi:pfNotOnPath', ...
                  ['pupil_find is not on the path -- addpath ' ...
                   '<mmacos>/design/src (run_sensitivities does this ' ...
                   'automatically when the method is selected).']);
        end
        % Restore the PRISTINE deck EP before saving: without this, the
        % save_rx below captures the PREVIOUS configuration's pf-written
        % sphere at nElt-1 (the config snapshot/restore covers only the
        % configuration's own elements), so every configuration after the
        % first is fit AND traced on a compounded EP state.  Measured on
        % the zoom fixture: cfg 1 dep_rms 1.9e-3 / vtx-FEX 0.384, cfgs 2-5
        % all ~5.4e-3 / 0.254 with identical nominal maps.  Gated by
        % tPupilFindMethod/test_config_sphere_is_independent_of_predecessors.
        macos.set_xp(xp0.vpt, xp0.psi, xp0.rad);
        session.modify();
        tmp = [tempname '.in'];
        cu  = onCleanup(@() delete_silent_(tmp));
        macos.save_rx(tmp);                    % the CONFIGURED state
        if has_selt
            sarg = {'ep_elt', stop_elt, 'stop_elt', stop_elt};
        else
            % No element stop given: the DECK must carry its own stop.
            % save_rx round-trips the stop state (header ApStop= in both
            % forms), so the temp deck is the checkable authority -- and
            % pupil_find leaves a deck-declared stop in force (the
            % object-space / segmented-primary idiom, Luis 2026-08-26).
            if isempty(regexp(fileread(tmp), '^\s*ApStop\s*=', ...
                              'once', 'lineanchors'))
                error('macos:dw_dx_multi:pfNeedsStopElt', ...
                      ['reset_xp_method=''pupil_find'' needs an aperture ' ...
                       'stop: pass ''stop_elt'' (element stop), or give ' ...
                       'the deck an ApStop= header / set stop_obj_pos ' ...
                       '(object-space stop -- carried through save_rx).']);
            end
            sarg = {};
        end
        pf = pupil_find(tmp, Ffield, sarg{:}, 'xp_elt', xp_elt, ...
                        'place', true, 'init', false, pf_opts{:});
        session.modify();
        varargout{1} = pf;
        return

    case 'is_powered'
        session = varargin{1};
        ep   = session.num_elt() - 1;
        info = macos.get_elt_info(ep);
        if any(info.elt_id == [3, 8])       % Reference | Return -> safe
            varargout{1} = false;  return;
        end
        varargout{1} = abs(session.get_elt_kr(ep)) < 1e22;   % curved => powered

    case 'fex'
        % The per-field exit-pupil re-reference.  FEX needs an aperture
        % stop to define the chief ray; if none is set (Rx has no ApStop=
        % and the caller passed no stop_elt / stop_obj_pos) rethrow with
        % an actionable supervisor-level message.  (A stop-state preflight
        % is not viable: get_stop_info reports only element-based stops and
        % errors on an object-space stop, which FEX itself handles fine --
        % so catching FEX's own verdict is the only false-negative-free
        % check.)  The stop state is constant across the loop, so that
        % branch only ever fires on the first field.
        %   The no-pupil-element case is ABSORBED, not raised: the engine
        % used to decline the write silently and return PASS, and the
        % supervisors' contract is built on that -- the loop runs to
        % completion and 'finalize' reports the no-effect verdict once.
        % The engine's XpEltOrWarn guard now turns that silent decline
        % into a clean FAIL (macos:fex:noPupilElt); swallowing it here
        % keeps the SAME diagnosis rather than aborting the run.
        ax = 1;   % xp_fnd mode 1 = chief-ray axis (the default doctrine)
        if numel(varargin) >= 2 && strcmp(varargin{2}, 'centroid')
            ax = 0;   % mode 0 = beam-centroid axis (diagnostic opt-in)
        end
        try
            macos.fex(ax);
        catch me
            switch me.identifier
                case 'macos:fex:noPupilElt'
                    % nothing to reset -- 'finalize' warns noPupil
                case 'macos:fex:noStop'
                    error('macos:dw_dx_multi:noStop', ...
                        ['reset_xp=true re-references the exit pupil ' ...
                         'per field via FEX, which needs an aperture ' ...
                         'stop, but none is set.  Add "ApStop= 0 0 1" ' ...
                         'to the Rx header (or 0 0 0 for a stop at the ' ...
                         'primary), pass stop_elt / stop_obj_pos, or ' ...
                         'set reset_xp=false to keep the ' ...
                         'prescription''s frozen EP.']);
                otherwise
                    rethrow(me);
            end
        end

    case 'check'
        [session, xp0, moved, ep_pow] = varargin{1:4};
        xpk = macos.get_xp();
        if ~ep_states_equal_(xpk, xp0)
            moved = true;
            if ep_pow
                macos.set_xp(xp0.vpt, xp0.psi, xp0.rad);   % undo before abort
                session.modify();
                error('macos:dw_dx_multi:resetClobbersOptic', ...
                    ['reset_xp wrote a pupil reference onto elt %d, a ' ...
                     'POWERED optic -- this replaces a real mirror with a ' ...
                     'reference sphere.  Put a dedicated Reference/Return ' ...
                     'surface at nElt-1, set exit_pupil_elt to one, or use ' ...
                     'reset_xp=false.'], session.num_elt() - 1);
            end
        end
        varargout{1} = moved;

    case 'finalize'
        [reset_xp, moved, wf_elt] = varargin{1:3};
        stamp = reset_xp;
        if reset_xp && ~moved
            warning('macos:dw_dx_multi:noPupil', ...
                ['reset_xp=true but this Rx carries no exit-pupil element ' ...
                 'at nElt-1 (elt %d is not a Return/Reference surface), so ' ...
                 'FEX wrote nothing and reset_xp behaved as FROZEN.  Add a ' ...
                 'pupil (Telescope.add_pupil, or the FP-Return-before-' ...
                 'ExitPupil 2-pass recipe) or set reset_xp=false.'], wf_elt);
            stamp = 'no-effect';
        end
        varargout{1} = stamp;

    otherwise
        error('reset_xp_guard:action', 'unknown action ''%s''', action);
end
end


function tf = ep_states_equal_(a, b)
% EP element states equal to round-off (vpt/psi BaseUnits, rad scalar).
tf = norm(a.vpt(:) - b.vpt(:)) <= 1e-9 * max(1, norm(b.vpt(:))) ...
   && norm(a.psi(:) - b.psi(:)) <= 1e-12 ...
   && abs(a.rad - b.rad)        <= 1e-9 * max(1, abs(b.rad));
end

function delete_silent_(p)
if exist(p, 'file') == 2, delete(p); end
end
