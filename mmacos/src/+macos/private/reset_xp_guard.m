function varargout = reset_xp_guard(action, varargin)
%RESET_XP_GUARD  Shared no-effect/clobber guard for the reset_xp path.
%   The four dw_d*_multi supervisors re-reference the exit pupil per field
%   by calling macos.fex(1), which the engine honors ONLY when nElt-1 is a
%   Return(EltID 8) or Reference(EltID 3) surface -- for any other type it
%   declines to write (macos_cmd_loop.inc FEX handler: "Invalid element
%   type", and xp_fnd still returns PASS), so reset_xp is a silent no-op.
%   By design (Dave 2026-08-04): the write needs a dedicated exit-pupil
%   element at nElt-1 -- the structure add_pupil builds (2-pass: FP_return
%   + ExitPupil inserted before the FocalPlane).  A bare focal deck (a
%   powered optic at nElt-1, no pupil) has nothing to update.  This helper
%   detects that no-pupil case (and the dangerous case where a write lands
%   on a powered optic) so the supervisors report the truth.
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
    case 'is_powered'
        session = varargin{1};
        ep   = session.num_elt() - 1;
        info = macos.get_elt_info(ep);
        if any(info.elt_id == [3, 8])       % Reference | Return -> safe
            varargout{1} = false;  return;
        end
        varargout{1} = abs(session.get_elt_kr(ep)) < 1e22;   % curved => powered

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
