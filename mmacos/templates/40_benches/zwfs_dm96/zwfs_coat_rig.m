function P = zwfs_coat_rig(P)
%ZWFS_COAT_RIG  Resolve the OAP coating knob for the rig the sheet builds.
%   P = zwfs_coat_rig(P) sets P.bench.coat_oap to the stack zwfs_run applies
%   (coat_oap_): 'auto' (the sheet's default) is the rig's own -- protected
%   aluminum on the OAP rig (bench.optics 'oap', DECIDED 2026-09-17 for that
%   rig), none on the lens rig, which has no mirrors to coat.  A coating
%   asked of the lens rig is ignored with ONE warning that names it
%   (zwfs_run:coat_oap_ignored) -- it refused the run until 2026-10-26
%   (PLAN_CONSOLIDATION 1.2 (nnn): a bare zwfs_run, and pdi_run with it,
%   stopped at its own defaults).  Called once, from zwfs_run's parse_.
%
%   See also: zwfs_params, zwfs_run.
oap = isfield(P.bench, 'optics') && strcmp(P.bench.optics, 'oap');
if ~isfield(P.bench, 'coat_oap') || isempty(P.bench.coat_oap)
    P.bench.coat_oap = 'none';
elseif strcmp(P.bench.coat_oap, 'auto')
    if oap, P.bench.coat_oap = 'protectedAl'; else, P.bench.coat_oap = 'none'; end
elseif ~oap && ~strcmp(P.bench.coat_oap, 'none')
    warning('zwfs_run:coat_oap_ignored', ...
        ['bench.coat_oap ''%s'' ignored: the lens rig (bench.optics ''lens'') ' ...
         'has no OAPs to coat -- set bench.optics ''oap'' to coat L1 and L2'], ...
        P.bench.coat_oap);
    P.bench.coat_oap = 'none';
end
end
