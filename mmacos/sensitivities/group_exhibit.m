function group_exhibit(out, groups, report_path, opts)
%GROUP_EXHIBIT  Append a group-vs-member sensitivity table to a report.
%   group_exhibit(OUT, GROUPS, REPORT_PATH) writes, for each rigid-body
%   element GROUP in a dw_dx / dw_dx_multi harvest, the group's six
%   column norms beside those of its member elements, and the
%   group/member ratio -- the number a template exists to show.  The
%   table is APPENDED to REPORT_PATH (run_sensitivities' own
%   <name>_sens_report.txt) so the committed artifact carries every
%   figure a README or driver header quotes.
%
%   A ratio BELOW 1 is intra-group COMPENSATION: the members' responses
%   partly cancel when they move as one body, so a per-member budget
%   OVERSTATES the assembly's sensitivity in that DOF.  A ratio at ~N
%   (the member count) is the opposite and equally expected: a rigid
%   TRANSLATION of N alike members is the sum of N alike columns.
%
%   Units: group and per-element columns share one convention --
%   OPD-per-metre for translations, OPD-per-rad for rotations -- so the
%   two sides are directly comparable and nothing is rescaled here.
%
%   OUT     struct from macos.dw_dx or macos.dw_dx_multi.  Needs
%           channel_names, kind, cbm and a Jacobian (dwdxall or dwdx).
%   GROUPS  the containers.Map handed to the harvest (name -> members).
%   OPTIONS
%     'members'  ids to tabulate ([] = every member).  Use it when a
%                group has many alike members and one stands for the
%                rest -- an 18-segment PM does not need 18 rows.
%     'ratio_to' the member id the ratio column is taken against
%                ([] = the first tabulated member).
%
%   See also: run_sensitivities, macos.dw_dx_multi, save_dw_flat.

arguments
    out (1,1) struct
    groups
    report_path (1,:) char
    opts.members double = []
    opts.ratio_to double = []
end

if ~isa(groups, 'containers.Map') || groups.Count == 0, return; end
if ~isfield(out, 'kind') || ~any(strcmp(out.kind, 'Group')), return; end
if isfield(out, 'dwdxall')
    A = out.dwdxall;
elseif isfield(out, 'dwdx')
    A = out.dwdx;
else
    return
end

LAB = {'Rx','Ry','Rz','Tx','Ty','Tz'};
% column RMS over the rows every channel reached
rmsn = @(M) sqrt(mean(M(all(isfinite(M), 2), :).^2, 1));

fid = fopen(report_path, 'a');
if fid < 0, return; end
closer = onCleanup(@() fclose(fid));
say = @(varargin) fprintf(1, varargin{:}) + fprintf(fid, varargin{:});

gnames = keys(groups);
for gi = 1:numel(gnames)
    nm  = gnames{gi};
    mem = double(groups(nm));  mem = mem(:);
    tag = sprintf('Grp[%s]', nm);
    gc  = find(strncmp(out.channel_names, tag, numel(tag)) ...
               & strcmp(out.kind(:), 'Group'));
    if numel(gc) < 6, continue; end

    show = opts.members;
    if isempty(show), show = mem; end
    show = intersect(show(:), mem, 'stable');
    ec = cell(numel(show), 1);
    for q = 1:numel(show)
        ec{q} = find(startsWith(out.channel_names, ...
                     sprintf('Elt %d ', show(q))));
    end
    keep = cellfun(@(v) numel(v) >= 6, ec);
    show = show(keep);  ec = ec(keep);
    if isempty(show), continue; end

    rref = opts.ratio_to;
    if isempty(rref), rref = show(1); end
    iref = find(show == rref, 1);
    if isempty(iref), iref = 1;  rref = show(1); end

    say('\n[%s exhibit] the GROUP against its member elements\n', nm);
    say('    members: %s (%d)\n', mat2str(reshape(mem, 1, [])), numel(mem));
    say(['    column RMS of dW/d(DOF): rotations in OPD-BaseUnits per ' ...
         'rad, translations\n    in OPD-BaseUnits per SI METRE -- the ' ...
         'same convention on both sides\n    (the OPD numerator is the ' ...
         'deck''s BaseUnits, as of 2026-08-25).\n']);
    hdr = sprintf('%-24s', 'channel');
    for d = 1:6, hdr = [hdr sprintf('%12s', LAB{d})]; end %#ok<AGROW>
    say('    %s\n', hdr);

    gv = rmsn(A(:, gc(1:6)));
    say('    %-24s%s\n', [tag ' (group)'], sprintf('%12.4e', gv));
    for q = 1:numel(show)
        ev = rmsn(A(:, ec{q}));
        say('    %-24s%s\n', sprintf('Elt %d (member)', show(q)), ...
            sprintf('%12.4e', ev));
        if q == iref
            say('    %-24s%s\n', sprintf('  group / Elt %d', rref), ...
                sprintf('%12.4f', gv ./ max(ev, realmin)));
        end
    end
    say(['    (a ratio BELOW 1 is intra-group COMPENSATION -- the ' ...
         'members'' responses\n     partly cancel when they move as one ' ...
         'body.  A ratio at ~%d, the member\n     count, is a rigid ' ...
         'motion adding up, which is equally expected.)\n'], numel(mem));
end
end
