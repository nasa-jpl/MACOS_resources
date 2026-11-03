function C = check_record()
%CHECK_RECORD  Re-measure the quantities the committed S4b record quotes.
%
%   The packaging study rests on numbers taken from committed decks, so the
%   decks and the prose that describes them have to agree.  This re-measures,
%   from the engine, the two things RESULTS.md section S4b.4 and
%   STATUS_S4B.md state about the folded demonstration:
%
%     * where the interface pupil ends up, and
%     * the z-slab the stated 1000 mm instrument envelope occupies.
%
%   NOTHING under challenges/afocal4 is modified.  This only reads.
%
%   See also PACK_LEGS, AFOCAL4_PACKAGING.

    here = fileparts(mfilename('fullpath'));
    up   = fileparts(here);
    addpath(here);  addpath(up);
    P = afocal4_params();

    C = struct();
    for d = {'afocal4_b_final.in', 'afocal4_b_final_folded.in'}
        f = fullfile(up, d{1});
        L = pack_legs(f, 'instr', P.pack.instr_len, 'quiet', true);
        k = L.nElt;
        p0 = L.vpt(:,k);
        a  = L.leg(end).d;
        p1 = p0 + a*P.pack.instr_len;
        hw = 0.5*P.pack.instr_dia;
        key = matlab.lang.makeValidName(erase(d{1},{'afocal4_','.in'}));
        C.(key) = struct('deck',f, 'names',{L.names}, 'z',L.z, ...
            'iface_vpt',p0, 'instr_end',p1, ...
            'instr_zslab',[min(p0(3),p1(3))-hw, max(p0(3),p1(3))+hw], ...
            'deepest', max(L.z), 'span_front', L.span_front_m);
        fprintf('\n  %s\n', d{1});
        fprintf('    elements:');
        for i = 1:L.nElt, fprintf('  %s %+0.4f', L.names{i}, L.z(i)); end
        fprintf('\n    interface plane vertex   [%+.4f %+.4f %+.4f] m\n', p0);
        fprintf('    instrument envelope ends [%+.4f %+.4f %+.4f] m\n', p1);
        fprintf('    instrument z-slab        %+.4f .. %+.4f m\n', C.(key).instr_zslab);
        fprintf('    deepest element behind M1 %+.4f m (M1-M2 %.4f m)\n', ...
                max(L.z), L.span_front_m);
    end

    fprintf(['\n  RESULTS.md section S4b.4 states, for the folded demonstration:\n' ...
             '    interface pupil at [+0.304, -0.004, +0.614] m,\n' ...
             '    envelope running to [+1.304, -0.017, +0.614] m,\n' ...
             '    z-slab +0.464 .. +0.764 m.\n' ...
             '  Compare the measured row above for afocal4_b_final_folded.in.\n']);
end
