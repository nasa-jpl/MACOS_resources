function path = iodc2010_seq(L, P, p, fields_deg, path)
%IODC2010_SEQ  Write a green lens as a CODE V .seq for the cross-check in CODE V.
%   iodc2010_seq(L, P, p, [0 5 10], 'lens.seq') writes the prescription in
%   CODE V's own convention (radius positive with its centre to +z, thickness
%   to the next surface, the glass as Schott's catalog name NBK7_SCHOTT, the
%   stop as a dummy surface at its axial position with STO) and appends the
%   evaluation the 2010 problem used: WAV (RMS wavefront, chief-ray reference,
%   default = no tilt/focus removal) at the listed semi-fields, DIS for the
%   distortion, FIR for the paraxial EFL.  The .seq is for a CODE V user (Luis)
%   to run; its printed RMS per field lands in README "The CODE V column".
%   The CODE V grid: NRD 32 -> 32 rays across the pupil (CODE V's WAV default
%   is 20); our deck uses p.nGridpts.
arguments
    L struct
    P struct
    p struct
    fields_deg (1,:) double
    path (1,:) char
end
fid = fopen(path, 'w');  assert(fid > 0, 'iodc2010_seq: cannot write %s', path);
c = onCleanup(@() fclose(fid));
w = @(varargin) fprintf(fid, [varargin{1} '\n'], varargin{2:end});

glass_cv = @(g) tern_(strcmpi(g, 'air'), '', sprintf(' %s', 'NBK7_SCHOTT'));

w('RDM');
w('LEN');
w('TITLE ''IODC 2010 green lens: %s (MACOS challenges/iodc2010)''', L.name);
w('EPD %.6f', P.D);
w('DIM M');
w('WL %g', p.lambda_nm);
w('REF 1');
w('WTW 1');
w('XAN %s', sprintf('%g ', zeros(size(fields_deg))));
w('YAN %s', sprintf('%g ', fields_deg));
w('WTF %s', sprintf('%g ', ones(size(fields_deg))));
w('SO 0.0 1.0E20');

% surfaces in z order with the stop dummy spliced into its gap
z_s = L.stop(1);  a_s = L.stop(2);
semidia = L.semidia;  if isempty(semidia), semidia = repmat(p.semidia_default, 1, L.N); end
if z_s < L.z(1)
    w('S 0.0 %.10f', L.z(1) - z_s);
    w('  STO');  w('  CIR %.6f', a_s);
    stop_done = true;
else
    stop_done = false;
end
for k = 1:L.N
    if L.sgn(k) == 0, Rtxt = '0.0'; else, Rtxt = sprintf('%.10f', L.sgn(k) * L.R); end
    if k < L.N, t = L.t(k); else, t = P.z_img - L.z(k); end
    in_gap = ~stop_done && z_s >= L.z(k) && (k == L.N || z_s < L.z(k + 1));
    if in_gap
        w('S %s %.10f%s', Rtxt, z_s - L.z(k), glass_cv(L.glass{k}));
        w('  CIR %.6f', semidia(k));
        w('S 0.0 %.10f%s', t - (z_s - L.z(k)), glass_cv(L.glass{k}));
        w('  STO');  w('  CIR %.6f', a_s);
        stop_done = true;
    else
        w('S %s %.10f%s', Rtxt, t, glass_cv(L.glass{k}));
        w('  CIR %.6f', semidia(k));
    end
end
w('SI 0.0 0.0');
w('GO');
w('! --- the 2010 evaluation (Juergens 2006, Sec. 4.4-4.5): RMS WFE chief-ray reference,');
w('! --- tilt and focus not removed; distortion at the image plane given; paraxial EFL.');
w('FIR');
w('WAV');
w('NRD 32');
w('GO');
w('DIS');
w('GO');
end

function out = tern_(c, a, b)
if c, out = a; else, out = b; end
end
