function [n, S] = iodc2010_glass(name, lambda_nm)
%IODC2010_GLASS  Refractive index of the challenge glasses (Schott Sellmeier).
%   n = iodc2010_glass('N-BK7', 532) returns the index at 532 nm.
%   [n, S] = ... also returns S.B (1x3), S.C (1x3, um^2), S.name.
%
%   The coefficients are Schott's data-sheet Sellmeier
%       n^2 - 1 = sum_i B_i L2 / (L2 - C_i),   L2 = lambda_um^2,
%   the formula tracesub.F getIndex evaluates for GlassType= Sellmeier.  The
%   six numbers here are the SAME six the deck emitter writes as GlassCoef=
%   and that macos_glass_list.txt carries on its N-BK7 / N-SF6 rows, so the
%   paraxial model, the independent tracer and the engine agree to round-off.
%
%   Provenance: Schott optical glass data sheets (N-BK7 517642.251,
%   N-SF6 805254.337), as tabulated by refractiveindex.info ("Schott" shelf).
%   Check values: n_d(N-BK7) = 1.51680, n_d(N-SF6) = 1.80518.
%
%   'air' returns 1.
arguments
    name (1,:) char
    lambda_nm (1,1) double {mustBePositive}
end
key = upper(strrep(strrep(name, '-', ''), ' ', ''));
switch key
    case 'NBK7'
        B = [1.03961212  0.231792344  1.01046945];
        C = [0.00600069867  0.0200179144  103.560653];
    case 'NSF6'
        B = [1.77931763  0.338149866  2.08734474];
        C = [0.0133714182  0.0617533621  174.01759];
    case 'AIR'
        n = 1;  S = struct('B', [], 'C', [], 'name', 'air');  return
    otherwise
        error('iodc2010_glass:unknown', ...
            'glass %s is not in the challenge table (N-BK7, N-SF6, air)', name);
end
L2 = (lambda_nm * 1e-3)^2;
n  = sqrt(1 + sum(B .* L2 ./ (L2 - C)));
S  = struct('B', B, 'C', C, 'name', name);
end
