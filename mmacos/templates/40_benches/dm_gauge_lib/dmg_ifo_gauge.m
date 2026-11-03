function IFO = dmg_ifo_gauge(AT, AR, QWP, THETAS, LAM)
%DMG_IFO_GAUGE  Four-step polarization-PSI measurement factory (TG96).
%   IFO = dmg_ifo_gauge(AT, AR, QWP, THETAS, LAM) measures the two arms'
%   analyzer bases on the flat DM, forms the null-phase reference, and
%   returns:
%     IFO.meas(M)   surface-height map (mm) of test-optic grid M via the
%                   four-step about the frozen null reference
%     IFO.msk       detector support (I0 > 0.1 max)
%     IFO.I0, IFO.p_null, IFO.Sr
%   AT/AR from dmg_arm_desc.  Six traces per measurement (3 analyzer
%   states x re-derived synth), as the campaign record states.
%   Extracted verbatim from tg96_s3/tg96_eprime @ 10cf593.
Sr = analyzer_basis(AR, QWP, []);
S0 = analyzer_basis(AT, QWP, []);
I0 = frame(S0, Sr, 0);
p_null = fourstep(S0, Sr, THETAS);
IFO = struct();
IFO.msk    = I0 > 0.1*max(I0(:));
IFO.I0     = I0;
IFO.p_null = p_null;
IFO.Sr     = Sr;
IFO.meas   = @(M) meas_surface(AT, QWP, M, Sr, p_null, THETAS, LAM);
% ---- frame-level access (noise stage): the four analyzer frames for a
% DM state + the pure-MATLAB reconstruction, so shot noise can be
% injected between capture and reconstruction without re-tracing.
IFO.frames = @(M) frames4_(analyzer_basis(AT, QWP, M), Sr, THETAS);
IFO.recon  = @(Fr) angle(exp(1i*(atan2(Fr(:,:,2)-Fr(:,:,4), ...
                 Fr(:,:,1)-Fr(:,:,3)) - p_null))) * LAM/(4*pi);
end

function Fr = frames4_(Sx, Sr, th)
Fr = cat(3, frame(Sx,Sr,th(1)), frame(Sx,Sr,th(2)), ...
            frame(Sx,Sr,th(3)), frame(Sx,Sr,th(4)));
end

% ==== the PSI chain (verbatim) ========================================
function a = lax(psi, deg)
u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function load_arm(A, QWP, an_deg, grid)
macos.load_rx(A.rx);  b = A.b;
if nargin >= 4 && ~isempty(grid)
    macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
end
macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg, grid)
load_arm(A, QWP, an_deg, grid);
E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
           macos.complex_field(A.iDET,'plane',2), ...
           macos.complex_field(A.iDET,'plane',3));
end

function S = analyzer_basis(A, QWP, grid)
E0  = arm_field(A, QWP,  0, grid);
E45 = arm_field(A, QWP, 45, grid);
E90 = arm_field(A, QWP, 90, grid);
S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
c = cosd(th);  s = sind(th);
E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function I = frame(Sx, Sr, th)
I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
p  = atan2(I2-I4, I1-I3);
end

function h = meas_surface(A, QWP, M, Sr, p_null, THETAS, LAM)
d = angle(exp(1i*(fourstep(analyzer_basis(A, QWP, M), Sr, THETAS) - p_null)));
h = d * LAM/(4*pi);
end
