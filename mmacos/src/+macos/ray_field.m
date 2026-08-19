function rf = ray_field(srf)
%MACOS.RAY_FIELD  Per-ray complex E-field + geometry at an element.
%   rf = macos.ray_field(SRF) returns, on the N x N ray grid at element
%   SRF (N = model size), the per-ray complex electric field RayE(3,:)
%   plus the ray direction cosines, the element surface normal, and the
%   per-ray status.  Requires a polarized trace: call
%   macos.polarization('on') and trace before this.
%
%   Returns a struct with N x N arrays:
%     .Ex .Ey .Ez   complex E-field components (RayE)
%     .kx .ky .kz    ray direction cosines (RayDir)
%     .nx .ny .nz    element surface normal (broadcast to the grid)
%     .status        per-ray RayStatus (0=OK,1=Obscured,2=Miss,3=Bracket,
%                    4=MaxIter,5=Undef/no-ray); non-OK points have zero field
%
%   This is the substrate for Jones-pupil / polarization-aberration
%   analysis (build the 2x2 Jones per pupil point from two orthogonal
%   input-state traces, using the geometry for the transverse basis and
%   .status to mask vignetted points).
%
%   See also: macos.polarization, macos.coating.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
[Ex, Ey, Ez, kx, ky, kz, nx, ny, nz, status] = mmacos('ray_field', srf);
rf = struct('Ex', Ex, 'Ey', Ey, 'Ez', Ez, ...
            'kx', kx, 'ky', ky, 'kz', kz, ...
            'nx', nx, 'ny', ny, 'nz', nz, ...
            'status', status);
end
