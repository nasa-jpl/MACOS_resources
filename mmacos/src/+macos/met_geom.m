function g = met_geom()
%MACOS.MET_GEOM  Geometry of the loaded Rx's laser-metrology beams.
%   g = macos.met_geom() runs the engine metrology compute (METcalc) and
%   returns the global endpoints of every gauge beam declared by the
%   prescription's nMetPos / tMetElt / metBeamFlg keywords:
%       g.src_pts  3 x n  gauge source points (launchers), BaseUnits
%       g.tgt_pts  3 x n  gauge target points (fiducials), BaseUnits
%       g.src_elt  1 x n  element carrying each source point
%       g.tgt_elt  1 x n  element carrying each target point
%       g.n        beam count (0 when the Rx declares no metrology --
%                  all other fields empty, not an error)
%   Beam k here is gauge k in macos.met().l -- the engine enumerates both
%   the same way.  Points are in the CURRENT (perturbed) state.  Units
%   are BaseUnits to match macos.get_ray_info positions (plot together).
%
%   Backs macos.view_rx's MET-path layer: works for ANY loaded Rx, no
%   design-layer structs needed.
%
%   See also: macos.met, macos.view_rx, macos.design.met_view,
%             macos.design.add_met.
n = mmacos('met_calc');
if n == 0
    g = struct('src_pts', zeros(3,0), 'tgt_pts', zeros(3,0), ...
               'src_elt', zeros(1,0), 'tgt_elt', zeros(1,0), 'n', 0);
    return
end
[src, tgt, se, te] = mmacos('met_geom_get', n);
g = struct('src_pts', src, 'tgt_pts', tgt, ...
           'src_elt', double(se(:).'), 'tgt_elt', double(te(:).'), 'n', n);
end
