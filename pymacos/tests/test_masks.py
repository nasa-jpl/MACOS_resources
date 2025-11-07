"""
    -------------------------------------------------------------------------------------------
[ ] Grating on Conic Base Srf.
    -------------------------------------------------------------------------------------------
    [ ] Rx_Mask_circ_001      circular mask, S2, R=5, dx=0, dy=0
"""

from dataclasses import dataclass
from itertools import chain  # to flatten lists
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import rx_data
from context import pymacos as _lib
from test_settings import _Tol


@pytest.fixture(scope="module",   # session, module, class, function,
                params=(128, ))   # check if changing the macos size is affecting computation:
def macos_setup(request):
    _lib.init(request.param)

    assert _lib._MODELSIZE == request.param
    assert _lib._SYSINIT
    return request.param

def rx_path(rx: str):
    return Path(".") / 'Rx' / rx


@dataclass
class SrfInfo:
    dx_fact: int  # Sign correction (psiElt <> TElt_z)
    line_id: int  # line # where in Element Rx "ApeVec" is defined
    line_id_obs: int  # line # where in Element Rx "nObs" is defined

RX_PARAMS = {"parabola": (rx_path("Rx_Mask_Parabolas.in"),
                            {2: SrfInfo(-1, 99-1, 101-1),      # xObs= -1e0  0e0  0e0
                             4: SrfInfo(+1, 148-1, 150-1)}),   # xObs= +1e0  0e0  0e0
             # ---------------------
             "parabola_glb": (rx_path("Rx_Mask_Parabolas_glb.in"),
                            {3: SrfInfo(-1, 124-1, 125-1),     # xObs= -1e0  0e0  0e0
                             5: SrfInfo(+1, 173-1, 174-1)}),   # xObs= +1e0  0e0  0e0
            }


def ray_pos_at_srf_in_tangent_plane(fid_macos: Path, lines, srf):

    # create modified Rx & load
    # fid_macos = session_dir / "tempo.in"
    with open(fid_macos, 'w') as fid:
        for item in lines:
            fid.write(f"{item}")
    img = _lib.load(fid_macos)

    # trace to img. srf (rtn srf)
    n_rays = _lib.traceWavefront(-3)[1]
    ray_pos, ray_dir, opl, ok_trace, ok_pass = _lib.getRayInfo(n_rays)

    # extract invalid ray index
    ray_pass = np.logical_and(ok_trace, ok_pass)

    # trace to surface with mask
    n_rays = _lib.traceWavefront(srf)[1]
    ray_pos, ray_dir, opl, ok_trace, ok_pass = _lib.getRayInfo(n_rays)

    # mapping information
    vpt = _lib.elt_vpt(srf)                  # ray_pos are given in glb
    csys = _lib.elt_csys(srf)[0][:3, :3, 0]  # csys, csys_lcs, csys_upd
    dp_glb2loc = np.dot(csys.T, vpt)

    # map pts into local CSYS & extract (x,y) positions
    pts = ray_pos[:, ray_pass].T - vpt[:, 0]
    pts = ((np.dot(csys.T, pts.T) + dp_glb2loc.reshape(-1, 1))[:2, :]).T

    return pts


class TestApeMasksCirc:
    """Check Circular Aperture Masks"""

    mask_sets = chain.from_iterable([[
                 (4.75, 0, 0),
                 *[(r, dx_, dy_)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for r in (2, 4, 6, 8, 10)],
                 *[(4.90, np.cos(w)*5.0, np.sin(w)*5.0)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),))
    @pytest.mark.parametrize("rad, dx, dy", mask_sets)
    def test_circ(self, macos_setup, session_dir, rad, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        dx_fact, line_id = info[srf].dx_fact, info[srf].line_id

        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update
        # update mask parameters
        lines[line_id-1] = f"ApType=  Circular\n"
        lines[line_id]   = f"ApVec=  {rad:23.16e} {dx*dx_fact:23.16e} {dy:23.16e}\n"
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        assert np.all(((pts[:, 0] - dx))**2 + ((pts[:, 1] - dy))**2 <= rad**2)


        # print(session_dir)
        # import matplotlib.pyplot as plt
        # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # x_, y_ = dx + np.cos(t)*rad, dy + np.sin(t)*rad
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()


class TestApeMasksEllipse:
    """Check Elliptical Aperture Masks
    """
    mask_sets = chain.from_iterable([[
                 (2, 4, 0, 0),
                 *[(1.5*side, side, dx_, dy_)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for side in (2.1, 4.1, 8.1, 10.1)],
                 *[(3, 5, np.cos(w)*4.75, np.sin(w)*4.75)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),))
    @pytest.mark.parametrize("a, b, dx, dy", mask_sets)
    def test_ellipse(self, macos_setup, session_dir, a, b, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        dx_fact, line_id = info[srf].dx_fact, info[srf].line_id

        # load Rx as txt
        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update
        # update mask parameters:  ApeVec = a, b, dx, dy
        lines[line_id-1] = f"ApType=  Elliptical\n"
        lines[line_id]   = f" ApVec= {a:23.16e} {b:23.16e} {dx*dx_fact:23.16e} {dy:23.16e}\n"
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        assert np.all((b*(pts[:, 0] - dx))**2 + (a*(pts[:, 1] - dy))**2 <= (a*b)**2)


        # print(session_dir)

        # import matplotlib.pyplot as plt
        # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # x_, y_ = dx + a*np.cos(t), dy + b*np.sin(t)
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()


class TestApeMasksRect:
    """Check Rectangular Aperture Masks
    """
    mask_sets = chain.from_iterable([[
                 (7, 7, 0, 0),
                 *[(1.5*side, side, dx_, dy_)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for side in (2.1, 4.1, 8.1, 10.1)
                      ],
                 *[(3, 5, np.cos(w)*4.75, np.sin(w)*4.75)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),
                                             ))
    @pytest.mark.parametrize("wx, wy, dx, dy", mask_sets)
    def test_rect(self, macos_setup, session_dir, wx, wy, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        dx_fact, line_id = info[srf].dx_fact, info[srf].line_id

        # load Rx as txt
        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update

        # update mask parameters:  ApeVec = xmin, xmax, ymin, ymax
        x = (dx*dx_fact - wx/2, dx*dx_fact + wx/2)
        xmin, xmax = min(x), max(x)

        y = (dy - wy/2, dy + wy/2)
        ymin, ymax = min(y), max(y)

        lines[line_id-1] = f"ApType=  Rectangular\n"
        lines[line_id]   = f" ApVec=  {xmin:23.16e} {xmax:23.16e} {ymin:23.16e} {ymax:23.16e}\n"
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        x = np.array((dx - wx/2, dx + wx/2), dtype=float)
        xmin, xmax = min(x), max(x)

        y = np.array((dy - wy/2, dy + wy/2), dtype=float)
        ymin, ymax = min(y), max(y)

        assert np.all(np.logical_and(pts[:, 0] >= xmin, pts[:, 0] <= xmax))
        assert np.all(np.logical_and(pts[:, 1] >= ymin, pts[:, 1] <= ymax))


        # print(session_dir)

        # import matplotlib.pyplot as plt
        # # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # # x_, y_ = dx + a*np.cos(t), dy + b*np.sin(t)
        # fig, ax = plt.subplots()
        # plt.plot(pts[:, 0], pts[:, 1], 'r.', ms=1)
        # # plt.plot(qq[:,0], qq[:,1], 'g.', ms=1)
        # plt.plot(*np.array([[xmin, ymin],[xmin, ymax],[xmax, ymax],[xmax,ymin],[xmin,ymin]]).T, 'k', lw=0.5,alpha=0.5)
        # # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()


class TestApeMasksPolygon:
    """Check Polygon Aperture Masks
       [ ] Approach
            => correct aperture shift when xObs(L) < 0
            => trace rays to image plane
            => get valid rays
            => trace to surface with mask
            => get ray-surface intersection points
            => check that all valid rays are inside aperture
    """
    @staticmethod
    def poly_lines(vertices: np.ndarray):
        """returns lin. equations for testing"""
        n = vertices.shape[0]

        def abc(v1, v2):
            tol = 3e-16
            if np.abs(v2[1]-v1[1]) <= tol:
                a, b, c = 0, 1, -v1[1]
            else:
                if np.abs(v2[0]-v1[0]) <= tol:
                    a, b, c = 1, 0, -v1[0]
                else:
                    m = (v2[1]-v1[1])/(v2[0]-v1[0])
                    a, b, c = -m, 1, v1[0]*m-v1[1]
            return a, b, c

        bounds = np.zeros((n, 3), dtype=float)
        for i, __ in enumerate(vertices):
            v1 = vertices[i, :]
            v2 = vertices[i-1, :]  # if i>0 else vertices[-1, :]
            bounds[i, :] = abc(v1, v2)
        return bounds

    @staticmethod
    def chk_pts(pts, dx, dy, bounds):
        state = False
        ref = np.zeros(2, dtype=float)
        x = pts[:, 0] - dx
        y = pts[:, 1] - dy
        for bound in bounds:
            pt_ref = ref[0]*bound[0] + ref[1]*bound[1] + bound[2]
            pts_ = x*bound[0] + y*bound[1] + bound[2]

            state_a = np.logical_and(pt_ref>=0, np.any(pts_ < 0))
            state_b = np.logical_and(pt_ref<=0, np.any(pts_ > 0))
            state =  not np.logical_or(state_a, state_b)
            if not state:
                return state
        return state

    # ----------------------------------------------------- hexagon via polygon

    @staticmethod
    def hexagon(s:float, dx:float=0, dy:float=0) -> np.ndarray:
        """Top-flat"""
        h = np.sqrt(3)/2*s
        v = np.array([[+s, 0], [+s/2, +h], [-s/2, +h],
                      [-s, 0], [-s/2, -h], [+s/2, -h]], dtype=float)
        return v+np.array([dx, dy], dtype=float)

    mask_sets = chain.from_iterable([[
                #   (2, 5, 0,  0),
                 *[(side, dx_, dy_) for dx_ in (-1, 0, 1)
                                    for dy_ in (-1, 0, 1)
                                    for side in (2, 4, 6, 8)],
                 *[(5, np.cos(w)*4.75, np.sin(w)*4.75)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),))
    @pytest.mark.parametrize("side, dx, dy", mask_sets)
    def test_hexagon(self, macos_setup, session_dir, side, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        dx_fact, line_id = info[srf].dx_fact, info[srf].line_id

        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # update mask parameters:  ApeVec = dx dy nVertices
        #                                   x1 y1
        #                                   ...
        #                                   xn yn
        v = self.hexagon(side,dx=0, dy=0)  # vertices
        bounds = self.poly_lines(v)        # boundary lines

        lines[line_id-1] = f"ApType=  Polygonal\n"
        lines[line_id]   = (f" ApVec=  {dx*dx_fact:3.21e} {dy:3.21e} 6\n"
                            f"         {v[0, 0]:3.21e} {v[0, 1]:3.21e}\n"
                            f"         {v[1, 0]:3.21e} {v[1, 1]:3.21e}\n"
                            f"         {v[2, 0]:3.21e} {v[2, 1]:3.21e}\n"
                            f"         {v[3, 0]:3.21e} {v[3, 1]:3.21e}\n"
                            f"         {v[4, 0]:3.21e} {v[4, 1]:3.21e}\n"
                            f"         {v[5, 0]:3.21e} {v[5, 1]:3.21e}\n"
                           )
        # print(lines[line_id])

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        with open(fid_macos, 'w') as fid:
            for item in lines:
                fid.write(f"{item}")
        img = _lib.load(fid_macos)

        # trace to img. srf (rtn srf)
        n_rays = _lib.traceWavefront(-3)[1]
        ray_pos, ray_dir, opl, ok_trace, ok_pass = _lib.getRayInfo(n_rays)

        # extract valid ray index
        ray_pass = np.logical_and(ok_trace, ok_pass)

        # trace to surface with mask
        n_rays = _lib.traceWavefront(srf)[1]
        ray_pos, ray_dir, opl, ok_trace, ok_pass = _lib.getRayInfo(n_rays)
        pts = ray_pos[:2, ray_pass].T

        # mapping information
        vpt = _lib.elt_vpt(srf)  # ray_pos are given in glb
        csys = _lib.elt_csys(srf)[0][:3, :3, 0]  # csys, csys_lcs, csys_upd
        dp_glb2loc = np.dot(csys.T, vpt)

        # map pts into local CSYS
        pts = ray_pos[:, ray_pass].T - vpt[:, 0]
        pts = ((np.dot(csys.T, pts.T) + dp_glb2loc.reshape(-1, 1))[:2, :]).T

        # check ray sets if they are within bounds
        assert self.chk_pts(pts, dx, dy, bounds)

        # print(session_dir)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # plt.plot(v[:, 0]+dx*dx_fact, v[:, 1]+dy)
        # plt.axis('equal')
        # plt.show()

    # ----------------------------------------------------- rectangular via polygon

    @staticmethod
    def rectangular(wx:float, wy:float, dx:float=0, dy:float=0) -> np.ndarray:
        hx, hy = wx/2, wy/2
        v = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]], dtype=float)
        return v+np.array([dx, dy], dtype=float)


    mask_sets = chain.from_iterable([[
                #   (4, 5, 0,  0),
                 *[(3, 5, dx_, dy_) for dx_ in (-1, 0, 1)
                                    for dy_ in (-1, 0, 1)],
                 *[(3, 5, np.cos(w)*4.75, np.sin(w)*4.75)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),))
    @pytest.mark.parametrize("wx, wy, dx, dy", mask_sets)
    def test_poly_rect(self, macos_setup, session_dir, wx, wy, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        dx_fact, line_id = info[srf].dx_fact, info[srf].line_id

        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # update mask parameters:  ApeVec = dx dy nVertices
        #                                   x1 y1
        #                                   ...
        #                                   xn yn
        v = self.rectangular(wx, wy, dx=0, dy=0)  # vertices
        bounds = self.poly_lines(v)        # boundary lines

        lines[line_id-1] = f"ApType=  Polygonal\n"
        lines[line_id]   = (f" ApVec=  {dx*dx_fact:3.21e} {dy:3.21e} 4\n"
                            f"         {v[0, 0]:3.21e} {v[0, 1]:3.21e}\n"
                            f"         {v[1, 0]:3.21e} {v[1, 1]:3.21e}\n"
                            f"         {v[2, 0]:3.21e} {v[2, 1]:3.21e}\n"
                            f"         {v[3, 0]:3.21e} {v[3, 1]:3.21e}\n"
                           )

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        with open(fid_macos, 'w') as fid:
            for item in lines:
                fid.write(f"{item}")
        img = _lib.load(fid_macos)

        # trace to img. srf (rtn srf)
        n_rays = _lib.traceWavefront(-3)[1]
        ray_pos, ray_dir, opl, ok_trace, ok_pass = _lib.getRayInfo(n_rays)

        # extract valid ray index
        ray_pass = np.logical_and(ok_trace, ok_pass)

        # trace to surface with mask
        n_rays = _lib.traceWavefront(srf)[1]
        ray_pos, ray_dir, opl, ok_trace, ok_pass = _lib.getRayInfo(n_rays)
        pts = ray_pos[:2, ray_pass].T

        # mapping information
        vpt = _lib.elt_vpt(srf)  # ray_pos are given in glb
        csys = _lib.elt_csys(srf)[0][:3, :3, 0]  # csys, csys_lcs, csys_upd
        dp_glb2loc = np.dot(csys.T, vpt)

        # map pts into local CSYS
        pts = ray_pos[:, ray_pass].T - vpt[:, 0]
        pts = ((np.dot(csys.T, pts.T) + dp_glb2loc.reshape(-1, 1))[:2, :]).T

        # check ray sets if they are within bounds
        assert self.chk_pts(pts, dx, dy, bounds)

        # print(session_dir)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # plt.plot(v[:, 0]+dx*dx_fact, v[:, 1]+dy)
        # plt.axis('equal')
        # plt.show()


    # ----------------------------------------------------- polygon via PolyApVec
    #
    # 4.2) This is an extension to 4.1). Now polygon aperture vertices
    #      in global coords can be entered directly into MACOS prescription.
    #      Here is the syntax
    #
    #      ...
    #      ApType= Polygonal
    #      PolyApVec= ctr_x ctr_y ctr_z nVertex  % optional, aperture "center" coords and # vertices
    #                 v1_x  v1_y  v1_z
    #                 v2_x  v2_y  v2_z
    #                 ...
    #
    #      MACOS will internally convert the global coords of vertices into
    #      their x and y coords in the aperture frame.

    # @staticmethod
    # def map_pts(pts, ctm):
    #     """
    #     pts: np.ndarray shape is [n_pts x 3]
    #     ctm: np.ndarray Coordinate Transformation Matrix: 3 x 4
    #     """
    #     cs, po = ctm[:, :3], ctm[:, -1, None]
    #     return (np.dot(cs, pts.T) + po).T

    # @staticmethod
    # def hexagon_3d(s:float, vpt: np.ndarray, dx:float=0, dy:float=0) -> np.ndarray:
    #     """Top-flat"""
    #     h = np.sqrt(3)/2*s
    #     v = np.array([[+s, 0, 0], [+s/2, +h, 0], [-s/2, +h, 0],
    #                   [-s, 0, 0], [-s/2, -h, 0], [+s/2, -h, 0]], dtype=float)
    #    return v+np.array([dx, dy], dtype=float)

    mask_sets = chain.from_iterable([[
                  (7, 0,  0),
                 *[(side, dx_, dy_) for dx_ in (-1, 0, 1)
                                    for dy_ in (-1, 0, 1)
                                    for side in (2, 4, 6, 8, 10)],
                 *[(5, np.cos(w)*4.75, np.sin(w)*4.75)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("vertixes_dz", (0, 75, -75))
    @pytest.mark.parametrize("srf, rx_tag", (#(2, "parabola"),  # Rx is not loaded in this case
                                             #(4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),
                                             ))
    @pytest.mark.parametrize("side, dx, dy", mask_sets)
    def test_hexagon_glb(self, macos_setup, session_dir, side, dx, dy, srf, rx_tag, vertixes_dz):

        rx_src, info = RX_PARAMS[rx_tag]
        dx_fact, line_id = info[srf].dx_fact, info[srf].line_id

        # Extract Data at Mask Location
        img = _lib.load(rx_src)
        vpt = _lib.elt_vpt(srf)
        csys = _lib.elt_csys(srf)[0][:3, :3, 0]  # csys, csys_lcs, csys_upd

        # load Rx as txt
        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        if 1==0:
            # update mask parameters:  ApeVec = dx dy nVertices
            #                                   x1 y1
            #                                   ...
            #                                   xn yn
            v = self.hexagon(side,dx=0, dy=0)  # vertices
            bounds = self.poly_lines(v)        # boundary lines

            lines[line_id-1] = f"ApType=  Polygonal\n"
            lines[line_id]   = (f" ApVec=  {dx*dx_fact:23.16e} {dy:23.16e} 6\n"
                                f"         {v[0, 0]:23.16e} {v[0, 1]:23.16e}\n"
                                f"         {v[1, 0]:23.16e} {v[1, 1]:23.16e}\n"
                                f"         {v[2, 0]:23.16e} {v[2, 1]:23.16e}\n"
                                f"         {v[3, 0]:23.16e} {v[3, 1]:23.16e}\n"
                                f"         {v[4, 0]:23.16e} {v[4, 1]:23.16e}\n"
                                f"         {v[5, 0]:23.16e} {v[5, 1]:23.16e}\n"
                            )

        else:
            # update mask parameters:
            #
            #      PolyApVec= nVertex
            #                 v1_x  v1_y  v1_z   <== in glb CSYS
            #                 v2_x  v2_y  v2_z
            #                 ...

            v = self.hexagon(side, dx=0, dy=0)  # vertices (in tangent plane)
            bounds = self.poly_lines(v)        # boundary lines

            # Map mask from Local CSYS to Glb. CSYS
            shift = np.array([dx*dx_fact, dy, 0]).reshape(-1, 1)  # in Local CSYS
            vs = np.dot(csys, shift + np.c_[v, [vertixes_dz, ]*6].T).T + vpt.T

            lines[line_id-1] =  f"      ApType=  Polygonal\n"
            lines[line_id]   = (f"   PolyApVec=  6\n"
                                f"              {vs[0, 0]:23.16e} {vs[0, 1]:23.16e} {vs[0, 2]:23.16e}\n"
                                f"              {vs[1, 0]:23.16e} {vs[1, 1]:23.16e} {vs[1, 2]:23.16e}\n"
                                f"              {vs[2, 0]:23.16e} {vs[2, 1]:23.16e} {vs[2, 2]:23.16e}\n"
                                f"              {vs[3, 0]:23.16e} {vs[3, 1]:23.16e} {vs[3, 2]:23.16e}\n"
                                f"              {vs[4, 0]:23.16e} {vs[4, 1]:23.16e} {vs[4, 2]:23.16e}\n"
                                f"              {vs[5, 0]:23.16e} {vs[5, 1]:23.16e} {vs[5, 2]:23.16e}\n"
                               )


        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check ray sets if they are within bounds
        assert self.chk_pts(pts, shift[0, 0], shift[1, 0], bounds)

        # print(session_dir)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # plt.plot(v[:, 0]+shift[0,0], v[:, 1]+shift[1,0])
        # plt.axis('equal')
        # plt.show()
        # plt.show()


class TestObsMasksCirc:
    """Check Circular Aperture Masks"""

    mask_sets = chain.from_iterable([[
                 (4.75, 0, 0),
                 *[(r, dx_, dy_)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for r in (4.1, 8.1, 10.1)],
                 *[(4.90, np.cos(w)*5.0, np.sin(w)*5.0)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),
                                             ))
    @pytest.mark.parametrize("rad, dx, dy", mask_sets)
    def test_circ(self, macos_setup, session_dir, rad, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        info = info[srf]
        dx_fact, ln_ape, ln_obs = info.dx_fact, info.line_id, info.line_id_obs

        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update
        # no aperture mask
        lines[ln_ape-1] = f"ApType=  None\n"
        lines[ln_ape]   = f"\n"

        # obscuration mask
        lines[ln_obs] = (f"   nObs=  1\n"
                          "ObsType=  Circular\n"
                         f" ObsVec=  {rad:23.16e} {dx*dx_fact:23.16e} {dy:23.16e}\n"
                         )
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        assert np.all(((pts[:, 0] - dx))**2 + ((pts[:, 1] - dy))**2 >= rad**2)


        # print(session_dir)

        # import matplotlib.pyplot as plt
        # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # x_, y_ = dx + np.cos(t)*rad, dy + np.sin(t)*rad
        # fig, ax = plt.subplots()
        # plt.plot(pts[:, 0], pts[:, 1], 'r.', ms=1)
        # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()


class TestObsMasksEllipse:
    """Check Elliptical Aperture Masks
    """
    mask_sets = chain.from_iterable([[
                 (2, 4, 0, 0),
                 *[(1.5*side, side, dx_, dy_)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for side in (1, 2, 4, 6, 8, 10.1)],
                 *[(3, 5, np.cos(w)*4.75, np.sin(w)*4.75)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),))
    @pytest.mark.parametrize("a, b, dx, dy", mask_sets)
    def test_ellipse(self, macos_setup, session_dir, a, b, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        info = info[srf]
        dx_fact, ln_ape, ln_obs = info.dx_fact, info.line_id, info.line_id_obs

        # load Rx as txt
        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update
        # no aperture mask
        lines[ln_ape-1] = f"ApType=  None\n"
        lines[ln_ape]   = f"\n"

        # obscuration mask: Elliptical (a, b, xc, yc)
        lines[ln_obs] = (f"   nObs=  1\n"
                          "ObsType=  Elliptical\n"
                         f" ObsVec=  {a:23.16e} {b:23.16e} {dx*dx_fact:23.16e} {dy:23.16e}\n"
                         )
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        assert np.all((b*(pts[:, 0] - dx))**2 + (a*(pts[:, 1] - dy))**2 >= (a*b)**2)

        # print(session_dir)

        # import matplotlib.pyplot as plt
        # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # x_, y_ = dx + a*np.cos(t), dy + b*np.sin(t)
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()

    mask_sets = chain.from_iterable([[
                 (2, 4, 0, 0, 30),
                 *[(1.5*side, side, dx_, dy_, phi)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for side in (4, 8, 10.1)
                      for phi in np.linspace(-6, 354, num=12, endpoint=False)],
                 *[(3, 5, np.cos(w)*4.75, np.sin(w)*4.75, phi)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)
                     for phi in np.linspace(-6, 354, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),
                                             ))
    @pytest.mark.parametrize("a, b, dx, dy, phi", mask_sets)
    def test_ellipse_rot(self, macos_setup, session_dir, a, b, dx, dy, phi, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        info = info[srf]
        dx_fact, ln_ape, ln_obs = info.dx_fact, info.line_id, info.line_id_obs

        # load Rx as txt
        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update
        # no aperture mask
        lines[ln_ape-1] = f"ApType=  None\n"
        lines[ln_ape]   = f"\n"

        # obscuration mask: RotEllipse (A B DX DY Rot_Angle)
        phi_ = phi*np.pi/180
        lines[ln_obs] = (f"   nObs=  1\n"
                          "ObsType=  RotElliptical\n"
                         f" ObsVec=  {a:23.16e} {b:23.16e} {dx*dx_fact:23.16e} {dy:23.16e} {phi_:23.16e}\n"
                         )
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        # inverse mask rotation and shift
        c, s = np.cos(phi_*dx_fact), np.sin(phi_*dx_fact)
        pp = np.dot(np.array([[c, s], [-s, c]]), (pts - [dx, dy]).T).T # - np.array([dx, dy])

        assert np.all((b*(pp[:, 0]))**2 + (a*(pp[:, 1]))**2 >= (a*b)**2)

        # print(session_dir)

        # import matplotlib.pyplot as plt
        # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # x_, y_ = dx + a*np.cos(t), dy + b*np.sin(t)
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # # plt.plot(pp[:,0], pp[:,1], 'r.', ms=1)
        # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()


class TestObsMasksRect:
    """Check Rectangular Aperture Masks
    """
    mask_sets = chain.from_iterable([[
                 (7, 7, 0, 0),
                 *[(1.5*side, side, dx_, dy_)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for side in (2.1, 4.1, 8.1, 10.1)],
                 *[(3, 5, np.cos(w)*4.75, np.sin(w)*4.75)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)],
                ]])
    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),))
    @pytest.mark.parametrize("wx, wy, dx, dy", mask_sets)
    def test_obs_rect(self, macos_setup, session_dir, wx, wy, dx, dy, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        info = info[srf]
        dx_fact, ln_ape, ln_obs = info.dx_fact, info.line_id, info.line_id_obs

        # load Rx as txt
        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update
        # no aperture mask
        lines[ln_ape-1] = f"ApType=  None\n"
        lines[ln_ape]   = f"\n"

        # obscuration mask: ObsVec = xmin, xmax, ymin, ymax
        x = (dx*dx_fact - wx/2, dx*dx_fact + wx/2)
        xmin, xmax = min(x), max(x)

        y = (dy - wy/2, dy + wy/2)
        ymin, ymax = min(y), max(y)

        lines[ln_obs] = (f"   nObs=  1\n"
                          "ObsType=  Rectangular\n"
                         f" ObsVec=  {xmin:23.16e} {xmax:23.16e} {ymin:23.16e} {ymax:23.16e}\n"
                         )
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        x = np.array((dx - wx/2, dx + wx/2), dtype=float)
        xmin, xmax = min(x), max(x)

        y = np.array((dy - wy/2, dy + wy/2), dtype=float)
        ymin, ymax = min(y), max(y)

        x_, y_ = pts[:, 0], pts[:, 1]
        assert np.all(np.logical_or(np.logical_or(x_ <= xmin, x_ >= xmax),
                                    np.logical_or(y_ <= ymin, y_ >= ymax)))

        # print(session_dir)

        # import matplotlib.pyplot as plt
        # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # x_, y_ = dx + a*np.cos(t), dy + b*np.sin(t)
        # fig, ax = plt.subplots()
        # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()


    mask_sets = chain.from_iterable([[
                 (5, 9, 0, 0, 0),
                 *[(1.5*side, side, dx_, dy_, phi)
                      for dx_ in (-1, 0, 1)
                      for dy_ in (-1, 0, 1)
                      for side in (4, 8, 10.1)
                      for phi in np.linspace(-6, 354, num=12, endpoint=False)],
                 *[(3, 5, np.cos(w)*4.75, np.sin(w)*4.75, phi)
                     for w in np.linspace(0, 2*np.pi, num=12, endpoint=False)
                     for phi in np.linspace(-6, 354, num=12, endpoint=False)],
                 ]])

    @pytest.mark.parametrize("srf, rx_tag", ((2, "parabola"),
                                             (4, "parabola"),
                                             (3, "parabola_glb"),
                                             (5, "parabola_glb"),
                                             ))
    @pytest.mark.parametrize("wx, wy, dx, dy, phi", mask_sets)
    def test_obs_rect_rot(self, macos_setup, session_dir, wx, wy, dx, dy, phi, srf, rx_tag):

        rx_src, info = RX_PARAMS[rx_tag]
        info = info[srf]
        dx_fact, ln_ape, ln_obs = info.dx_fact, info.line_id, info.line_id_obs

        # load Rx as txt
        with open(rx_src, 'r') as fid:
            lines = fid.readlines()

        # ------------------------------------ Mask Update
        # no aperture mask
        lines[ln_ape-1] = f"ApType=  None\n"
        lines[ln_ape]   = f"\n"

        # obscuration mask: ObsVec = W H DX DY Rot_Angle [radian] (cw)
        phi_ = phi*np.pi/180
        lines[ln_obs] = (f"   nObs=  1\n"
                          "ObsType=  RotRectangular\n"
                         f" ObsVec=  {wx:23.16e} {wy:23.16e} {dx*dx_fact:23.16e} {dy:23.16e} {phi_:23.16e}\n"
                         )
        # ------------------------------------

        # create modified Rx & load
        fid_macos = session_dir / "tempo.in"
        pts = ray_pos_at_srf_in_tangent_plane(fid_macos, lines, srf)

        # check
        x = np.array((-wx/2, +wx/2), dtype=float)
        xmin, xmax = min(x), max(x)

        y = np.array((-wy/2, +wy/2), dtype=float)
        ymin, ymax = min(y), max(y)

        # inverse mask rotation and shift
        c, s = np.cos(phi_*dx_fact), np.sin(phi_*dx_fact)
        pp = np.dot(np.array([[c, s], [-s, c]]), (pts - [dx, dy]).T).T

        x_, y_ = pp[:, 0], pp[:, 1]
        assert np.all(np.logical_or(np.logical_or(x_ <= xmin, x_ >= xmax),
                                    np.logical_or(y_ <= ymin, y_ >= ymax)))


        # print(session_dir)

        # import matplotlib.pyplot as plt
        # t = np.linspace(0, 2*np.pi, num=501, endpoint=True)
        # # x_, y_ = dx + a*np.cos(t), dy + b*np.sin(t)
        # fig, ax = plt.subplots()
        # # plt.plot(pts[:,0], pts[:,1], 'r.', ms=1)
        # plt.plot(pp[:,0], pp[:,1], 'r.', ms=1)
        # # plt.plot(np.array([[xmin, ymin],[xmin, ymax],[xmax, ymax]]), 'ko', lw=1)
        # plt.plot(*np.array([[xmin, ymin],[xmin, ymax],[xmax, ymax],[xmax,ymin],[xmin,ymin]]).T, 'k', lw=1)
        # # plt.plot(x_, y_, 'k', lw=0.2)
        # plt.axis('equal')
        # plt.show()
