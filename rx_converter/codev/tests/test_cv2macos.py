# from __future__ import absolute_import

# https://docs.python-guide.org/writing/structure/
# https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
import os
import sys
# from functools import partial
from itertools import chain  # to flatten lists
from pathlib import Path

import numpy as np
import pytest
import win32com.client
from context import PRJ_PATH, TOL, cv_setup
from context import pymacos as _lib
from context import rmTTP


@pytest.fixture(scope="module",  # session, module, class, function,
                params=(256, ))   # check if changing the macos size is affecting computation
def macos_setup(request):

    _lib.init(request.param)

    assert _lib.macos._MODELSIZE == request.param
    assert _lib.macos._SYSINIT
    return request.param

# def ape_apply_srf(default_srf=2):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             if "srf" not in kwargs:
#                 kwargs["srf"] = default_srf
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator


def ape_ellipse(a, b, xc, yc, srf=2, tag='CLR') -> str:
    """compose codev cmd for defining an elliptical aperture"""
    return (f"ELX s{srf} {tag} {a};ELY s{srf} {tag} {b};"
            f"ADX s{srf} {tag} {xc};ADY s{srf} {tag} {yc};"
            "ca ape;")


def ape_rectangle(a, b, xc, yc, srf=2, tag='CLR') -> str:
    """compose codev cmd for defining a rectangular aperture"""
    return (f"REX s{srf} {tag} {a};REY s{srf} {tag} {b};"
            f"ADX s{srf} {tag} {xc};ADY s{srf} {tag} {yc};"
            "ca ape;")


def ape_circle(r, xc, yc, srf=2, tag='CLR') -> str:
    """compose codev cmd for defining a circular aperture"""
    return (f"CIR s{srf} {tag} {r};"
            f"ADX s{srf} {tag} {xc};ADY s{srf} {tag} {yc};"
            "ca ape;")


def obs_rot_rectangular(rdy, a, b, xc, yc, phi=0, srf=2, tag='OBS') -> str:
    """codev cmd: rectangular obscuration with a clear circ. ape"""
    return (f"CIR CLR s{srf} {rdy};"
            f"REX s{srf} {tag} {a};REY s{srf} {tag} {b};"
            f"ADX s{srf} {tag} {xc};ADY s{srf} {tag} {yc};"
            f"ARO s{srf} {tag} {phi};"
            "ca ape;")

def obs_rot_ellipse(rdy, a, b, xc, yc, phi=0, srf=2, tag='OBS') -> str:
    """codev cmd: elliptical obscuration with a clear circ. ape"""
    return (f"CIR CLR s{srf} {rdy};"
            f"ELX s{srf} {tag} {a};ELY s{srf} {tag} {b};"
            f"ADX s{srf} {tag} {xc};ADY s{srf} {tag} {yc};"
            f"ARO s{srf} {tag} {phi};"
            "ca ape;")

def obs_circle(rdy, r, xc, yc, srf=2, tag='CLR') -> str:
    """compose codev cmd for defining a circular obscuration"""
    return (f"CIR CLR s{srf} {rdy};"
            f"CIR s{srf} {tag} {r};"
            f"ADX s{srf} {tag} {xc};ADY s{srf} {tag} {yc};"
            "ca ape;")

def ape_sweep(rx, n_sample, iwl, fct, par, dr, n_phi, srf=2, tag='CLR'):
    phi = np.linspace(0, 2*np.pi, num=n_phi, endpoint=False)
    return [(rx, n_sample, iwl, fct(*par, np.round(np.cos(w)*dr, 2).item(),
                                          np.round(np.sin(w)*dr, 2).item(),
                                          srf=srf, tag=tag))
           for w in phi]


def ape_sweep2(rx, n_sample, iwl, fct, par, dr, n_phi, **kwargs):
    phi = np.linspace(0, 2*np.pi, num=n_phi, endpoint=False)
    return [(rx, n_sample, iwl, fct(*par, np.round(np.cos(w)*dr, 2).item(),
                                          np.round(np.sin(w)*dr, 2).item(),
                                          **kwargs))
           for w in phi]


class NoTestGugus:

    rx_set = (
            # ('Rx_0000', 5, 1),
            # ('Rx_0000a', 11, 1),
            # ('Rx_0000c', 11, 1),
            #   ('Rx_0001a', 11, 1),  # small shifted cir. ape.
            #   ('Rx_0001b', 11, 1),
            #   ('Rx_0001c', 11, 1),
            #   ('Rx_0002', 11, 1),  # small shifted rect. ape.
            #   ('Rx_0002a', 11, 1),
            #   ('Rx_0002b', 11, 1),
            #   ('Rx_0002c', 11, 1),
            #   ('Rx_0003', 11, 1),
            #   ('Rx_0003a', 11, 1),
            #   ('Rx_0003b', 11, 1),
              ('Rx_0003c', 11, 1),
              ('Rx_0003d', 11, 1),
              )

    @pytest.mark.parametrize("rx, n_sampling, i_wl", rx_set)
    def test_demo6(self, cv_setup, session_dir, rx, n_sampling, i_wl):
        """ray-trace comparisons"""

        # spec
        # rx         = 'Rx_0000'   # CodeV RX
        # n_sampling = 11          # pupil sampling
        # i_wl       = 1           # wavelength ID

        i_fov      = 0   # for codev trace => collect for all FoV positions
        i_zoo      = 1

        # -------------------------------------------------------
        # CODE V: trace rays
        # -------------------------------------------------------
        rx_data = cv_setup.test_ray_trace(rx, n_sampling, i_wl, i_fov, i_zoo)

        n_rays = rx_data['n_rays']
        n_pts = np.ceil(np.sqrt(n_rays)) + 1

        model_size = max(128,
                         2**np.int32(np.ceil(np.log10(n_pts)/np.log10(2))))
        assert model_size < 2048

        # -------------------------------------------------------
        # CODE V: Rx conversion
        # -------------------------------------------------------
        fid_macos = session_dir / f'{rx}.in'
        cv_setup.cv2macos(fid_macos,
                          i_origin=rx_data['origin'],
                          i_wl=i_wl,
                          i_fov=1,
                          i_zoo=i_zoo)

        # -------------------------------------------------------
        # MACOS: trace rays
        # -------------------------------------------------------
        # update model size if needed
        if _lib.model_size() < model_size:
            _lib.init(model_size)
            assert _lib.model_size() == model_size

        # fid_macos = 'Rx_0000c_Tmp.in'   # debug
        _lib.load(fid_macos)

        # trace rays to Src. Ref. Srf.  &
        __ = _lib.traceWavefront(rx_data['entry_port'])[0]

        # upload input ray definitions from ref. code
        # ==> ensures the same ray input definition
        ray_in = rx_data['ray_in']
        _lib.setRayInfo(ray_in[:, :3].T, ray_in[:, 3:].T,
                        np.zeros(n_rays), np.ones(n_rays) )

        # trace to 'final' Srf.
        __, n_rays = _lib.traceWavefront(rx_data['exit_port'])[:2]
        p_macos, r_macos, opl_macos, ok_trace, ok_pass = _lib.getRayInfo(n_rays)

        # -------------------------------------------------------
        # data comparisons
        # -------------------------------------------------------
        # check for trace failures
        ok_trace_cv = rx_data['RER'] == 0
        assert ok_trace.shape == ok_trace_cv.shape
        assert np.all(ok_trace == ok_trace_cv)

        # check for blocked rays
        ok_pass_cv = rx_data['BLS'] == 0
        assert ok_pass.shape == ok_pass_cv.shape
        assert np.all(ok_pass == ok_pass_cv)

        jj = np.logical_and(rx_data['BLS'] == 0,
                            rx_data['RER'] == 0)
        if np.any(jj):
            p_cv, r_cv, opl_cv = (rx_data['ray_out'][jj, :3],
                                  rx_data['ray_out'][jj, 3:],
                                  rx_data['opl'][jj])

            p_macos, r_macos, opl_macos = (p_macos[:, jj].T,
                                           r_macos[:, jj].T,
                                           opl_macos[jj])

            if 1==1:
                # ray-intersection positions [x,y,z] at final surface
                np.testing.assert_allclose(p_cv, p_macos, TOL['P'][0], TOL['P'][1],
                                        err_msg="pos comparisons failed")

                # ray-directions [L,M,N] at 'final' surface
                #   => Code V: there is a flip in the ray direction after each reflection
                #              need to compare neg. dir. as well
                try:
                    np.testing.assert_allclose(r_cv, r_macos, TOL['r'][0], TOL['r'][1],
                                            err_msg="dir comparisons failed")
                except:
                    np.testing.assert_allclose(-r_cv, r_macos, TOL['r'][0], TOL['r'][1],
                                            err_msg="dir comparisons failed")

                # OPL from 1st (where rays were defined) to final surface
                np.testing.assert_allclose(opl_cv, opl_macos, TOL['L'][0], TOL['L'][1],
                                        err_msg="OPL comparisons failed")


            if 1==1:
                # pts, centroid, shift, csys = _lib.spot(srf=2,
                #                                 vpt_center=True,
                #                                 beam_csys=2,
                #                                 reset_trace=True)
                # print(_lib.elt_psi(2))

                import matplotlib
                import matplotlib.pyplot as plt

                # fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
                # plt.plot(pts[:, 0], pts[:, 1], '.')
                # plt.show()
                # dp = (p_macos-p_cv)[:, :2]
                dp = (p_macos)[:, :2]
                # dp = dp[np.abs(dp[:,] < 1e-6]
                fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
                plt.plot(dp[:, 0], dp[:, 1], '.')
                plt.show()
                print(p_cv.shape, p_macos.shape)


class TestMasks:

    masks_elliptical_parabola = chain.from_iterable([[
        (rx, 13, 1, ape_ellipse(a=9.0, b=9.0, xc=0, yc=0, srf=srf, tag=tag)),
        (rx, 13, 1, ape_ellipse(a=3.0, b=3.0, xc=0, yc=0, srf=srf, tag=tag)),
        *ape_sweep(rx, 13, 1, ape_ellipse, (3.0, 5.0), 5.0, 12, srf=srf, tag=tag),
        *ape_sweep(rx, 13, 1, ape_ellipse, (4.5, 2.5), 5.0, 12, srf=srf, tag=tag),
         ] for (rx, srf) in (('Rx_0000', 2),
                             ('Rx_0001', 3),
                             ('Rx_0001', 2),
                             ('Rx_0002', 2),
                             ('Rx_0002', 3),
                             )
           for tag in ('CLR', 'OBS', )])

    masks_rectangular_parabola = chain.from_iterable([[
        (rx, 13, 1, ape_rectangle(a=8.0, b=8.0, xc=+0, yc=+0, srf=srf, tag=tag)),
        (rx, 13, 1, ape_rectangle(a=4.0, b=4.0, xc=+0, yc=+0, srf=srf, tag=tag)),
        *ape_sweep(rx, 13, 1, ape_rectangle, (3.0, 5.0), 4.8, 12, srf=srf, tag=tag),
        *ape_sweep(rx, 13, 1, ape_rectangle, (4.5, 2.5), 4.8, 12, srf=srf, tag=tag),
         ] for (rx, srf) in (('Rx_0000', 2),
                             ('Rx_0001', 3),
                             ('Rx_0001', 2),
                             ('Rx_0002', 2),
                             ('Rx_0002', 3),
                             )
           for tag in ('CLR', 'OBS', )])

    masks_circular_parabola = chain.from_iterable([[
        (rx, 13, 1, ape_circle(r=9.0, xc=0, yc=0, srf=srf, tag=tag)),
        (rx, 13, 1, ape_circle(r=4.0, xc=0, yc=0, srf=srf, tag=tag)),
        *ape_sweep(rx, 13, 1, ape_circle, (3.0, ), 5.0, 12, srf=srf, tag=tag),
         ] for (rx, srf) in (('Rx_0000', 2),
                             ('Rx_0001', 3),
                             ('Rx_0001', 2),
                             ('Rx_0002', 2),
                             ('Rx_0002', 3),
                             )
           for tag in ('CLR', 'OBS')])

    masks_obs_rect_parabola = chain.from_iterable([[
        (rx, 31, 1, obs_rot_rectangular(rdy=10, a=1, b=3, xc=5, yc=0, phi=-10, srf=srf, tag='OBS')),
        *ape_sweep2(rx, 13, 1, obs_rot_rectangular, (10, 1.5, 3), 5.0, 12, phi=phi, srf=srf, tag='OBS'),
        ] for phi in np.arange(-6, 360, 36)
          for (rx, srf) in (('Rx_0000', 2),
                            ('Rx_0001', 3),
                            ('Rx_0001', 2),
                            ('Rx_0002', 2),
                            ('Rx_0002', 3),
                             )
        ])

    masks_obs_ellipse_parabola = chain.from_iterable([[
        (rx, 31, 1, obs_rot_ellipse(rdy=10, a=1.75, b=3, xc=5, yc=0, phi=-12, srf=srf, tag='OBS')),
        *ape_sweep2(rx, 13, 1, obs_rot_ellipse, (10, 1.75, 3), 5.0, 12, phi=phi, srf=srf, tag='OBS'),
        ] for phi in np.arange(-6, 360, 36)
          for (rx, srf) in (('Rx_0000', 2),
                            ('Rx_0001', 3),
                            ('Rx_0001', 2),
                            ('Rx_0002', 2),
                            ('Rx_0002', 3),
                             )
        ])

    masks_obs_circular_parabola = chain.from_iterable([[
        (rx, 21, 1, obs_circle(rdy=10, r=9.0, xc=0, yc=0, srf=srf, tag='OBS')),
        (rx, 13, 1, obs_circle(rdy=10, r=4.0, xc=0, yc=0, srf=srf, tag='OBS')),
        *ape_sweep(rx, 13, 1, obs_circle, (10, 3.0, ), 5.0, 12, srf=srf, tag='OBS'),
         ] for (rx, srf) in (('Rx_0000', 2),
                             ('Rx_0001', 3),
                             ('Rx_0001', 2),
                             ('Rx_0002', 2),
                             ('Rx_0002', 3),
                             )])


    rx_set = (*masks_circular_parabola,
              *masks_rectangular_parabola,
              *masks_elliptical_parabola,
              *masks_obs_rect_parabola,
              *masks_obs_ellipse_parabola,
              *masks_obs_circular_parabola,
              )

    @pytest.mark.parametrize("rx, n_sampling, i_wl, cmd", rx_set)
    def test_demo7(self, cv_setup, session_dir, rx, n_sampling, i_wl, cmd):
        """ray-trace comparisons"""

        i_fov      = 0  # 0: for codev trace => collect for all FoV positions
        i_zoo      = 1

        # -------------------------------------------------------
        # CODE V: trace rays
        # -------------------------------------------------------
        cv_setup.cmd(f"in {rx}")
        # cv_setup.cmd("thi si 5")   # for testing
        if len(cmd) > 0:
            cv_setup.cmd(cmd)
        rx_data = cv_setup.test_ray_trace("", n_sampling, i_wl, i_fov, i_zoo)


        n_rays = rx_data['n_rays']
        n_pts = np.ceil(np.sqrt(n_rays)) + 1

        model_size = max(128,
                         2**np.int32(np.ceil(np.log10(n_pts)/np.log10(2))))
        assert model_size < 2048

        # -------------------------------------------------------
        # CODE V: Rx conversion
        # -------------------------------------------------------
        fid_macos = session_dir / f'{rx}.in'
        cv_setup.cv2macos(fid_macos,
                          i_origin=rx_data['origin'],
                          i_wl=i_wl,
                          i_fov=1,
                          i_zoo=i_zoo)

        # -------------------------------------------------------
        # MACOS: trace rays
        # -------------------------------------------------------
        # update model size if needed
        if _lib.model_size() < model_size:
            _lib.init(model_size)
            assert _lib.model_size() == model_size

        # fid_macos = 'Rx_0000c_Tmp.in'   # debug
        _lib.load(fid_macos)

        # trace rays to Src. Ref. Srf.  &
        __ = _lib.traceWavefront(rx_data['entry_port'])[0]

        # upload input ray definitions from ref. code
        # ==> ensures the same ray input definition
        ray_in = rx_data['ray_in']
        _lib.setRayInfo(ray_in[:, :3].T, ray_in[:, 3:].T,
                        np.zeros(n_rays), np.ones(n_rays))

        # trace to 'final' Srf.
        __, n_rays = _lib.traceWavefront(rx_data['exit_port'])[:2]
        p_macos, r_macos, opl_macos, ok_trace, ok_pass = _lib.getRayInfo(n_rays)

        # -------------------------------------------------------
        # data comparisons
        # -------------------------------------------------------
        # check for trace failures
        ok_trace_cv = rx_data['RER'] == 0
        assert ok_trace.shape == ok_trace_cv.shape
        assert np.all(ok_trace == ok_trace_cv)

        # check for blocked rays
        ok_pass_cv = rx_data['BLS'] == 0
        assert ok_pass.shape == ok_pass_cv.shape
        # print("==========", ok_pass.shape, np.sum(ok_pass), np.sum(ok_pass_cv))
        # NOTE: not always in agreement with CODE V
        # assert np.all(ok_pass == ok_pass_cv)

        jj = np.logical_and(rx_data['BLS'] == 0,
                            rx_data['RER'] == 0)

        if np.any(jj):
            p_cv, r_cv, opl_cv = (rx_data['ray_out'][jj, :3],
                                  rx_data['ray_out'][jj, 3:],
                                  rx_data['opl'][jj])

            p_macos, r_macos, opl_macos = (p_macos[:, jj].T,
                                           r_macos[:, jj].T,
                                           opl_macos[jj])

            if 1==1:
                # ray-intersection positions [x,y,z] at final surface
                np.testing.assert_allclose(p_cv, p_macos, TOL['P'][0], TOL['P'][1],
                                           err_msg="pos comparisons failed")

                # ray-directions [L,M,N] at 'final' surface
                #   => Code V: there is a flip in the ray direction after each
                #              reflection need to compare neg. dir. as well
                try:
                    np.testing.assert_allclose(r_cv, r_macos, TOL['r'][0], TOL['r'][1],
                                               err_msg="dir comparisons failed")
                except:
                    np.testing.assert_allclose(-r_cv, r_macos, TOL['r'][0], TOL['r'][1],
                                            err_msg="dir comparisons failed")

                # OPL from 1st (where rays were defined) to final surface
                np.testing.assert_allclose(opl_cv, opl_macos, TOL['L'][0], TOL['L'][1],
                                        err_msg="OPL comparisons failed")


            if 1==0:
                # pts, centroid, shift, csys = _lib.spot(srf=2,
                #                                 vpt_center=True,
                #                                 beam_csys=2,
                #                                 reset_trace=True)
                # print(_lib.elt_psi(2))

                import matplotlib
                import matplotlib.pyplot as plt

                # fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
                # plt.plot(pts[:, 0], pts[:, 1], '.')
                # plt.show()
                # dp = (p_macos-p_cv)[:, :2]
                dp = (p_macos)[:, :2]
                # dp = dp[np.abs(dp[:,] < 1e-6]
                fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
                plt.plot(dp[:, 0], dp[:, 1], '.', ms=2)
                plt.show()
                print(p_cv.shape, p_macos.shape)

        print(session_dir)


class TestSrc:

    # ------------------------------------- Point Source Definition

    rx_set = chain.from_iterable([
        # ------------ prolate Ellipse
        [('Rx_0010', 13, 1, f"ade s3 {ade};cde s3 {cde}")
              for ade in np.linspace(-30, 30, num=9, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        # ------------ prolate Ellipse, closer to sphere
        [('Rx_0011', 13, 1, f"ade s3 {ade};cde s3 {cde}")
              for ade in np.linspace(-60, 60, num=9, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        # ------------ parabola => parabola
        [('Rx_0012', 13, 1, f"ade s3 {ade};cde s3 {cde}")
              for ade in np.linspace(-60, 60, num=9, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        # ------------ Sphere (create virtual source) => Hyperbola
        [('Rx_0013', 13, 1, f"ade s3 {ade};cde s3 {cde}")
              for ade in np.linspace(-80, 80, num=11, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        ])

    @pytest.mark.parametrize("rx, n_sampling, i_wl, cmd", rx_set)
    def test_point_src(self, macos_setup, cv_setup, session_dir,
                             rx, n_sampling, i_wl, cmd):
        """ray-trace -- Testing at conjugate points, perfect imaging"""

        i_fov = 1  # 0: for codev trace => collect for all FoV positions
        i_zoo = 1

        # -------------------------------------------------------
        # CODE V: Rx conversion
        # -------------------------------------------------------
        cv_setup.cmd(f"in {rx}")
        if len(cmd) > 0:
            cv_setup.cmd(cmd)

        fid_macos = session_dir / f'{rx}.in'
        i_origin = int(cv_setup.eval(f"(slb s'origin')"))
        # i_stop = int(cv_setup.eval(f"(slb s'stop')"))

        cv_setup.cv2macos(fid_macos,
                          i_origin=i_origin,
                          i_wl=i_wl,
                          i_fov=i_fov,
                          i_zoo=i_zoo)

        # -------------------------------------------------------
        # MACOS: trace rays
        # -------------------------------------------------------
        # model_size = 128
        # _lib.init(model_size)
        _lib.load(fid_macos)
        _lib.src_sampling(n_sampling)   # reduce ray sampling
        # _lib.stop(i_stop)
        # _lib.fex()

        img = _lib.num_elt()-2  # XP may not be correctly set up;
                                # hence, trace to return img srf

        pts = _lib.spot(img, vpt_center=True, beam_csys=3, reset_trace=True)[0]
        rad = np.linalg.norm(pts, axis=1)
        pts_ref = np.zeros_like(pts[:, 0], dtype=float)

        # -------------------------------------------------------
        # data comparisons: perfect conjugate point-point imaging
        # -------------------------------------------------------
        tol = TOL['P']

        np.testing.assert_allclose(pts[:, 0], pts_ref, *tol,
                                   err_msg="pos comparisons failed")

        np.testing.assert_allclose(pts[:, 1], pts_ref, *tol,
                                   err_msg="pos comparisons failed")

        assert np.all(rad < tol[0])

        print(session_dir)


class TestXP:

    # ------------------------------------- Exit Pupil & Surfaces
    #
    # Note: ensure that XP location is in front of Image Plane
    #       => modifying the XP location (zde ss ...)

    rx_set = chain.from_iterable([
        [('Keck', 21, 1, "", 1e-3, 1),
         ('Rx_0014_TMA', 21, 1, "", 0.02, 1),
         ('Rx_0015_Yolo_5M', 21, 1, "", 0.003, 1),
         ('Rx_0015_Yolo_5M', 21, 1, "", 0.011, 2),
         ('Rx_0015_Yolo_5M', 21, 1, "", 0.010, 3),         # Anamorphic (no Kc_x, Kc_y)
         ('Rx_0015_Yolo_3M', 21, 1, "", 0.017, 1),             # Anamorphic (no Kc_x, Kc_y)
         ('Rx_0016_Pressmann_Camichel', 21, 1, "", 0.035, 1),  # Conical
         ],
        # ------------ prolate Ellipse
        [('Rx_0010', 13, 1, f"zde ss 800;nao 0.05;ade s3 {ade};cde s3 {cde}", 1e-10, 1)
              for ade in np.linspace(-30, 30, num=9, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        # ------------ prolate Ellipse, closer to sphere
        [('Rx_0011', 13, 1, f"zde ss 500;nao 0.05;ade s3 {ade};cde s3 {cde}", 1e-10,1)
              for ade in np.linspace(-60, 60, num=9, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        # ------------ parabola => parabola
        [('Rx_0012', 13, 1, f"ade s3 {ade};cde s3 {cde}", 1e-10,1)
              for ade in np.linspace(-60, 60, num=9, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        # ------------ Sphere (create virtual source) => Hyperbola
        [('Rx_0013', 13, 1, f"ade s3 {ade};cde s3 {cde}", 1e-10,1)
              for ade in np.linspace(-80, 80, num=9, endpoint=True)
              for cde in np.linspace(0, 360, num=11, endpoint=True)],
        ])

    @pytest.mark.parametrize("rx, n_sampling, i_wl, cmd, tol, fov", rx_set)
    def test_exit_pupil(self, macos_setup, cv_setup, session_dir,
                             rx, n_sampling, i_wl, cmd, tol, fov):
        """ray-trace -- Testing at conjugate points, perfect imaging"""

        i_fov = fov
        i_zoo = 1

        # -------------------------------------------------------
        # CODE V: Rx conversion
        # -------------------------------------------------------
        rx_ = Path(".") / 'Rx' / rx
        cv_setup.cmd(f"in {rx_}")
        if len(cmd) > 0:
            cv_setup.cmd(cmd)

        fid_macos = session_dir / f'{rx}.in'
        i_origin = int(cv_setup.eval(f"(slb s'origin')"))
        # i_stop = int(cv_setup.eval(f"(slb ss)"))

        cv_setup.cv2macos(fid_macos,
                          i_origin=i_origin,
                          i_wl=i_wl,
                          i_fov=i_fov,
                          i_zoo=i_zoo)

        # -------------------------------------------------------
        # MACOS: trace rays
        # -------------------------------------------------------
        _lib.load(fid_macos)
        _lib.src_sampling(n_sampling)   # reduce ray sampling
        # _lib.stop(i_stop)
        # _lib.fex()

        img = _lib.num_elt()

        pts = _lib.spot(img, vpt_center=True, beam_csys=3, reset_trace=True)[0]
        cxy = np.sum(pts, axis=0)/pts.shape[0]

        pts -= cxy
        rad = np.linalg.norm(pts, axis=1).max()

        # -------------------------------------------------------
        # data comparisons: perfect conjugate point-point imaging
        # -------------------------------------------------------
        assert np.abs(pts[:, 0]).max() < tol
        assert np.abs(pts[:, 1]).max() < tol
        assert np.abs(rad) < tol

        print(session_dir)
