from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Source:
    ChfRayDir: tuple[float, ...]   # 0d0  0d0  1d0
    ChfRayPos: tuple[float, ...]   # 0d0  0d0 -1.705433249610300d+004
    zSource:   float               #  1d0
    IndRef:    float               #  1d0
    Extinc:    float               #  0d0
    Wavelen:   float               #  6.870000000000000d-004
    Flux:      float               #  1d0
    nGridpts:  int                 #  1024
    xGrid:     tuple[float, ...]   #  1d0  0d0  0d0
    yGrid:     tuple[float, ...]   #  0d0  1d0  0d0
    Aperture:  float               #  3.125000000000000d-001    % maximise usage
    Obscratn:  float               #  0d0
    StopElt:   int                 #  1

    StopOffset:tuple[float, float] = (0e0, 0e0)
    BaseUnits: str = 'mm'
    WaveUnits: str = 'mm'
    GridType:  str = 'Circular'


# https://www.codesdope.com/course/python-subclass-of-a-class/
class RxDataBase:
    """pytest manual: 2.3.12 Modularity: using fixtures from a fixture function"""

    def __init__(self, data):
        self.rx_settings = data
        self.tol_abs = 1e-15
        self.tol_rel = 1e-15
        self.n_elt = data["nElt"]
        self.rx = data["Rx"]


def Rx_tuple(name):
    return namedtuple(name, 'Rx, nElt, KcElt, KrElt')

def _TElt_eye(nElt):
    TElt = np.zeros((6,6,nElt))
    for i in range(nElt):
        TElt[:,:,i] = np.eye(6,6)
    return TElt

# eltSrfTypeName:  [str]:  01) Flat      05) Interpolated  09) GridData
#                          02) Conic     06) Anamorphic    10) Toric
#                          03) Aspheric  07) UserDefined   11) AsGrData
#                          04) Monomial  08) Zernike

rx_trace_path = Path('..') / 'test_ray_trace_comps'


# -------------------------------------------------------------------------------------------
# [rx_trace_planes_001.in] Single Flat Parallel Window
# -------------------------------------------------------------------------------------------
def rx_trace_planes_001():

    nElt = 7

    # -----------------------------------------------
    # EltSrfCFrame: xMon, yMon, zMon, pMon
    #
    active = np.zeros(nElt)
    pMon = np.zeros((3, nElt))
    xMon = np.zeros((3, nElt))
    yMon = np.zeros((3, nElt))
    zMon = np.zeros((3, nElt))

    # -----------------------------------------------
    # TElt  ==> local coordinate frame on surfaces
    #
    TElt        = _TElt_eye(nElt)      # no Local Coord. Frame (LCF) was defined  => GCF
    TElt_status = np.zeros((1,nElt))   # default for GCF -- no LCF defined
    TElt_update = np.zeros((1,nElt))   # default for GCF -- GCF does not alter with perturbation

    TElt_ = (TElt, np.int32(TElt_status), np.int32(TElt_update))


    data = {'Rx': 'rx_trace_planes_001.in',

            # --------------------------------
            'Src': Source(ChfRayDir= (0e0, 0e0, 1e0),
                          ChfRayPos= (0e0, 0e0, -89.),
                          zSource=   -1e0,
                          IndRef=    1e0,
                          Extinc=    0e0,
                          Wavelen=   587.5618e-006,
                          Flux=      1e0,
                          nGridpts=  1024,
                          xGrid=     (1e0, 0e0, 0e0),
                          yGrid=     (0e0, 1e0, 0e0),
                          GridType=  'Circular',
                          Aperture=  0.25e0,
                          Obscratn=  0e0,
                          StopElt=   -1,     # exclude from Stop Loc. determination (crash)
                          BaseUnits= 'mm',
                          WaveUnits= 'mm'),
            'nElt': nElt,
            # --------------------------------
            'Grp': ((1, range(1, 8)),
                    (3, (3, 4)),
                    (7, (5, 7))),
            # --------------------------------
            'KrElt': np.array([1e22,]*nElt, dtype=np.float64),
            # --------------------------------
            'KcElt': np.array([0e0,]*nElt, dtype=np.float64),
            # --------------------------------
            # eltSrfTypeName:  [str]:  01) Flat      05) Interpolated  09) GridData
            #                          02) Conic     06) Anamorphic    10) Toric
            #                          03) Aspheric  07) UserDefined   11) AsGrData
            #                          04) Monomial  08) Zernike
            'EltSrfType': np.asarray([1,]*nElt, dtype=np.int32),
            # --------------------------------
            'VptElt': np.asarray([[0e0, 0e0, -10e0],
                                  [0e0, 0e0,   0e0],
                                  [0e0, 0e0,  30e0],
                                  [0e0, 0e0,  50e0],
                                  [0e0, 0e0, 100e0],
                                  [0e0, 0e0,  50e0],
                                  [0e0, 0e0, 100e0]]).T,
            # --------------------------------
            'RptElt': np.asarray([[0e0, 0e0, -10e0],
                                  [0e0, 0e0,   0e0],
                                  [0e0, 0e0,  30e0],
                                  [0e0, 0e0,  50e0],
                                  [0e0, 0e0, 100e0],
                                  [0e0, 0e0,  50e0],
                                  [0e0, 0e0, 100e0]]).T,
            # --------------------------------
            'PsiElt': np.tile(np.array([[0e0, 0e0, 1e0]]), (nElt, 1)).T,
            # --------------------------------
            'TElt': TElt_,
            # --------------------------------
            'EltSrfCFrame': (active, pMon, xMon, yMon, zMon)
            }

    return data


# -------------------------------------------------------------------------------------------
# Rx_003
# -------------------------------------------------------------------------------------------

def data_Rx_003():

    nElt = 8

    # ----------------------
    # TElt                    ==> local coordinate frame on surfaces
    # ----------------------
    TElt        = _TElt_eye(nElt)      # no Local Coord. Frame (LCF) was defined  => GCF
    TElt_status = np.zeros((1,nElt))   # default for GCF -- no LCF defined
    TElt_update = np.zeros((1,nElt))   # default for GCF -- GCF does not alter with perturbation

    TElt_ = (TElt, TElt_status, TElt_update)


    # ----------------------
    # EltSrfCFrame: xMon, yMon, zMon, pMon
    # ----------------------
    active = np.zeros(nElt)
    pMon = np.zeros((3, nElt))
    xMon = np.zeros((3, nElt))
    yMon = np.zeros((3, nElt))
    zMon = np.zeros((3, nElt))

    iElt = np.array([2,3])
    pMon[:,iElt-1] = np.array([[0e0, 0e0, 0e0], [0e0, 0e0, 5e0]]).T
    xMon[:,iElt-1] = np.array([[1e0, 0e0, 0e0], [1e0, 0e0, 0e0]]).T
    yMon[:,iElt-1] = np.array([[0e0, 1e0, 0e0], [0e0, 1e0, 0e0]]).T
    zMon[:,iElt-1] = np.array([[0e0, 0e0, 1e0], [0e0, 0e0, 1e0]]).T
    active[iElt-1] = 1

    EltSrfCFrame = (active, pMon, xMon, yMon, zMon)

    # -----------------------------------------------
    data = {'Rx': 'Rx_003.in',
            # --------------------------------
            'Src': Source(ChfRayDir= (0e0, 0e0, +1e0),
                          ChfRayPos= (0e0, 0e0, -6e0),
                          zSource=   -1e22,
                          IndRef=    1e0,
                          Extinc=    0e0,
                          Wavelen=   550e0,
                          Flux=      1e0,
                          nGridpts=  1024,
                          xGrid=     (1e0, 0e0, 0e0),
                          yGrid=     (0e0, 1e0, 0e0),
                          GridType=  'Circular',
                          Aperture=  40e0,
                          Obscratn=  0e0,
                          StopElt=   2,
                          BaseUnits= 'mm',
                          WaveUnits= 'nm'),
            'nElt': nElt,
            # --------------------------------
            'Grp': ((2, (2, 3)),
                    (4, (4, 5)),
                    (8, (6, 8)), ),
            # --------------------------------
            'KcElt': np.array([ *[0e0,]*4, -2.305910241705553e+000, 0e0, 0e0, 0e0], dtype=np.float64),
            # --------------------------------
            'KrElt': np.array([ *[1e+022,]*4, -5e+001, 1e+022, -1.034958992644120e+002, 1e+022], dtype=np.float64),
            # --------------------------------
            # SrfTypeName:  [str]:  01) Flat      05) Interpolated  09) GridData
            #                       02) Conic     06) Anamorphic    10) Toric
            #                       03) Aspheric  07) UserDefined   11) AsGrData
            #                       04) Monomial  08) Zernike
            'EltSrfType': np.asarray((2, 9, 9, 2, 2, 1, 2, 1), dtype=np.int32),
            # --------------------------------
            'VptElt': np.asarray([[0e0, 0e0, -5e0                   ],
                                 [0e0, 0e0,  0e0                   ],
                                 [0e0, 0e0,  5e0                   ],
                                 [0e0, 0e0,  1e+001                ],
                                 [0e0, 0e0,  2e+001                ],
                                 [0e0, 0e0,  1.164278519004112e+002],
                                 [0e0, 0e0,  1.293195263599920e+001],
                                 [0e0, 0e0,  1.164278519004112e+002]]).T,
            # --------------------------------
            'RptElt': np.asarray([[0e0, 0e0, -5e0],
                                 [0e0, 0e0,  0e0],
                                 [0e0, 0e0,  5e0],
                                 [0e0, 0e0,  1e+001],
                                 [0e0, 0e0,  2e+001],
                                 [0e0, 0e0,  1.164278519004112e+002],
                                 [0e0, 0e0,  1.293195263599920e+001],
                                 [0e0, 0e0,  1.164278519004112e+002]]).T,
            # --------------------------------
            'PsiElt': np.asarray([[0e0, 0e0,  1e0],
                                  [0e0, 0e0, -1e0],
                                  [0e0, 0e0, -1e0],
                                  [0e0, 0e0, -1e0],
                                  [0e0, 0e0, -1e0],
                                  [0e0, 0e0,  1e0],
                                  [0e0, 0e0,  1e0],
                                  [0e0, 0e0,  1e0]]).T,
            # --------------------------------
            'TElt': TElt_,
            # --------------------------------
            'EltSrfCFrame': EltSrfCFrame,
            # --------------------------------
            'GridInfo': ((2, 0.4, 101), # (Srf., dx, npts)
                         (3, 0.4, 101)),
           }

    return data


def data_Rx_003_2():
    data = Rx_tuple('Rx_003')

    return data(Rx='Rx_003.in',\
                nElt = 8,\
                KcElt = np.array([[ *[0e0]*4, -2.305910241705553e+000, 0e0, 0e0, 0e0]], dtype=np.float64),\
                KrElt = np.array([[ *[1e+022]*4, -5e+001, 1e+022, -1.034958992644120e+002, 1e+022]], dtype=np.float64))


# -------------------------------------------------------------------------------------------
# 4-Mirror Freeform (on-axis Zernike: Born&Wolf)
# -------------------------------------------------------------------------------------------
def freeform_4mirrors():

    nElt = 10
    data = {'Rx': 'Rx_4Mirror_Freeform_Zrn_B&W.in',
            # --------------------------------
            'Src': Source(ChfRayDir= (-1.7449749160682679e-02, -1.7449749160682679e-02, 9.9969545988188746e-01),
                          ChfRayPos= ( 3.4904814088900267e+00,  3.4904814088900267e+00,-1.9996954598818874e+02),
                          zSource=   -1e22,
                          IndRef=    1e0,
                          Extinc=    0e0,
                          Wavelen=   1e3,
                          Flux=      1e0,
                          nGridpts=  65,
                          GridType=  'Circular',
                          xGrid=     (1e0, 0e0, 0e0),
                          yGrid=     (0e0, 1e0, 0e0),
                          Aperture=  1e2,
                          Obscratn=  0e0,
                          StopElt=   2,
                          BaseUnits= 'mm',
                          WaveUnits= 'nm'),
            'nElt': nElt,
            # --------------------------------
            'KcElt': np.array([0e0, 0e0, -7.7378189475495351e-01, 4.0478550315165016e+00,
                               0e0, 4.4006894195499999e-03, *[0e0, ]*4], dtype=np.float64),
            # --------------------------------
            'KrElt': np.array([-1e22, -1e22, -3.9753769113617301e+02, -1.6765902648925811e+02,
                               -1.5684709672072661e+05, -2.5880088420651100e+02, -1e22, -1e22,
                               -6.6000000000000000e+01, -1e22], dtype=np.float64),
            # --------------------------------
           }

    # :return eltSrfTypeName:  [str]:  01) Flat      05) Interpolated  09) GridData
    #                                  02) Conic     06) Anamorphic    10) Toric
    #                                  03) Aspheric  07) UserDefined   11) AsGrData
    #                                  04) Monomial  08) Zernike
    data['EltSrfType'] = np.asarray((2, 2, 8, 8, 8, 8, 2, 2, 2, 2), dtype=np.int32)

    data['VptElt'] = np.asarray([[0e0,  0.0000000000000000e+00, -1.0000000000000000e+02],
                                 [0e0,  0.0000000000000000e+00,  0.0000000000000000e+00],
                                 [0e0, -1.2377662295585850e+02,  1.1567555387677251e+01],
                                 [0e0, -1.0749591357599994e+02, -1.5543718776952548e+02],
                                 [0e0, -1.6356907622618735e+02, -9.6724395810660170e+01],
                                 [0e0,  8.8382618845860236e+01, -6.2115341288029200e+01],
                                 [0e0,  8.2436817822147532e+01, -8.1034851354576972e+01],
                                 [0e0, -1.8539777091163518e+02, -8.1034823693092491e+01],
                                 [0e0, -1.1939777091163553e+02, -8.1034830509456214e+01],
                                 [0e0, -1.8539777091163518e+02, -8.1034823693092491e+01]]).T

    data['RptElt'] = data['VptElt']

    # -------------------------
    data['PsiElt'] = None
    tmp = np.asarray([[0e0,  0.0000000000000000e+00,  1.0000000000000000e+00],
                      [0e0,  0.0000000000000000e+00,  1.0000000000000000e+00],
                      [0e0,  6.0501562085986005e-02, -9.9816810256847799e-01],
                      [0e0, -8.2345632590321155e-02, -9.9660383141612474e-01],
                      [0e0, -6.1581498944968283e-01,  7.8789079114372595e-01],
                      [0e0, -9.9857897215967562e-01,  5.3291991523356064e-02],
                      [0e0, -9.9999999999999456e-01,  1.0327823828093496e-07],
                      [0e0, -9.9999999999999456e-01,  1.0327823828093496e-07],
                      [0e0, -9.9999999999999456e-01,  1.0327823828093496e-07],
                      [0e0, -9.9999999999999456e-01,  1.0327823828093496e-07]]).T

    # normalise
    for iElt in np.arange(data['nElt']):
        tmp[:,iElt] = tmp[:,iElt]/np.sqrt(np.sum(tmp[:,iElt]**2))
    data['PsiElt'] = tmp

    # ---------------------------
    TElt        = _TElt_eye(nElt)      # no Local Coord. Frame (LCF) was defined  => GCF
    TElt_status = np.zeros((1,nElt))   # default for GCF -- no LCF defined
    TElt_update = np.zeros((1,nElt))   # default for GCF -- GCF does not alter with perturbation

    data['TElt'] = (TElt, TElt_status, TElt_update)

    # ----------------------
    # EltSrfCFrame: xMon, yMon, zMon, pMon

    active = np.zeros(nElt)
    pMon = np.zeros((3, nElt))
    xMon = np.zeros((3, nElt))
    yMon = np.zeros((3, nElt))
    zMon = np.zeros((3, nElt))

    iElt = np.array([3, 4, 5, 6], dtype=np.int32)
    active[iElt-1] = 1

    pMon[:,iElt-1] = np.asarray([
        [0e0, -1.2377662295585850e+02,  1.1567555387677251e+01],
        [0e0, -1.0749591357599994e+02, -1.5543718776952548e+02],
        [0e0, -1.6356907622618735e+02, -9.6724395810660170e+01],
        [0e0,  8.8382618845860236e+01, -6.2115341288029200e+01]]).T


    xMon[:,iElt-1] = np.asarray([[ 1e0, 0e0, 0e0],
                                 [ 1e0, 0e0, 0e0],
                                 [ 1e0, 0e0, 0e0],
                                 [ 1e0, 0e0, 0e0]]).T

    yMon[:,iElt-1] = np.asarray([
              [0e0, 9.9816810256847799e-01,  6.0501562085986005e-02],
              [0e0, 9.9660383141612474e-01, -8.2345632590321155e-02],
              [0e0, 7.8789079114372595e-01,  6.1581498944968283e-01],
              [0e0, 5.3291991523356064e-02,  9.9857897215967562e-01]]).T

    zMon[:,iElt-1] = np.asarray([
              [0e0, -6.0501562085986005e-02, 9.9816810256847799e-01],
              [0e0,  8.2345632590321155e-02, 9.9660383141612474e-01],
              [0e0, -6.1581498944968283e-01, 7.8789079114372595e-01],
              [0e0, -9.9857897215967562e-01, 5.3291991523356064e-02]]).T

    data['EltSrfCFrame'] = (active, pMon, xMon, yMon, zMon)

          # (/'ANSI',             ! 1
          #   'BornWolf',         ! 2
          #   'Fringe',           ! 3
          #   'NormANSI',         ! 4
          #   'NormBornWolf',     ! 5
          #   'NormFringe',       ! 6
          #   'NormHex',          ! 7
          #   'NormNoll',         ! 8
          #   'NormAnnularNoll'/) ! 9
          #   'Fringe'             10
          #   'Ext. Fringe'        11

    s3_coef= [ 1.8808716236115210e-03,  0e0,  0e0, -8.9930760031364507e-02,  0e0,  0e0, 0e0,  0e0,  1.4127443220608779e-02,
              -2.1495606221683089e-02, -1.0932165940550929e-02,  3.1389721102123859e-03,-1.6470186915986469e-03,  0e0,  0e0,
               0e0,  0e0,  0e0, -1.5706296199668101e-04, -1.0438219839649929e-03,  1.2805548290162169e-03, -1.0469301653118880e-05,
              -1.1779921320702789e-04, -3.0635277943000274e-05, 2.2846077094476931e-04,  0e0,  0e0,  0e0,  0e0,  0e0, 0e0,
               0e0,  0e0,  0e0,  0e0,  0e0, 0e0,  0e0,  0e0,  0e0, -5.3921474066666317e-06,  0e0, 0e0,  0e0,  0e0]

    s4_coef = [-5.0556020431600693e-02,  0e0,  0e0, -1.2410187346941959e-02,  0e0,  0e0, 0e0,  0e0,  2.6323879480166769e-02,
               -3.4506010539637533e-02, -7.2049647458806102e-03,  2.0452088797563470e-02, 4.9927322272141410e-02,  0e0,  0e0,
               0e0,  0e0,  0e0, 8.3469947518146140e-03, -1.6075980862470200e-03,  7.0304251531787753e-04,  3.5724244618149428e-04,
               3.7609680318358499e-04,  8.4695969896432608e-04, -5.1527225709601309e-04,  0e0,  0e0,  0e0,  0e0,  0e0,
                0e0,  0e0,  0e0,  0e0,  0e0,  0e0, 0e0,  0e0,  0e0,  0e0,  1.1342585369968070e-04,  0e0, 0e0,  0e0,  0e0]

    s5_coef = [-9.7741737538668896e-03,  0e0,  0e0,  4.9803111556519081e-01,  0e0,  0e0, 0e0,  0e0,  1.2267193798764610e-01,
               -5.5212161814936134e-01, -9.7800310090672082e-02, -7.0204343726668311e-02,  1.0296015298780230e-02,  0e0,  0e0,
                0e0,  0e0,  0e0, -1.5610226350813590e-02,  1.2546945845651361e-01,  2.4924468840703380e-02,  1.3397963269362489e-02,
                3.0538136222757330e-02, -8.6572676812586569e-03, 4.2542768387393229e-04,  0e0,  0e0,  0e0,  0e0,  0e0,
              0e0,  0e0,  0e0,  0e0,  0e0,  0e0, 0e0,  0e0,  0e0,  0e0, -9.6413851046729684e-05,  0e0, 0e0,  0e0,  0e0]

    s6_coef = [ 1.2634448926564819e-02,  0e0,  0e0, -4.3846415176309918e-02,  0e0,  0e0, 0e0,  0e0,  7.0300968483755968e-03,
               -5.2796271694644167e-03, -1.1471816142188389e-03, -1.5283197257486810e-03, -1.3087556711651200e-02,  0e0,  0e0,
                0e0,  0e0,  0e0, 1.9022582814053041e-04, -2.1306297906181351e-05, -1.8729200002406840e-04, -2.6164664595012861e-05,
               -4.0458757223932833e-05, -3.2342768272353053e-05, -4.7129011670166170e-04,  0e0,  0e0,  0e0,  0e0,  0e0,
                0e0,  0e0,  0e0,  0e0,  0e0,  0e0, 0e0,  0e0,  0e0,  0e0, -1.8182341966327051e-05,  0e0, 0e0,  0e0,  0e0]

    data['Zern'] = ({"Srf":3, "ZrnType":2, "ZrnMode":range(1, 45+1), "ZrnCoef":np.array(s3_coef), "NormRad": 1.7139932510674731e+02},
                    {"Srf":4, "ZrnType":2, "ZrnMode":range(1, 45+1), "ZrnCoef":np.array(s4_coef), "NormRad": 3.3621890944880562e+01},
                    {"Srf":5, "ZrnType":2, "ZrnMode":range(1, 45+1), "ZrnCoef":np.array(s5_coef), "NormRad": 1.2442990944567720e+02},
                    {"Srf":6, "ZrnType":2, "ZrnMode":range(1, 45+1), "ZrnCoef":np.array(s6_coef), "NormRad": 9.2218293140642643e+01},
                    )  # (Srf, ZernType, ZernMode, Coefs, lMon)

    return data


# -------------------------------------------------------------------------------------------
# Rx_test.in <= jwst_Rx_OO.in
# -------------------------------------------------------------------------------------------
def data_Rx_test():

    data = data_jwst_Rx_OO()
    data['Rx'] = 'Rx_test.in'

    # Note: - tracesub.F: in CTRACE(...)
    #              ZernTypeL = {1,4} => ZernToMon1    Norm/ Malacara
    #                        = {2,5} => ZernToMon2    Norm/ Born & Wolf   => calls ZernToMon1
    #                        = {3,6} => ZernToMon3    Norm/ Fringe        => calls ZernToMon1
    #                        = {  7} => ZernToMon4    NormHex
    #                        = {  8} => ZernToMon6    NormNoll            => calls ZernToMon1
    #                        = {  9} => ZernToMon7    NormAnnularNoll

    # SM: Srf = 25, NormNoll
    data['Zern'] = ({"Srf":25, "ZrnType":8, "ZrnMode":range(1, 45+1), "ZrnCoef":np.arange(1, 45+1, dtype=float), "NormRad":369.5}, )  # Srf, ZernType, ZernMode, Coefs, lMon

    return data


# -------------------------------------------------------------------------------------------
# Microscope Objective (grp prb)
# -------------------------------------------------------------------------------------------
def Rx_prb_grp():

    nElt = 25
    data = {'Rx': 'Rx_Conic_prb_grp.in',
            # --------------------------------
            'Src': Source(ChfRayDir= (0e0, 0e0,    1e0),
                          ChfRayPos= (0e0, 0e0, -110e0),
                          zSource=   -1e22,
                          IndRef=    1e0,
                          Extinc=    0e0,
                          Wavelen=   6.5e-04,
                          Flux=      1e0,
                          nGridpts=  256,
                          GridType=  'Circular',
                          xGrid=     (1e0, 0e0, 0e0),
                          yGrid=     (0e0, 1e0, 0e0),
                          Aperture=  2.1305e+02,
                          Obscratn=  0e0,
                          StopElt=   15,
                          BaseUnits= 'mm',
                          WaveUnits= 'mm'),
            'nElt': nElt,
            # --------------------------------
            'KcElt': None,
            # --------------------------------
            'KrElt': None,
            # --------------------------------
           }

    # :return eltSrfTypeName:  [str]:  01) Flat      05) Interpolated  09) GridData
    #                                  02) Conic     06) Anamorphic    10) Toric
    #                                  03) Aspheric  07) UserDefined   11) AsGrData
    #                                  04) Monomial  08) Zernike
    data['EltSrfType'] = np.ones(nElt)*2
    data['EltSrfType'][[22,24]] = 1  # img

    data['VptElt'] = None

    data['RptElt'] = data['VptElt']

    # -------------------------
    data['PsiElt'] = None

    # # normalise
    # for iElt in np.arange(data['nElt']):
    #     tmp[:,iElt] = tmp[:,iElt]/np.sqrt(np.sum(tmp[:,iElt]**2))
    # data['PsiElt'] = tmp

    # ---------------------------
    # TElt        = _TElt_eye(nElt)      # no Local Coord. Frame (LCF) was defined  => GCF
    # TElt_status = np.zeros((1,nElt))   # default for GCF -- no LCF defined
    # TElt_update = np.zeros((1,nElt))   # default for GCF -- GCF does not alter with perturbation

    # data['TElt'] = (TElt, TElt_status, TElt_update)

    # ----------------------
    # EltSrfCFrame: xMon, yMon, zMon, pMon
    #              (Srf, (Srf in Grp)), ...
    data['Grp'] = ((2, (3,4)), (3, (3,4)), (4, (3,4)), (5, (5,6)),
                   (6, (5,6)), (7, (7,8)), (8, (7,8)), (9, (9,10)),
                   (10, (9,10)), (11, (11,12)), (12, (11,12)),
                   (13, (13,14)), (14, (13,14)), (16, (16,17)),
                   (17, (16,17)), (18, (18,19,20)),
                   (19, (18,19,20)), (20, (18,19,20)), (21, (21,22)),
                   (22, (21,22)), (25, (23,25)))

    return data


# -------------------------------------------------------------------------------------------
# Grating Example 001  -- VALUES ARE not UPDATED
# -------------------------------------------------------------------------------------------
def Rx_Grating_001():

    nElt = 4
    data = {'Rx': 'Grating_example_001.in',
            # --------------------------------
            'Src': Source(ChfRayDir= (0e0, 0e0,   1e0),
                          ChfRayPos= (0e0, 0e0, -40e0),
                          zSource=   1e22,
                          IndRef=    1e0,
                          Extinc=    0e0,
                          Wavelen=   6.5e-04,
                          Flux=      1e0,
                          nGridpts=  256,
                          GridType=  'Circular',
                          xGrid=     (1e0, 0e0, 0e0),
                          yGrid=     (0e0, 1e0, 0e0),
                          Aperture=  2.1305e+02,
                          Obscratn=  0e0,
                          StopElt=   15,
                          BaseUnits= 'mm',
                          WaveUnits= 'mm'),
            'nElt': nElt,
            # --------------------------------
            'KcElt': None,
            # --------------------------------
            'KrElt': None,
            # --------------------------------
            #  (1) Refl. Grating  (0) Trans. Grating
            #           Srf: (Type, Order, RuleWidth, direction)
            'Grating': {1: (1, 1, 0.001, [1, 0, 0]),}
           }

    return data
