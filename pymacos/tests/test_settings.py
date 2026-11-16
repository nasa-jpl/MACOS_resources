import numpy as np

# ---------------------------------------------------------------------
#  define pass / fail tolerances
# ---------------------------------------------------------------------
# pass/fail threshold for data comparisons: abs(actual - desired) > atol + rtol * abs(desired)
_Tol = {'P': (1e-10, 1e-10),          # max Positional  rel. & abs. error
        'r': (1e-13, 1e-13),          # max Directional rel. & abs. error
        'L': (1e-11, 1e-11),          # max Path-Length rel. & abs. error
        'v': (1e-15, 1e-15),          # max value       rel. & abs. error
      'eps': (np.finfo(float).eps)*2} # eps value : 2.2204460492503131e-16 for float64


# ---------------------------------------------------------------------
# Support functions
# ---------------------------------------------------------------------
def Qform(th, omega):
    """
    Defines Rotation Matrix as utilised in MACOS

    :param    th: [1x3] (list, tuple,ndarray): defines the rotation axis
    :param omega: [1x1]                scalar: rotation angle in [radians]

    :return    R: [3x3] (ndarray): Rotation matrix
    """
    qmag = omega
    qhat = th
    cq   = np.cos(qmag)
    sq   = np.sin(qmag)
    omcq = 1-cq
    Q    = np.zeros((3,3))
    Q[0,0] = omcq*qhat[0]**2 + cq
    Q[0,1] = omcq*qhat[0]*qhat[1] - sq*qhat[2]
    Q[0,2] = omcq*qhat[0]*qhat[2] + sq*qhat[1]
    Q[1,0] = omcq*qhat[1]*qhat[0] + sq*qhat[2]
    Q[1,1] = omcq*qhat[1]**2 + cq
    Q[1,2] = omcq*qhat[1]*qhat[2] - sq*qhat[0]
    Q[2,0] = omcq*qhat[2]*qhat[0] - sq*qhat[1]
    Q[2,1] = omcq*qhat[2]*qhat[1] + sq*qhat[0]
    Q[2,2] = omcq*qhat[2]**2 + cq

    return Q
