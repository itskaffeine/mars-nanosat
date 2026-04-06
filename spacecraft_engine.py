import numpy as np
import transformations as trans
class Spacecraft:
    """ Defines a spacecraft at some initial true latitude along a given orbit.

    The spacecraft is bound to an orbit object which provides environmental info.

    Params:
        init (1x6 numpy array) - concatenated state vector
        -> init[:3] 1x3 modified roderigues parameter of body attitude w.r.t the inertial frame
        -> init[3:] 1x3 body angular velocity w.r.t the inertial frame [deg/s]
        true_lat_init (float) - initial true latitude along the provided orbit [deg]
        orbit (obj) - orbit class object
        I_B (3x3 numpy array) - body frame inertia tensor [kg m^2]
    """
    def __init__(self, init, orbit, I_B):
        self.x0 = init;
        self.orbit = orbit;
        self.I_B = I_B;

    def attitude_error_evaluation(self, t, x, dcm_RN, omega_RN_N):
        """ Computes the tracking errors for body frame attitude and angular velocity with respect to a given reference frame.

        Params:
            t (float) - time on orbit (s)
            x = (1x6 numpy array) - initial state vector
            -> x[:3] 1x3 modified roderigues parameter of body attitude w.r.t the inertial frame
            -> x[3:] 1x3 body angular velocity w.r.t the inertial frame [deg/s]
            dcm_RN (3x3 numpy array) - direction cosine matrix from inertial frame to reference frame
            omega_RN_N omega_RnN_N (3x1 numpy array) - inertial rotational velocity of the reference frame with respect to the inertial frame [rad/s]

        Ret:
            x_err = (1x6 numpy array) - tracking errors for body frame attitude and angular velocity with respect to the reference frame
            -> x_err[:3] 1x3 modified roderigues parameter of attitude error from the reference frame
            -> x_err[3:] 1x3 body angular velocity w.r.t the reference frame [deg/s]
        """
        sigma_BN = x[:3]
        omega_BN_B = x[3:]

        sigma_RN = trans.C2MRP(dcm_RN)
        sigma_BR = trans.subMRP(sigma_BN,sigma_RN)

        dcm_BN = trans.MRP2C(sigma_BN)

        omega_RN_B = dcm_BN @ omega_RN_N
        omega_RN_B = np.asarray(omega_RN_B).flatten()

        omega_BR_B = (omega_BN_B - omega_RN_B)

        return np.concatenate([sigma_BR, omega_BR_B])