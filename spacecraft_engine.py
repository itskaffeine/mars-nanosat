import numpy as np
import transformations as trans
class Spacecraft:
    """ Defines a spacecraft at some initial true latitude along a given orbit.

    The spacecraft is bound to an orbit object which provides environmental info.

    Args:
        init (1x6 numpy array) - concatenated state vector
        -> init[:4] 1x3 modified roderigues parameter of body attitude w.r.t the inertial frame
        -> init[4:] 1x3 body angular velocity w.r.t the inertial frame
        true_lat_init (float) - initial true latitude along the provided orbit [deg]
        orbit (obj) - orbit class object
        I_B (3x3 numpy array) - body frame inertia tensor [kg m^2]
    """
    def __init__(self, init, true_lat_init, orbit, I_B):
        self.x = init;
        self.true_lat = true_lat_init;
        self.orbit = orbit;
        self.I_B = I_B;

    def dcm_inertial2hill(self, t):
        """ Provides direction cosine matrix from inertial frame to hill frame.
        Args:
            t (float) - time on orbit (s)

        Ret:
            dcm (3x3 numpy array) - direction cosine matrix from inertial frame to hill frame
        """
        true_lat = self.true_lat + np.degrees(t*self.orbit.rate)
        angles = (self.orbit.RAAN, self.orbit.incl, true_lat)
        HN = trans.eul2dcm(angles, "313", degrees=True)

        return HN