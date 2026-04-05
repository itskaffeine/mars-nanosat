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
        dcm_HN = trans.eul2dcm(angles, "313", degrees=True)

        return dcm_HN
    
    def dcm_inertial2sun(self):
        """ Provides direction cosine matrix from inertial frame to sun-pointing reference frame.

        Note that the inertial frame is psudeo inertial s.t. the sun is always along n2.

        Ret:
            dcm (3x3 numpy array) - direction cosine matrix from inertial frame to sun-pointing reference frame
        """
        r3 = np.array([0,1,0])  # desire b_3 pointing along n_2 (sun direction)
        r1 = np.array([-1,0,0]) # fixing attitude s.t. b_1 along -n_1
        dcm_RsN = trans.axes2dcm(r1,r3,"xz")

        return dcm_RsN
    
    def dcm_inertial2nadir(self, t):
        """ Provides direction cosine matrix from inertial frame to nadir-pointing reference frame.

        Args:
            t (float) - time on orbit (s)

        Ret:
            dcm (3x3 numpy array) - direction cosine matrix from inertial frame to nadir-pointing reference frame
        """
        dcm_HN = self.dcm_inertial2hill(t)

        r1 = np.array([-1,0,0]) # desire b_1 pointing along -i_r (nadir direction)
        r2 = np.array([0,1,0])  # fixing attitude s.t b_2 along i_theta

        dcm_RnH = trans.axes2dcm(r1,r2,"xy")
        dcm_RnN = dcm_RnH @ dcm_HN

        return dcm_RnN
    
    def omega_nadir2inertial_N(self, t):
        """ Computes inertial rotational velocity of the nadir-pointing reference frame with respect to the inertial frame.

        Args:
            t (float) - time on orbit (s)

        Ret:
            omega_RnN_N (3x1 numpy array) - inertial rotational velocity of the nadir-pointing frame with respect to the inertial frame (rad/s)
        """
        true_lat  = np.radians(self.true_lat) + t*self.orbit.rate
        angles = (self.orbit.RAAN, self.orbit.incl, true_lat)
        
        # kinematic matrix which premultiplies with 313 euler angle rates to compute angular velocity
        k_matrix = np.array([[np.sin(angles[2])*np.sin(angles[1]), np.cos(angles[2]), 0], \
                        [np.cos(angles[2])*np.sin(angles[1]), -np.sin(angles[2]), 0], \
                        [np.cos(angles[1]), 0, 1]])
        
        # hill frame has identical angular velocity to nadir-pointing reference frame (both are body fixed)
        omega_HN_H = k_matrix @ np.array([0,0,self.orbit.rate])

        dcm_HN = self.dcm_inertial2hill(t)

        omega_RnN_N = dcm_HN.T @ omega_HN_H

        return omega_RnN_N
