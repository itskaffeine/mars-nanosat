import numpy as np
import transformations as trans
class Orbit:
    """ Defines an arbitrary circular orbit around a target planet.

    The circular orbit is given in terms on the RAAN, inclination, and true latitude.

    Args:
        radius (float) - orbital radius from planetary center [km]
        RAAN (float) - right ascension of the ascending node [deg]
        inclination - orbital inclination [deg]
        rate - time derivative of the true latitude [rad/s]
        mu - standard gravitational parameter of the planet [km^3/s^2]
    """
    def __init__(self, radius, RAAN, inclination, rate, mu):
        self.radius = radius;
        self.RAAN = RAAN;
        self.incl = inclination;
        self.rate = rate;
        self.mu = mu;

    def true_lat2state(self, true_lat):
        """ Transforms orbital coordinates to an inertial state vector.

        Args:
            true_lat (float) - true latitude of orbital coords [deg].

        Ret:
            r_N, v_N (tuple of 1x3 numpy arrays)
            -> Inertial position [km]
            -> Inertial velocity [km/s]
        """
        orbital_speed = np.sqrt(self.mu/self.radius) #  [km/s]
        angles = (self.RAAN, self.incl, true_lat) #     [deg]

        r_O = np.array([self.radius, 0, 0])
        v_O = np.array([0, orbital_speed, 0])

        ON = trans.eul2dcm(angles, "313", degrees=True)

        r_N = ON.T @ r_O
        v_N = ON.T @ v_O

        return r_N, v_N
