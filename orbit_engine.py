import numpy as np
import transformations as trans
import numerical_tooling as numtools
class Orbit:
    """ Defines an arbitrary circular orbit around a target planet.

    The circular orbit is given in terms on the RAAN, inclination, and true latitude.

    Params:
        radius (float) - orbital radius from planetary center [km]
        RAAN (float) - right ascension of the ascending node [deg]
        inclination - orbital inclination [deg]
        rate - time derivative of the true latitude [rad/s]
        mu - standard gravitational parameter of the planet [km^3/s^2]
    """
    def __init__(self, radius, RAAN, inclination, true_lat_init, rate, mu):
        self.radius = radius;
        self.RAAN = RAAN;
        self.incl = inclination;
        self.true_lat = true_lat_init;
        self.rate = rate;
        self.mu = mu;

    def get_true_lat(self, t):
        """ Updates initial true latitude for a given time

        Params:
            t (float) - time on orbit (s)

        Ret:
            true_latitude (float) - true latitude [deg]
        """
        return self.true_lat + np.degrees(t*self.rate)

    def true_lat2state(self, t):
        """ Transforms orbital coordinates to an inertial state vector.

        Params:
            t (float) - time on orbit (s)

        Ret:
            r_N, v_N (tuple of 1x3 numpy arrays)
            -> Inertial position [km]
            -> Inertial velocity [km/s]
        """
        true_lat = self.get_true_lat(t)

        orbital_speed = np.sqrt(self.mu/self.radius) #  [km/s]
        angles = (self.RAAN, self.incl, true_lat) #     [deg]

        r_O = np.array([self.radius, 0, 0])
        v_O = np.array([0, orbital_speed, 0])

        ON = trans.eul2dcm(angles, "313", degrees=True)

        r_N = ON.T @ r_O
        v_N = ON.T @ v_O

        return r_N, v_N
    
    def dcm_inertial2hill(self, t):
        """ Provides direction cosine matrix from inertial frame to hill frame.

        Params:
            t (float) - time on orbit (s)

        Ret:
            dcm (3x3 numpy array) - direction cosine matrix from inertial frame to hill frame
        """
        true_lat = self.get_true_lat(t)
        angles = (self.RAAN, self.incl, true_lat)
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

        Params:
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

        Params:
            t (float) - time on orbit (s)

        Ret:
            omega_RnN_N (3x1 numpy array) - inertial rotational velocity of the nadir-pointing frame with respect to the inertial frame (rad/s)
        """
        true_lat  = np.radians(self.true_lat) + t*self.rate
        angles = (self.RAAN, self.incl, true_lat)
        
        # kinematic matrix which premultiplies with 313 euler angle rates to compute angular velocity
        k_matrix = np.array([[np.sin(angles[2])*np.sin(angles[1]), np.cos(angles[2]), 0], \
                        [np.cos(angles[2])*np.sin(angles[1]), -np.sin(angles[2]), 0], \
                        [np.cos(angles[1]), 0, 1]])
        
        # hill frame has identical angular velocity to nadir-pointing reference frame (both are body fixed)
        omega_HN_H = k_matrix @ np.array([0,0,self.rate])

        dcm_HN = self.dcm_inertial2hill(t)

        omega_RnN_N = dcm_HN.T @ omega_HN_H

        return omega_RnN_N
    
    def dcm_inertial2comms(self, t, target):
        """ Provides direction cosine matrix from inertial frame to comms-pointing reference frame.

        The communication frame points towards a target satellite.

        Params:
            t (float) - time on orbit (s)
            target (orbit class object) - spacecraft recieving communication broadcast

        Ret:
            dcm (3x3 numpy array) - direction cosine matrix from inertial frame to nadir-pointing reference frame
        """
        R_N_local = self.true_lat2state(t)[0]
        R_N_target = target.true_lat2state(t)[0]

        # build dcm around inertial vector pointing to target
        Delta_R_N = R_N_target - R_N_local
        r2_full = np.cross(Delta_R_N, np.array([0,0,1]))

        r1 = -Delta_R_N / np.linalg.norm(Delta_R_N) # desire -b_1 pointing along Delta_R (comms direction)
        r2 = r2_full / np.linalg.norm(r2_full)      # fixing attitude s.t. b_2 along [Delta_R x n_3]

        dcm_RcN = trans.axes2dcm(r1,r2,"xy")

        return dcm_RcN

    def omega_comms2inertial_N(self, t, target):
        """ Computes inertial rotational velocity of the comms-pointing reference frame with respect to the inertial frame.

        Params:
            t (float) - time on orbit (s)
            target (orbit class object) - spacecraft recieving communication broadcast

        Ret:
            omega_RcN_N (3x1 numpy array) - inertial rotational velocity of the comms-pointing frame with respect to the inertial frame (rad/s)
        """
        dcm_RcN = self.dcm_inertial2comms(t, target)

        # very difficult analytical problem, instead take derivative with scipy module
        dcm_RcN_dot = numtools.matrix_derivative(self.dcm_inertial2comms, t, shape=(3,3), args=(target,))

        # rearrangement of the dcm kinematic differential equation
        omega_tilde = -dcm_RcN_dot @ dcm_RcN.T
        omega_RcN_Rc = np.array([-omega_tilde[1,2], omega_tilde[0,2], -omega_tilde[0,1]])
        # dcm kinematic diffeq returns omega in rotating frame, convert
        omega_RcN_N = dcm_RcN.T @ omega_RcN_Rc 

        return omega_RcN_N