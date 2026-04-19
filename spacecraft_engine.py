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
        self.I_B_inv = np.linalg.inv(I_B)

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
    
    def rk4(self, tmax, dt, control, target=None):
        t = 0
        x = self.x0
        u = control(x,t,target)
        x_hist = [x.copy()]
        u_hist = [u]
        t_hist = [t]

        while t < tmax:
            k1 = dt*self.dynamics(x,t,u)
            k2 = dt*self.dynamics(x+k1/2, t+dt/2, u)
            k3 = dt*self.dynamics(x+k2/2, t+dt/2, u)
            k4 = dt*self.dynamics(x+k3, t+dt, u)
            x = x + 1/6*(k1+2*k2+2*k3+k4)

            sigma_sq = np.dot(x[:3], x[:3])
            if sigma_sq > 1:
                x[:3] = -x[:3]/sigma_sq
            
            t = t+dt
            u = control(x,t,target)

            x_hist.append(x.copy())
            t_hist.append(t)
            u_hist.append(u)
            print(x,t,u)
            
        return np.array(x_hist), np.array(u_hist), np.array(t_hist)

    def dynamics(self, state, t, u):

        sigma = state[:3]
        omega = state[3:]

        sigma_tilde = np.array([[0,-sigma[2],sigma[1]], [sigma[2],0,-sigma[0]], [-sigma[1],sigma[0],0]])
        B_sigma = 1/4 * ((1-np.dot(sigma,sigma))*np.eye(3) + 2*sigma_tilde + 2*np.outer(sigma,sigma))
        sigma_dot = B_sigma @ omega

        omega_dot = self.I_B_inv @ (u - np.cross(omega, self.I_B @ omega))

        return np.concatenate([sigma_dot, omega_dot])
    
    def sun_pointing_torque(self, x, t, target=None):
        dcm_RsN = self.orbit.dcm_inertial2sun()
        x_err = self.attitude_error_evaluation(t, x, dcm_RsN, np.array([0,0,0]))
        u_B = -1/180 * x_err[:3] - 1/6 * x_err[3:]
        return u_B
    
    def nadir_pointing_torque(self, x, t, target=None):
        dcm_RnN = self.orbit.dcm_inertial2nadir(t)
        omega_RnN_N = self.orbit.omega_nadir2inertial_N(t)
        x_err = self.attitude_error_evaluation(t, x, dcm_RnN, omega_RnN_N)
        u_B = -1/180 * x_err[:3] - 1/6 * x_err[3:]
        return u_B

    def comms_pointing_torque(self, x, t, target=None):
        dcm_RcN = self.orbit.dcm_inertial2comms(t, target)
        omega_RcN_N = self.orbit.omega_comms2inertial_N(t, target)
        x_err = self.attitude_error_evaluation(t, x, dcm_RcN, omega_RcN_N)
        u_B = -1/180 * x_err[:3] - 1/6 * x_err[3:]
        return u_B
    
    def mission_torque(self, x, t, target=None):
        R_N_local = self.orbit.true_lat2state(t)[0]
        R_N_target = target.true_lat2state(t)[0]
        pos_angle = np.acos(np.dot(R_N_local,R_N_target)/(np.linalg.norm(R_N_local)*np.linalg.norm(R_N_target)))

        if R_N_local[2] > 0:
            print("SUN POINTING")
            return self.sun_pointing_torque(x,t)
        elif np.rad2deg(pos_angle) < 35:
            print("COMMS POINTING")
            return self.comms_pointing_torque(x,t,target)
        else:
            print("NADIR POINTING")
            return self.nadir_pointing_torque(x,t)

def zero_torque(x, t, target=None):
    return [0,0,0]

def const_torque(x, t, target=None):
    return [.01,-.01,.02]


