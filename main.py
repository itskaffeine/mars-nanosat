import numpy as np
import matplotlib.pyplot as plt
import constants as c
import submission as subm
import transformations as trans
import orbit_engine as oe
import spacecraft_engine as se

lmo = oe.Orbit(c.LMO_RADIUS,c.LMO_RAAN,c.LMO_INCL,c.NS_INIT_TRUE_LAT,c.LMO_ORBIT_RATE,c.PLANET_MU)
gmo = oe.Orbit(c.GMO_RADIUS,c.GMO_RAAN,c.GMO_INCL,c.MC_INIT_TRUE_LAT,c.GMO_ORBIT_RATE,c.PLANET_MU)
nanosat = se.Spacecraft(c.NS_INIT, lmo, c.NS_I_B)

# Task 1
lmo450 = lmo.true_lat2state(450)
gmo1150 = gmo.true_lat2state(1150)

subm.create_submission_txt("tests/t1/p1",lmo450[0])
subm.create_submission_txt("tests/t1/p2",lmo450[1])
subm.create_submission_txt("tests/t1/p3",gmo1150[0])
subm.create_submission_txt("tests/t1/p4",gmo1150[1])

# Task 2
hn300 = lmo.dcm_inertial2hill(300)

subm.create_submission_txt("tests/t2/p1",hn300)

# Task 3
RsN = lmo.dcm_inertial2sun() 

subm.create_submission_txt("tests/t3/p1",RsN)
subm.create_submission_txt("tests/t3/p2", np.zeros(3))

# Task 4
RnN330 = lmo.dcm_inertial2nadir(330)
omega_RnN_N330 = lmo.omega_nadir2inertial_N(330)

subm.create_submission_txt("tests/t4/p1",RnN330)
subm.create_submission_txt("tests/t4/p2",omega_RnN_N330)

# Task 5
RcN330 = lmo.dcm_inertial2comms(330,gmo)
omega_RcN_N330 = lmo.omega_comms2inertial_N(330,gmo)

subm.create_submission_txt("tests/t5/p1",RcN330)
subm.create_submission_txt("tests/t5/p2",omega_RcN_N330)

# Task 6
RsN0 = lmo.dcm_inertial2sun()

RnN0 = lmo.dcm_inertial2nadir(0)
omega_RnN_N0 = lmo.omega_nadir2inertial_N(0)

RcN0 = lmo.dcm_inertial2comms(0,gmo)
omega_RcN_N0 = lmo.omega_comms2inertial_N(0,gmo)

BRs_err = nanosat.attitude_error_evaluation(0,c.NS_INIT,RsN0,np.array([0,0,0]))
BRn_err = nanosat.attitude_error_evaluation(0,c.NS_INIT,RnN0,omega_RnN_N0)
BRc_err = nanosat.attitude_error_evaluation(0,c.NS_INIT,RcN0,omega_RcN_N0)

subm.create_submission_txt("tests/t6/p1",BRs_err[:3])
subm.create_submission_txt("tests/t6/p2",BRs_err[3:])
subm.create_submission_txt("tests/t6/p3",BRn_err[:3])
subm.create_submission_txt("tests/t6/p4",BRn_err[3:])
subm.create_submission_txt("tests/t6/p5",BRc_err[:3])
subm.create_submission_txt("tests/t6/p6",BRc_err[3:])

# Task 7
x_task7, u_task7, t_task7, x_err_task7, state_task7 = nanosat.rk4(500, 1, nanosat.zero_torque)
x_500 = x_task7[-1,:]
omega_BN_B_500 = x_500[3:]
angMom_B_500 = c.NS_I_B @ omega_BN_B_500
subm.create_submission_txt("tests/t7/p1",angMom_B_500)

kEnergy_500 = 1/2*omega_BN_B_500.T @ c.NS_I_B @ omega_BN_B_500
subm.create_submission_txt("tests/t7/p2",[kEnergy_500])

sigma_BN_500 = x_500[:3]
subm.create_submission_txt("tests/t7/p3",sigma_BN_500)

dcm_BN_500 = trans.MRP2C(sigma_BN_500)
angMom_N_500 = np.asarray(dcm_BN_500.T @ angMom_B_500)
subm.create_submission_txt("tests/t7/p4",angMom_N_500)

x_task7, u_task7, t_task7, x_err_task7, state_task7 = nanosat.rk4(100, 1, nanosat.const_torque)
x_100 = x_task7[-1,:]
sigma_BN_100 = x_100[:3]
subm.create_submission_txt("tests/t7/p5",sigma_BN_100)

# # Task 8
# print("TASK 8 BEGINS HERE")
# x_task8, u_task8, t_task8, x_err_task8, state_task8 = nanosat.rk4(400, 1, nanosat.sun_pointing_torque)
# sigma_BN_400 = x_task8[400,:3]
# sigma_BN_200 = x_task8[200,:3]
# sigma_BN_100 = x_task8[100,:3]
# sigma_BN_15 = x_task8[15,:3]
# subm.create_submission_txt("tests/t8/p0",[1/6,1/180])
# subm.create_submission_txt("tests/t8/p1",sigma_BN_15)
# subm.create_submission_txt("tests/t8/p2",sigma_BN_100)
# subm.create_submission_txt("tests/t8/p3",sigma_BN_200)
# subm.create_submission_txt("tests/t8/p4",sigma_BN_400)

# # Task 9
# print("TASK 9 BEGINS HERE")
# x_task9, u_task9, t_task9, x_err_task9, state_task9 = nanosat.rk4(400, 1, nanosat.nadir_pointing_torque)
# sigma_BN_400 = x_task9[400,:3]
# sigma_BN_200 = x_task9[200,:3]
# sigma_BN_100 = x_task9[100,:3]
# sigma_BN_15 = x_task9[15,:3]
# subm.create_submission_txt("tests/t9/p1",sigma_BN_15)
# subm.create_submission_txt("tests/t9/p2",sigma_BN_100)
# subm.create_submission_txt("tests/t9/p3",sigma_BN_200)
# subm.create_submission_txt("tests/t9/p4",sigma_BN_400)

# # Task 10
# print("TASK 10 BEGINS HERE")
# x_task10, u_task10, t_task10, x_err_task10, state_task10 = nanosat.rk4(400, 1, nanosat.comms_pointing_torque, gmo)
# sigma_BN_400 = x_task10[400,:3]
# sigma_BN_200 = x_task10[200,:3]
# sigma_BN_100 = x_task10[100,:3]
# sigma_BN_15 = x_task10[15,:3]
# subm.create_submission_txt("tests/t10/p1",sigma_BN_15)
# subm.create_submission_txt("tests/t10/p2",sigma_BN_100)
# subm.create_submission_txt("tests/t10/p3",sigma_BN_200)
# subm.create_submission_txt("tests/t10/p4",sigma_BN_400)

# # Task 11
# print("TASK 11 BEGINS HERE")
# x_task11, u_task11, t_task11, x_err_task11, state_task11 = nanosat.rk4(6500, 1, nanosat.mission_torque, gmo)
# sigma_BN_300 = x_task11[300,:3]
# sigma_BN_2100 = x_task11[2100,:3]
# sigma_BN_3400 = x_task11[3400,:3]
# sigma_BN_4400 = x_task11[4400,:3]
# sigma_BN_5600 = x_task11[5600,:3]
# subm.create_submission_txt("tests/t11/p1",sigma_BN_300)
# subm.create_submission_txt("tests/t11/p2",sigma_BN_2100)
# subm.create_submission_txt("tests/t11/p3",sigma_BN_3400)
# subm.create_submission_txt("tests/t11/p4",sigma_BN_4400)
# subm.create_submission_txt("tests/t11/p5",sigma_BN_5600)

#plotting
t_arr = np.linspace(0, 88620, 100) #7102 #88620

lmo_pos_list, lmo_vel_list = [], []
gmo_pos_list, gmo_vel_list = [], []
lmo_omega_nadir_list = []
lmo_omega_comms_list = []

for t_val in t_arr:
    p_l, v_l = lmo.true_lat2state(t_val)
    lmo_pos_list.append(p_l)
    lmo_vel_list.append(v_l)
    
    p_g, v_g = gmo.true_lat2state(t_val)
    gmo_pos_list.append(p_g)
    gmo_vel_list.append(v_g)
    
    lmo_omega_nadir_list.append(lmo.omega_nadir2inertial_N(t_val))
    lmo_omega_comms_list.append(lmo.omega_comms2inertial_N(t_val,gmo))

lmo_pos = np.array(lmo_pos_list)
lmo_vel = np.array(lmo_vel_list)
gmo_pos = np.array(gmo_pos_list)
gmo_vel = np.array(gmo_vel_list)
lmo_omega_nadir = np.array(lmo_omega_nadir_list)
lmo_omega_comms = np.array(lmo_omega_comms_list)

def finalize_plot(ylabel):
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(r'Time $[s]$', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

# LMO Position
plt.subplot(2,1,1)
plt.plot(t_arr, lmo_pos[:, 0], label=r'$\hat{n}_1$')
plt.plot(t_arr, lmo_pos[:, 1], label=r'$\hat{n}_2$')
plt.plot(t_arr, lmo_pos[:, 2], label=r'$\hat{n}_3$')
finalize_plot(r'Position [$km$]')

# LMO Velocity
plt.subplot(2,1,2)
plt.plot(t_arr, lmo_vel[:, 0], label=r'$\hat{n}_1$')
plt.plot(t_arr, lmo_vel[:, 1], label=r'$\hat{n}_2$')
plt.plot(t_arr, lmo_vel[:, 2], label=r'$\hat{n}_3$')
finalize_plot(r'Velocity [$km/s$]')
plt.show()

# GMO Position
plt.subplot(2,1,1)
plt.plot(t_arr, gmo_pos[:, 0], label=r'$\hat{n}_1$')
plt.plot(t_arr, gmo_pos[:, 1], label=r'$\hat{n}_2$')
plt.plot(t_arr, gmo_pos[:, 2], label=r'$\hat{n}_3$')
finalize_plot(r'Position [$km$]')

# GMO Velocity
plt.subplot(2,1,2)
plt.plot(t_arr, gmo_vel[:, 0], label=r'$\hat{n}_1$')
plt.plot(t_arr, gmo_vel[:, 1], label=r'$\hat{n}_2$')
plt.plot(t_arr, gmo_vel[:, 2], label=r'$\hat{n}_3$')
finalize_plot(r'Velocity [$km/s$]')
plt.show()

# LMO Nadir Angular Velocity
plt.figure()
plt.plot(t_arr, lmo_omega_nadir[:, 0], label=r'$\omega_x$')
plt.plot(t_arr, lmo_omega_nadir[:, 1], label=r'$\omega_y$')
plt.plot(t_arr, lmo_omega_nadir[:, 2], label=r'$\omega_z$')
finalize_plot(r'Angular Velocity [$rad/s$]')
#plt.show()

# LMO Comms Angular Velocity
plt.figure()
plt.plot(t_arr, lmo_omega_comms[:, 0], label=r'$\hat{n}_1$')
plt.plot(t_arr, lmo_omega_comms[:, 1], label=r'$\hat{n}_2$')
plt.plot(t_arr, lmo_omega_comms[:, 2], label=r'$\hat{n}_3$')
finalize_plot(r'Angular Velocity [$rad/s$]')
#plt.show()

