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

x_task8, u_task8, t_task8, x_err_task8, mode_task8 = nanosat.rk4(400, 1, nanosat.sun_pointing_torque)

sigma_sun_err_hist = x_err_task8[:,:3]
omega_sun_err_hist = x_err_task8[:,3:]
sigma_sun_err_hist_norm = np.linalg.norm(sigma_sun_err_hist, axis=1)
omega_sun_err_hist_norm = np.linalg.norm(omega_sun_err_hist, axis=1)

def finalize_plot(ylabel):
    plt.ylabel(ylabel, fontsize=12, wrap=True)
    plt.xlabel(r'Time $[s]$', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

# Sun Torque Err Pointing
plt.subplot(2,2,1)
plt.plot(t_task8, sigma_sun_err_hist[:,0], label=r'$\sigma_1$')
plt.plot(t_task8, sigma_sun_err_hist[:,1], label=r'$\sigma_2$')
plt.plot(t_task8, sigma_sun_err_hist[:,2], label=r'$\sigma_3$')
finalize_plot(r'Attitude Errors')

plt.subplot(2,2,2)
plt.plot(t_task8, omega_sun_err_hist[:,0], label=r'$\omega_1$')
plt.plot(t_task8, omega_sun_err_hist[:,1], label=r'$\omega_2$')
plt.plot(t_task8, omega_sun_err_hist[:,2], label=r'$\omega_3$')
finalize_plot(r'Angular Velocity Errors [$rad/s$]')

ax = plt.subplot(2,2,3)
plt.plot(t_task8, sigma_sun_err_hist_norm)
threshold = sigma_sun_err_hist_norm[0] / np.exp(1)
plt.axhline(threshold, color='r', linestyle='--', zorder=5) 
ax.text(120, 1.02, '120', transform=ax.get_xaxis_transform(), 
        ha='center')
plt.axvline(120, color='r', linestyle='--', zorder=5)
ax.text(1.01, threshold, r'$|\sigma|_0/e$', transform=ax.get_yaxis_transform(), 
        va='center')
finalize_plot(r'Attitude Error Magnitude')

ax = plt.subplot(2,2,4)
plt.plot(t_task8, omega_sun_err_hist_norm)
threshold = omega_sun_err_hist_norm[0] / np.exp(1)
plt.axhline(threshold, color='r', linestyle='--', zorder=5) 
ax.text(120, 1.02, '120', transform=ax.get_xaxis_transform(), 
        ha='center')
plt.axvline(120, color='r', linestyle='--', zorder=5)
ax.text(1.01, threshold, r'$|\omega|_0/e$', transform=ax.get_yaxis_transform(), 
        va='center')
finalize_plot(r'Angular Velocity Error Magnitude [$rad/s$]')
plt.show()