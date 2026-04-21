import numpy as np
import matplotlib.pyplot as plt
import constants as c
import submission as subm
import transformations as trans
import orbit_engine as oe
import spacecraft_engine as se
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

lmo = oe.Orbit(c.LMO_RADIUS,c.LMO_RAAN,c.LMO_INCL,c.NS_INIT_TRUE_LAT,c.LMO_ORBIT_RATE,c.PLANET_MU)
gmo = oe.Orbit(c.GMO_RADIUS,c.GMO_RAAN,c.GMO_INCL,c.MC_INIT_TRUE_LAT,c.GMO_ORBIT_RATE,c.PLANET_MU)
nanosat = se.Spacecraft(c.NS_INIT, lmo, c.NS_I_B)

x_task11, u_task11, t_task11, x_err_task11, mode_task11 = nanosat.rk4(6500, 1, nanosat.mission_torque, gmo)
sigma_sun_err_hist = x_err_task11[:,:3]
omega_sun_err_hist = x_err_task11[:,3:]
sigma_sun_err_hist_norm = np.linalg.norm(sigma_sun_err_hist, axis=1)
omega_sun_err_hist_norm = np.linalg.norm(omega_sun_err_hist, axis=1)
control_norm = np.linalg.norm(u_task11, axis=1)
att_norm = np.linalg.norm(x_task11[:,3:], axis=1)

t_arr = np.linspace(0, 6500, 100) #7102 #88620

indices = [i for i in range(len(mode_task11) - 1) if mode_task11[i] != mode_task11[i+1]]
print(indices)

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
    plt.ylabel(ylabel, fontsize=12, wrap=True)
    plt.xlabel(r'Time $[s]$', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

# Sun Torque Err Pointing
plt.subplot(2,2,1)
plt.plot(t_task11, sigma_sun_err_hist[:,0], label=r'$\sigma_1$')
plt.plot(t_task11, sigma_sun_err_hist[:,1], label=r'$\sigma_2$')
plt.plot(t_task11, sigma_sun_err_hist[:,2], label=r'$\sigma_3$')
finalize_plot(r'Attitude Errors')

plt.subplot(2,2,2)
plt.plot(t_task11, omega_sun_err_hist[:,0], label=r'$\omega_1$')
plt.plot(t_task11, omega_sun_err_hist[:,1], label=r'$\omega_2$')
plt.plot(t_task11, omega_sun_err_hist[:,2], label=r'$\omega_3$')
finalize_plot(r'Inertial Angular Velocity Errors [$rad/s$]')

plt.subplot(2,2,3)
plt.plot(t_task11, sigma_sun_err_hist_norm)
threshold = sigma_sun_err_hist_norm[0] / np.exp(1)
finalize_plot(r'Attitude Error Magnitude')

plt.subplot(2,2,4)
plt.plot(t_task11, omega_sun_err_hist_norm)
threshold = omega_sun_err_hist_norm[0] / np.exp(1)
finalize_plot(r'Inertial Angular Velocity Error Magnitude [$rad/s$]')
plt.show()

# LMO Mission Pointing
plt.subplot(2,1,1)
plt.plot(t_task11, x_task11[:,0], label=r'$\sigma_1$')
plt.plot(t_task11, x_task11[:,1], label=r'$\sigma_2$')
plt.plot(t_task11, x_task11[:,2], label=r'$\sigma_3$')
finalize_plot(r'Attitude')

plt.subplot(2,1,2)
plt.plot(t_task11, mode_task11[:])
ticks = [-1, 0, 1]
labels = ['GMO-pointing', 'Sun-pointing', 'Nadir-pointing']
plt.yticks(ticks, labels)
finalize_plot(r'Mission Mode')
plt.show()

# Control Plotting
plt.subplot(2,2,1)
plt.plot(t_task11, x_task11[:,3], label=r'$\omega_1$')
plt.plot(t_task11, x_task11[:,4], label=r'$\omega_2$')
plt.plot(t_task11, x_task11[:,5], label=r'$\omega_3$')
finalize_plot(r'Inertial Angular Velocity [$rad/s$]')

plt.subplot(2,2,2)
plt.plot(t_task11, u_task11[:,0], label=r'$\tau_1$')
plt.plot(t_task11, u_task11[:,1], label=r'$\tau_2$')
plt.plot(t_task11, u_task11[:,2], label=r'$\tau_3$')
finalize_plot(r'Control Torque [$Nm$]')

plt.subplot(2,2,3)
plt.plot(t_task11, att_norm, label=r'$\omega_1$')
finalize_plot(r'Inertial Angular Velocity Magnitude [$rad/s$]')

plt.subplot(2,2,4)
plt.plot(t_task11, control_norm[:], label=r'$\tau$')
finalize_plot(r'Control Torque Magnitude [$Nm$]')
plt.show()

# Mars Trajectory Plotting
res = 100
phi = np.linspace(0, 2*np.pi, res)
theta = np.linspace(0, np.pi, res)
phi, theta = np.meshgrid(phi, theta)

x = c.PLANET_RADIUS * np.sin(theta) * np.cos(phi)
y = c.PLANET_RADIUS * np.sin(theta) * np.sin(phi)
z = c.PLANET_RADIUS * np.cos(theta)

img = Image.open('mars_texture.jpg').convert('RGB').resize((res, res))
img_data = np.array(img)

colorscale = []
for j in range(res):
    for i in range(res):
        r, g, b = img_data[j, i]
        colorscale.append([ (j*res + i)/(res*res), f'rgb({r},{g},{b})' ])

surf_color = np.arange(res*res).reshape((res, res))
fig = go.Figure()
fig.add_trace(go.Surface(
    x=x, y=y, z=z,
    surfacecolor=surf_color,
    colorscale=colorscale,
    showscale=False,
    hoverinfo='none',
    name='Mars'
))

# Add Trajectories
fig.add_trace(go.Scatter3d(
    x=lmo_pos[:,0], y=lmo_pos[:,1], z=lmo_pos[:,2],
    mode='lines+text',
    line=dict(color='blue', width=5),
    name='LMO',
    text=[None] * (len(lmo_pos)-1) + ['LMO'], 
    textposition="top center",
    textfont=dict(size=18, color='blue'),
    showlegend=False
))

fig.add_trace(go.Scatter3d(
    x=gmo_pos[:,0], y=gmo_pos[:,1], z=gmo_pos[:,2],
    mode='lines+text',
    line=dict(color='orange', width=5),
    name='GMO',
    text=[None] * (len(gmo_pos)-1) + ['GMO'], 
    textposition="top center",
    textfont=dict(size=18, color='orange'),
    showlegend=False
))

# 5. Perfect Aspect Ratio & Dark Theme
fig.update_layout(
    scene=dict(
        aspectmode='data',
        xaxis=dict(
            title=dict(text='<br>n1 [km]<br>', font=dict(size=22)),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title=dict(text='<br>n2 [km]<br>', font=dict(size=22)),
            tickfont=dict(size=16)
        ),
        zaxis=dict(
            title=dict(text='<br>n3 [km]<br>', font=dict(size=22)),
            tickfont=dict(size=16)
        ),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    # Ensure legend is visible and styled
    showlegend=True,
    legend=dict(
        font=dict(size=18),
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)" # Semi-transparent background
    ),
    margin=dict(l=80, r=80, b=80, t=80)
)

pio.renderers.default = 'browser'
fig.show(include_mathjax='cdn')