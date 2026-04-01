import numpy as np

# Planetary Properties
PLANET_RADIUS = 3396.19 #             [km]
PLANET_MU = 42828.3 #                 [km^3/s^2]
PLANET_ROTATION_PERIOD = 88620 #      [s]

# Nanosat Orbit Properties
# Assuming circular orbit
LMO_HEIGHT = 400 #                      [km]
LMO_RADIUS = PLANET_RADIUS + LMO_HEIGHT # [km]
LMO_RAAN = 20 #                         [deg]
LMO_INCL = 30 #                         [deg]
LMO_ORBIT_RATE = .000884797 #           [rad/s]

# Mothercraft Orbit Properties
# Assuming circular orbit
GMO_RADIUS = 20424.2 #          [km]
GMO_RAAN = 0 #                  [deg]
GMO_INCL = 0 #                  [deg]
GMO_ORBIT_RATE = .0000709003 #  [rad/s]

# Nanosat Initial Conditions
NS_INIT_ATT_MRP = np.array([.3, -.4, .5])
NS_INIT_OMEGA_B = np.array([1.00, 1.75, -2.20]) #           [deg/s]
NS_INIT_TRUE_LAT = 60 #                                     [deg]
NS_INIT = np.concatenate([NS_INIT_ATT_MRP, NS_INIT_OMEGA_B])
# Nanosat Physical Constants
NS_I_B = np.array([10,0,0], [0,5,0], [0,0,7.5]) #           [kg m^2]

# Mothercraft Initial Conditions
MC_INIT_TRUE_LAT = 250 # [deg]