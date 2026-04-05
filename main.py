import numpy as np
import constants as c
import submission as subm
import orbit_engine as oe
import spacecraft_engine as se

lmo = oe.Orbit(c.LMO_RADIUS,c.LMO_RAAN,c.LMO_INCL,c.LMO_ORBIT_RATE,c.PLANET_MU)
gmo = oe.Orbit(c.GMO_RADIUS,c.GMO_RAAN,c.GMO_INCL,c.GMO_ORBIT_RATE,c.PLANET_MU)
nanosat = se.Spacecraft(c.NS_INIT,c.NS_INIT_TRUE_LAT, lmo, c.NS_I_B)

# Task 1
lmo450 = lmo.true_lat2state(c.NS_INIT_TRUE_LAT+np.degrees(450*c.LMO_ORBIT_RATE))
gmo1150 = gmo.true_lat2state(c.MC_INIT_TRUE_LAT+np.degrees(1150*c.GMO_ORBIT_RATE))

subm.create_submission_txt("tests/t1p1",lmo450[0])
subm.create_submission_txt("tests/t1p2",lmo450[1])
subm.create_submission_txt("tests/t1p3",gmo1150[0])
subm.create_submission_txt("tests/t1p4",gmo1150[1])

# Task 2
hn300 = nanosat.dcm_inertial2hill(300)

subm.create_submission_txt("tests/t2p1",hn300)

# Task 3
RsH = nanosat.dcm_inertial2sun() 

subm.create_submission_txt("tests/t3p1",RsH)
subm.create_submission_txt("tests/t3p2", np.zeros(3))

# Task 4
RnN330 = nanosat.dcm_inertial2nadir(330)
omega_RnN_N330 = nanosat.omega_nadir2inertial_N(330)
print(RnN330)
print(omega_RnN_N330)
subm.create_submission_txt("tests/t4p1",RnN330)
subm.create_submission_txt("tests/t4p2",omega_RnN_N330)
