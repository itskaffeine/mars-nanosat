import numpy as np
import constants as c
import submission as subm
import orbit_engine as oe
import spacecraft_engine as se


lmo = oe.Orbit(c.LMO_RADIUS,c.LMO_RAAN,c.LMO_INCL,c.LMO_ORBIT_RATE,c.PLANET_MU)
gmo = oe.Orbit(c.GMO_RADIUS,c.GMO_RAAN,c.GMO_INCL,c.GMO_ORBIT_RATE,c.PLANET_MU)

lmo450 = lmo.true_lat2state(c.NS_INIT_TRUE_LAT+np.degrees(450*c.LMO_ORBIT_RATE))
gmo1150 = gmo.true_lat2state(c.MC_INIT_TRUE_LAT+np.degrees(1150*c.GMO_ORBIT_RATE))

anslist = (lmo450,gmo1150)
print(lmo450,gmo1150)

print(np.linalg.norm(anslist[0][0]), c.LMO_RADIUS)
print(np.linalg.norm(anslist[1][0]), c.GMO_RADIUS)


subm.create_submission_txt("tests/t1p1",anslist[0][0])
subm.create_submission_txt("tests/t1p2",anslist[0][1])
subm.create_submission_txt("tests/t1p3",anslist[1][0])
subm.create_submission_txt("tests/t1p4",anslist[1][1])