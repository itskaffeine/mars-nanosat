# mars-nanosat
Attitude Dynamics and Control of a Nano-Satellite Orbiting Mars

# Project Overview
This project considers a small satellite orbiting Mars at a low altitude gathering science data. However,
this small satellite needs to transfer this data to a larger mother satellite at a higher altitude. Further, to
keep the batteries from draining all the way, periodically the satellite must point its solar panels at the
sun to recharge. Thus, three mission goals must be considered by the satellite: 1) point sensor platform
straight down at Mars, 2) point communication platform at the mother satellite, and 3) point the solar
arrays at the sun. In all scenarios the small spacecraft and mother craft are on simple circular orbits
whose motion is completely known.
The high-level goal of this project is to design a thruster-based attitude control to achieve these
attitude control scenarios. In order to satisfy the mission requirements, the spacecraft’s body-frame
B must be driven towards various reference body-frames R that corresponds to the desired attitude.
The reference attitude is computed from the knowledge of the spacecraft position and velocity about
Mars, as well as the knowledge of the mother satellite motion for the communication scenario and the
knowledge of the sun heading for the power generation scenario. Once this reference is derived, you will
implement the torque control law u that drives the current attitude MRP σB/N and the angular velocity
ωB/N towards their reference values (σR/N , ωR/N ). The scope of this project encompasses reference
frame generation, attitude characterization and feedback control. By the end of this project, you will
have gained valuable practical experience in the aforementioned areas thanks to analytic derivations
and software implementation of the different project milestones.