### TELLS_v1.0.py

Thermo-Electrodynamic Lightning Leader Simulation (TELLS) model (version 1.0)

This is a Python implementation of a model for the simulation of upward lightning leader initiation and dynamics.

############################PARAMETERS######################
accepts float
-rod_end: the height of the triggering object in m
-E_amb: ambient field in V/m, is a constant value
-rq_rod: radius in m of the charge around the central current channel in the triggering object
-rq_air: radius in m of the charge around the central current channel in the leader (corona sheath)
-con_cu: conductivity in S/m of triggering object. High conductivities require more fine temporal resolution (scaling linearly)
-con_air: conductivity in S/m of ambient air
-r_cu: radius in m of triggering object, assumed constant and cylindrical
-r_I: radius in m of current channel in leader/streamer objects
-T_amb: ambient temperature in K. Also used as minimum temperature
-E_step: the size of the streamer zone in m when it is established and the subsequent step length of the leaders. Must be multiple of space_seg_len
-altitude: altitude of triggering object base in m. Used for air density calculations
-E_stab: stability field in V/m. Used to enable streamer flags
-sigstr: conductivity in S/m of the streamer zone after the zone is established
-thermal_rad_factor: factor of the thermal radius over the current radius. For example, with a current radius r_I of 1mm, thermal_rad_factor = 2 will result in a thermal radius r_g of 2mm.

############################SETTINGS######################
accepts True/False
-ignite_negative: If True, several parameters (listed below in NEGATIVE LEADER MULTIPLIERS) will be replaced to simulate a negative polarity leader
-wait_to_start: If True, the timer to maximum time tmax will not start until the leader is initiated. Recording to the output .txt file is also not done. Will attempt to ignite a leader for 1 ms before quitting

############################SIMULATION######################
accepts float
-tmax: Maximum simulation time in seconds
-tstep: Time step in seconds
-picno: Number of evenly spaced simulation snapshots to save to text file
-rod_seg_len: grid size of triggering object in m, each cylindrical component of object will have this height
-domain_length: maximum height of leader in m, not including triggering object
-space_seg_len: grid size of leader in m, each cylindrical component of the leader/streamer will have this height
-crit_temp: temperature in K when leader flags are enabled. At this temperature, the LTE conductivity takes over in the conductivity relaxation equation

############################NEGATIVE LEADER MULTIPLIERS######################
accepts float
these multipliers will be applied to these parameters when ignite_negative=True
-rq_mult: charge radius r_q multiplier
-E_stab_mult: stability field E_stab multiplier
-sigstr_mult: streamer zone conductivity sigstr multiplier
-E_step_mult: streamer zone length E_step multiplier
-tau_mult: timescale multiplier
-thermal_rad_mult: thermal radius r_g multiplier

############################ERRORS######################
IF YOU ARE GETTING NAN OR QUIT ERRORS, COMMON FIXES:
!!!-reduce time step tstep. 1 nanosecond (1e-9) is the default when triggering object conductivity is 60e4, for each order of magnitude gain in conductivity, decrease time step by an order of magnitude
!!-make sure that space_seg_len is not larger than rq_air, that is space_seg_len < rq_air
!-make sure that rod_seg_len is not larger than rq_rod, that is rod_seg_len < rq_rod
!!!-make sure that the domain_length is larger than E_step
!!!-make sure that E_step is a multiple of space_seg_len
