# -*- coding: utf-8 -*-
"""

Thermo-Electrodynamic Lightning Leader Simulation (TELLS) model v1.0

Created on Tue Jun 30 13:57:33 2026

@authors: John Pantuso & Caitano da Silva, New Mexico Tech
"""

#TELLS Model v1.0
#Thermo-
#Electro
#Lightning
#Leader
#Simulation
import numpy as np
import os
from datetime import datetime
from numba import njit

np.set_printoptions(legacy='1.25')

############################set this to the location you want the running folder to appear in
###############CHANGE THIS!!!!!!!!########################################
#datapath = ################your data location here. This is where output will be written to

@njit(parallel = False)
def Eflagger(Eflags,sigflags,sigmajorflags,indexlist,E,sigma,E_stab,s_rod,steps_per_E_check,sigstr):
    for i in indexlist:
        chunk = np.mean(E[i-steps_per_E_check+1:i+1])
        if np.abs(chunk) > np.abs(E_stab):
            for j in range(i-steps_per_E_check+1,i+1):
                Eflags[j] = 1
        if sigma[i] > 0.63*sigstr:
            for j in range(len(s_rod),i+1):
                sigflags[j] = 1
        if sigma[i] > sigstr:
            for j in range(len(s_rod),i+1):
                sigmajorflags[j] = 1
    return Eflags, sigflags, sigmajorflags

#############this is the loop that iterates charge###############
@njit(parallel = True)
def chargeiterate(q,A,s,E_amb,G,tstep,I):
    #grab the potential distribution
    V = np.dot(A,q[1:])
    V = np.append(-V[1],V)
    #create the electric field distribution
    E = (-np.diff(V)/(np.diff(s))) + E_amb
    #current distribution
    I = np.multiply(E,G)
    dI = np.diff(I) #/abs((si[i+1]-si[i]))
    dI = np.append(0,dI)
    dI = np.append(dI,0)
    #fix charges
    q[0] = -q[2]
    q[1] = 0
    return V,E,dI,I


################this is the loop that iterates temperature###################
@njit(parallel = True)
def conductiterate(etaT,sigmalte,T,T_amb,sigma,sigstr,Eflags,sigflags,sigmajorflags,tau,tau_heating,tstep,con_air,G,s_space,s_rod,r_I,radeps,k_t,Ch,joule_heating,thermal_cooling,radiative_cooling,vertical_conduction,E,dT,space_seg_len,E_step,Tflags,streamer_on,leader_on,leader_decay,delta,E_stab,ignite_negative,crit_temp,thermal_rad_factor,thermal_rad_mult):
    #to prevent NAN outs
    T[-1] = T[-2]
    if 0 in T:
        print("ZERO ERROR IN TEMP!")
    if 0 in E:
        print("ZERO ERROR IN FIELD!")
    Ekj = 0.71*(T_amb/T)*30e5
    etaT = (0.1 + 0.9*(np.tanh(T/T_amb - 4) + 1)/2)
    for i in range(len(etaT)):
        if E[i] != 0 and np.abs(3*Ekj[i]/E[i]) < 1:
            etaT[i] = (0.1 + 0.9*(np.tanh(T[i]/T_amb - 4) + 1)/2)*np.abs((3*Ekj[i]/E[i]))
    for i in range(len(etaT)):
        if sigflags[i] == 0:
            etaT[i] = 0
    sigmalte = 0.4*T**(9/8)*np.exp(-13539/T)
    dsigma = np.zeros(len(sigma))
    for i in range(len(s_rod)+1,len(sigma)):
        if Eflags[i] == 1 and sigmajorflags[i-(int(E_step/space_seg_len))] == 1:
            streamer_on[i] = 1
        if T[i] >= crit_temp:
            leader_on[i] = 1
            streamer_on[i] = 0
        if leader_on[i] == 1 and T[i] <= crit_temp and E[i] < Ekj[i]:
            leader_decay[i] = 1
        if leader_decay[i] == 1 and E[i] > Ekj[i]:
            leader_decay[i] = 0
            streamer_on[i] = 1
        dsigma[i] = ((streamer_on[i]*(sigma[i] - sigstr))/tau) + ((leader_on[i]*(sigma[i] - sigmalte[i]))/tau_heating)
    for i in range(len(G)-len(s_space)+1,len(G)):
        G[i] = sigma[i]*np.pi*r_I**2
    radeps = 1.1e10*(T**0.63)*np.exp(-98000/T)
    k_t = 6.2e-2*(T/300) 
    Ch = (820*(T/300)**-0.5)*delta
    if 0 in Ch:
        print("ZERO ERROR IN HEAT CAPACITY!")
    r_g = thermal_rad_factor*r_I
    if ignite_negative == True:
        r_g = thermal_rad_mult*r_g
    for i in range(len(T)-len(s_space)+1,len(T)):
        joule_heating[i] = etaT[i] * sigma[i] * np.abs(E[i])**2
        thermal_cooling[i] = 4*k_t[i]*(T[i]-T_amb)/((r_g)**2)
        radiative_cooling[i] = 4*radeps[i]*np.pi
        vertical_conduction[i] = (T[i+1]-2*T[i]+T[i-1])*k_t[i]/(space_seg_len**2)
        dT[i] = ((joule_heating[i])-thermal_cooling[i]-radiative_cooling[i] + vertical_conduction[i])/Ch[i]  
    for i in range(len(T)):
        if T[i] < T_amb:
            T[i] = T_amb
    return etaT,radeps,k_t,Ch,sigmalte,sigma,G,joule_heating,thermal_cooling,radiative_cooling,vertical_conduction,dT,dsigma,streamer_on,leader_on,leader_decay

###########################Parameters###############################
###########################FOR POSITIVE LEADERS
###########################FOR NEGATIVE LEADERS, SEE NEGATIVE LEADER MULTIPLIERS
rod_end = 350 #m #height of triggering object (copper wire)
E_amb = 30e3 #V/m #ambient electric field
rq_rod = 1 #m #charge radius around wire
rq_air = 1 #m #charge radius around leader
con_cu = 60e4 #60e5 #ohm-1 m-1 #conductivity of triggering object (copper)
con_air = 5e-12 #ohm-1 m-1 $conductivity of air
r_cu = 0.11e-3 #m #radius of triggering object (copper wire)
r_I = 0.1e-2 #m #radius of current channel
T_amb = 300 #K #ambient temperature
E_step = 1 #m #step length #must be multiple of space_seg_len
altitude = 3288 #m #desired altitude of triggering object base
E_stab = 400e3 #V/m
sigstr = 0.1 #S/m #streamer conductivity
thermal_rad_factor = 1.3 #thermal radius/current radius
###########################Settings##################################
ignite_negative = False #use negative leader
wait_to_start = True #wait until initiation to start timer
###########################Simulation################################
tmax = 1e-3 #maximum time
tstep = 1e-9 #time step
picno = 1000 #number of time steps to save
rod_seg_len = 1 #m #length of triggering object segment
domain_length = 1400 #m #length of domain (maximum height of leader, not including triggering object)
space_seg_len = 1 #m #length of leader segment
crit_temp = 1801 #K #temperature to enable leader flag
############################Negative Leader Multipliers#####################
rq_mult = 1 #charge radius multiplier
E_stab_mult = 3 #stability field multiplier
sigstr_mult = 1 #streamer zone conductivity multiplier
E_step_mult = 10 #step length multiplier
tau_mult = 1 #timescale multiplier
thermal_rad_mult = 1 #thermal radius multiplier

#creation of space
print("Creating Space")
s_rod = np.arange(0,rod_end,rod_seg_len)
space_end = rod_end+domain_length #m extent of vertical space
s_space = np.arange(rod_end,space_end+space_seg_len,space_seg_len)
s = np.append(s_rod,s_space)
s = np.append(-rod_seg_len,s)
si = np.zeros(len(s)-1)
for i in range(len(si)):
    si[i] = (s[i+1]+s[i])/2

#constants
print("Loading Constants")
k = (4*np.pi*8.85e-12)**-1 #N^2/Cm^2 #Coulomb's constant
if ignite_negative == True:
    rq_air = rq_mult*rq_air
rq = np.zeros(len(s_rod)+1) #m #radius of corona sheath
for i in range(len(rq)):
    rq[i] = rq_rod
rq_add = np.zeros(len(s_space))
for i in range(len(rq_add)):
    rq_add[i] = rq_air

rq = np.append(rq,rq_add)    
delta = np.exp((-altitude-rod_end)/9700)
E_stab = delta * E_stab #V/m #stability field from Lalande
if ignite_negative == True:
    E_stab = E_stab_mult*E_stab
if ignite_negative == True:
    sigstr = sigstr_mult*sigstr
#derived constants
A_cu = np.pi*r_cu**2 #m2 #cross-sectional area of copper wire
A_I = np.pi*r_I**2 #m2 #cross-sectional area of current channel
G_cu = A_cu * con_cu #S/m #conductance of copper
G_air = A_I * con_air #S/m #initial conductance of air


#Initial Conductivity
print("Initializing Arrays")
G = np.zeros(len(s_rod)+1)
for i in range(len(G)):
    G[i] = G_cu
G_add = np.zeros(len(s_space)-1)
for i in range(len(G_add)):
    G_add[i] = G_air

G = np.append(G,G_add)

#array initiation
q = np.zeros(len(s))
V = np.zeros(len(s))
E = np.zeros(len(si))
T = np.zeros(len(si))
for i in range(len(T)):
    T[i] = T_amb
V_amb = -E_amb*s
etaT = np.zeros(len(T))
sigma = np.zeros(len(T))
sigma1 = np.zeros(len(T))
for i in range(len(sigma)):
    sigma[i] = con_cu
for i in range(len(s_rod)+1,len(sigma)):
    sigma[i] = con_air
sigmalte = np.zeros(len(T))
radeps = np.zeros(len(T))
k_t = np.zeros(len(T))
Ch = np.zeros(len(T))
dT = np.zeros(len(T))
joule_heating = np.zeros(len(T))
thermal_cooling = np.zeros(len(T))
radiative_cooling = np.zeros(len(T))
vertical_conduction = np.zeros(len(T))
Eflags = np.zeros(len(T))
sigflags = np.zeros(len(T))
Eminorflags = np.zeros(len(T))
sigmajorflags = np.zeros(len(T))
Tflags = np.zeros(len(T))
I = np.zeros(len(T))
streamer_on = np.zeros(len(sigma))
leader_on = np.zeros(len(sigma))
leader_decay = np.zeros(len(sigma))

con_air_list = np.zeros(len(sigma))
for i in range(len(con_air_list)):
    con_air_list[i] = con_air

#NOTE1: The arrays in qm and sm are cut like this to 1.) remove the first 'mirror' term in each entry and 2.) remove the
#duplicate 0
#Capacitance kernel
print("Creating Kernel")
A = np.zeros((len(q[1:]),len(q[1:])))
for i in range(len(A)): #fill matrix A with the kernels
    for j in range(len(A)):
        r = abs(s[j+1] - s[i+1]) #distance between r_n and r_i
        r_image = (s[j+1] + s[i+1])
        A[i,j] = ((2*k)/(rq[j]**2))*((np.sqrt(r**2 + rq[j]**2) - r) - (np.sqrt(r_image**2 + rq[j]**2) - r_image)) #potential at r_i from charge n

#initial fill of sigmajorflags
for i in range(len(sigmajorflags)):
    if sigma[i] > sigstr:
        sigmajorflags[i] = 1
    if sigma[i] > 0.63*sigstr:
        sigflags[i] = 1

t = 0 #initial time
picstep = tmax / picno
timestop = np.linspace(0,tmax,picno+1)

#set up stepping parameters
if ignite_negative == True:
    E_step = E_step_mult*E_step
zonelength = int(E_step/space_seg_len)+1
#tau = E_step*10e-6
tau = 10e-6*E_step
tau_heating = 10e-6
if ignite_negative == True:
    tau = tau_mult*tau

#create data file
print("Creating Save File")
dateandtime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.makedirs(datapath+'/Working/Running/'+dateandtime)
txtdoc = open(datapath+'/Working/Running/'+dateandtime+'/output.txt','w+')
txtdoc.write('t (s) \t q (C) \t Vcharge (V) \t Vambient (V) \t Vtotal (V) \t E (V/m) \t sigma (S/m) \t I (A) \t s (m) \t ##### \t si (m) \t Max Field (kV/m) \t Max Charge (C) \t Current Into Ground (A) \t ##### \t ##### \t ##### \t Temp (K) \t Eflags \t sigflags \t sigmajorflags \t joule heating \t thermal cooling \t radiative cooling \t vertical conduction')
txtdoc.close()

txtdoc = open(datapath+'/Working/Running/'+dateandtime+'/settings.txt','w+')
txtdoc.write('Eamb (V/m) \t  height (m) \t space_end (m) \t dels (m) \t delt (s) \t Estep (m) \t tmax (s)')
txtdoc.write('\n{} \t {} \t {} \t {} \t {} \t {} \t {}'.format(E_amb,rod_end,space_end,space_seg_len,tstep,E_step,tmax))
txtdoc.write('\n\nri (m) \t rq (m) \t concu (S/m) \t conair (S/m) \t delta \t Tamb (K) \t rod_len (m) \t r_cu (m)')
txtdoc.write('\n{} \t {} \t {} \t {} \t {} \t {} \t {} \t {}'.format(r_I,rq[-1],con_cu,con_air,delta,T_amb,rod_seg_len,r_cu))
txtdoc.write('\n\nsigstr (S/m) \t tau (s)')
txtdoc.write('\n{} \t {}'.format(sigstr,tau))

stoppert = 0

#location to check for flags on step
E_checks = np.arange(rod_end+E_step,space_end,E_step)
for i in range(len(E_checks)):
    E_checks[i] = round(E_checks[i],3)
steps_per_E_check = int(E_step/space_seg_len)

indexlist = []
for i in range(len(s)-1):
        if np.any(E_checks == round(s[i],3)) == True:
            indexlist.append(i-1)
indexlist = np.asarray(indexlist)

wait_ticker = 0
wait_time = 1e-3
count_try = wait_time/tstep
collapse_cap = 5
collapse_counter = 0

time_grab = 0
time_cut = tmax/10
timesget = []
timescheck = []
velos = []
time_array = np.arange(0,tmax+time_cut,time_cut)

print("Beginning Simulation")
while t < tmax:
    if t > time_array[time_grab]:
        call = si[np.argmax(E)]
        timesget.append(call)
        timescheck.append(t)
        velos.append((call-rod_end)/t)
        time_grab = time_grab+1
    
    Eflags,sigflags,sigmajorflags = Eflagger(Eflags,sigflags,sigmajorflags,indexlist,E,sigma,E_stab,s_rod,steps_per_E_check,sigstr)
    mmm = (T > 10000)
    Tflags[mmm] = 1
    
    #First half step
    V,E,dI,I = chargeiterate(q,A,s,E_amb,G,tstep,I)
    q1 = q - (dI*tstep)
    etaT,radeps,k_t,Ch,sigmalte,sigma,G,joule_heating,thermal_cooling,radiative_cooling,vertical_conduction,dT,dsigma,streamer_on,leader_on,leader_decay = conductiterate(etaT,sigmalte,T,T_amb,sigma,sigstr,Eflags,sigflags,sigmajorflags,tau,tau_heating,tstep,con_air,G,s_space,s_rod,r_I,radeps,k_t,Ch,joule_heating,thermal_cooling,radiative_cooling,vertical_conduction,E,dT,space_seg_len,E_step,Tflags,streamer_on,leader_on,leader_decay,delta,E_stab,ignite_negative,crit_temp,thermal_rad_factor,thermal_rad_mult)
    T1 = T + (dT*tstep)  
    sigma1 = np.maximum(sigma-(dsigma*tstep),con_air_list)
    
    #Second half step
    V,E,dI1,I = chargeiterate(q1,A,s,E_amb,G,tstep,I)
    q = q - (0.5*tstep*(dI+dI1))
    etaT,radeps,k_t,Ch,sigmalte,sigma,G,joule_heating,thermal_cooling,radiative_cooling,vertical_conduction,dT1,dsigma1,streamer_on,leader_on,leader_decay = conductiterate(etaT,sigmalte,T1,T_amb,sigma1,sigstr,Eflags,sigflags,sigmajorflags,tau,tau_heating,tstep,con_air,G,s_space,s_rod,r_I,radeps,k_t,Ch,joule_heating,thermal_cooling,radiative_cooling,vertical_conduction,E,dT,space_seg_len,E_step,Tflags,streamer_on,leader_on,leader_decay,delta,E_stab,ignite_negative,crit_temp,thermal_rad_factor,thermal_rad_mult)
    T = T + (0.5*tstep*(dT+dT1)) 
    sigma = np.maximum(sigma-((dsigma+dsigma1)*0.5*tstep),con_air_list)
    if Tflags[-2] == 1:
        print("Leader reached end of simulation domain")
        quit()
    svoltsave = s  
    Vtotal = V_amb + V
    
    #Writing to file
    if t > timestop[stoppert]:
        #check for nan if quit
        if np.any(np.isnan(E)):
            quit()
        maxind = 0
        txtdoc = open(datapath+'/Working/Running/'+dateandtime+'/output.txt','a')
        txtdoc.write('\n{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}'.format((t),list(q),list(V),list(V_amb),list(Vtotal),list(E),list(sigma),list(I),list(s),list(svoltsave),list(si),max(E),max(q),max(E),sum(q),I[0],s[maxind],list(T),list(Eflags),list(sigflags),list(sigmajorflags),list(joule_heating),list(thermal_cooling),list(radiative_cooling),list(vertical_conduction)))
        txtdoc.close()
        print('{}%'.format(t/tmax*100))
        locmax = np.argmax(E)
        print(f'Current height = {s[locmax]} m')
        stoppert = stoppert + 1
    
    if wait_to_start == True and sum(Eflags) == 0:
        t = t
        wait_ticker = wait_ticker+1
        print(f"Waiting to initiate: ticks = {wait_ticker}/{count_try}")
        print(f'Maximum field = {max(E)/1000} kV/m')
        if wait_ticker > count_try:
            print("Ran out of attempts")
            quit()
    else:
        t = t + tstep
        wait_to_start = False
print("Done!")
if 1 in Eflags:
    print("Initiated")
else:
    print("DID NOT initiatie")
if 1 in Tflags:
    print("Made leader")
else:
    print("DID NOT make leader")