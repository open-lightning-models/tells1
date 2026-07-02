# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:49:47 2026

@author: johnp
"""

from numpy import loadtxt
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def convtolist(q_pre):
    q = []
    for j in range(len(q_pre)):
        qnew = q_pre[j].split(sep=',')
        for i in range(len(qnew)):
            qnew[i] = qnew[i].replace('[','')
            qnew[i] = qnew[i].replace(']','')
            qnew[i] = float(qnew[i])
        q.append(qnew)
    q = np.asarray(q)
    return q

def grabvariables(lastfile):
    #grab all variables
    t = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = float, skiprows=1, usecols=0, max_rows=None)
    q_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|', dtype = list, skiprows=1, usecols=1, max_rows=None)
    V_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=2, max_rows=None)
    Vambient_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=3, max_rows=None)
    Vtotal_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=4, max_rows=None)
    E_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=5, max_rows=None)
    G_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=6, max_rows=None)
    I_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=7, max_rows=None)
    s_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=8, max_rows=None)
    si_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=10, max_rows=None)
    temps_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=17, max_rows=None)
    Eflags_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=18, max_rows=None)
    joule_heating_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=21, max_rows=None)
    thermal_cooling_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=22, max_rows=None)
    radiative_cooling_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=23, max_rows=None)
    vertical_conduction_pre = loadtxt(lastfile+'/output.txt', comments = None, delimiter = '|',dtype = list, skiprows=1, usecols=24, max_rows=None)
    return t,q_pre,V_pre,Vambient_pre,Vtotal_pre,E_pre,G_pre,I_pre,s_pre,si_pre,temps_pre,Eflags_pre,joule_heating_pre,thermal_cooling_pre,radiative_cooling_pre,vertical_conduction_pre

def grabsettings(lastfile):
    Eamb = loadtxt(lastfile+'/settings.txt', comments = None, delimiter = '\t',dtype = float, skiprows=1, usecols=0, max_rows=1)
    wireheight = loadtxt(lastfile+'/settings.txt', comments = None, delimiter = '\t',dtype = float, skiprows=1, usecols=1, max_rows=1)
    space_end = loadtxt(lastfile+'/settings.txt', comments = None, delimiter = '\t',dtype = float, skiprows=1, usecols=2, max_rows=1)
    space_seg_len = loadtxt(lastfile+'/settings.txt', comments = None, delimiter = '\t',dtype = float, skiprows=1, usecols=3, max_rows=1)
    tstep = loadtxt(lastfile+'/settings.txt', comments = None, delimiter = '\t',dtype = float, skiprows=1, usecols=4, max_rows=1)
    Estep = loadtxt(lastfile+'/settings.txt', comments = None, delimiter = '\t',dtype = float, skiprows=1, usecols=5, max_rows=1)
    return Eamb,wireheight,space_end,space_seg_len,tstep,Estep

datapath = #####################your data location here#######################################


print("Loading data...")
t,q_pre,V_pre,Vambient_pre,Vtotal_pre,E_pre,G_pre,I_pre,s_pre,si_pre,temps_pre,Eflags_pre,joule_heating_pre,thermal_cooling_pre,radiative_cooling_pre,vertical_conduction_pre = grabvariables(lastfile1)
Eamb,wireheight,space_end,space_seg_len,tstep,Estep = grabsettings(lastfile1)

print("Converting data...")

q = convtolist(q_pre)
s = convtolist(s_pre)
V = convtolist(Vtotal_pre)
E = convtolist(E_pre)
si = convtolist(si_pre)
I = convtolist(I_pre)
sigma = convtolist(G_pre)
T = convtolist(temps_pre)
Vambient = convtolist(Vambient_pre)
Eflags = convtolist(Eflags_pre)
joule_heating = convtolist(joule_heating_pre)
thermal_cooling = convtolist(thermal_cooling_pre)
radiative_cooling = convtolist(radiative_cooling_pre)
vertical_conduction = convtolist(vertical_conduction_pre)

####################control figure parameters
fig, axs = plt.subplots(2,3,figsize=(50,25))
plt.rcParams['axes.linewidth'] = 10
labelfont = 60
linewidthfont = 4
ticklabelfont = 30
tickwidth = 5
ticklength = 40


rod_point = wireheight
y_bottom = rod_point-0.1*rod_point
y_top = space_end

qline, = axs[0,0].plot(q[0]*1e6,s[0],linewidth=linewidthfont)
axs[0,0].set_xlabel("Charge density (\u00B5C/m)",fontsize = labelfont)
axs[0,0].set_ylabel("Height (m)",fontsize = labelfont)
axs[0,0].set_ylim(y_bottom,y_top)
axs[0,0].minorticks_on()
axs[0,0].tick_params(axis='both', which='major', top = True, right = True, labelsize=ticklabelfont, width=tickwidth, length=ticklength)
axs[0,0].tick_params(axis='both', which='minor', top = True, right = True, labelsize=0.5*ticklabelfont, width=0.5*tickwidth, length=0.5*ticklength)
looptext = axs[0,0].text(0.6,0.7, 'TESTTEXT', transform=axs[0,0].transAxes,fontsize = 1.1 * labelfont)

Vline, = axs[0,1].plot(V[0]/1e6,s[0],linewidth=linewidthfont)
axs[0,1].plot(Vambient[0]/1e6,s[0],linewidth=linewidthfont,label = 'Ambient potential',color='green',linestyle='--')
axs[0,1].set_xlabel("Potential (MV)",fontsize = labelfont)
axs[0,1].set_ylim(y_bottom,y_top)
axs[0,1].legend(loc='best',fontsize=0.6*labelfont)
axs[0,1].minorticks_on()
axs[0,1].tick_params(axis='both', which='major', top = True, right = True, labelsize=ticklabelfont, width=tickwidth, length=ticklength)
axs[0,1].tick_params(axis='both', which='minor', top = True, right = True, labelsize=0.5*ticklabelfont, width=0.5*tickwidth, length=0.5*ticklength)

Eline, = axs[0,2].plot(E[0]/1000,si[0],linewidth=linewidthfont)
axs[0,2].set_xlabel("Electric field (kV/m)",fontsize = labelfont)
axs[0,2].set_ylabel("Height (m)",fontsize = labelfont)
axs[0,2].set_ylim(y_bottom,y_top)
axs[0,2].legend(loc='best',fontsize=0.6*labelfont)
axs[0,2].minorticks_on()
axs[0,2].tick_params(axis='both', which='major', top = True, right = True, labelsize=ticklabelfont, width=tickwidth, length=ticklength)
axs[0,2].tick_params(axis='both', which='minor', top = True, right = True, labelsize=0.5*ticklabelfont, width=0.5*tickwidth, length=0.5*ticklength)

Iline, = axs[1,0].plot(I[0],si[0],linewidth=linewidthfont)
axs[1,0].set_xlabel("Current (A)",fontsize = labelfont)
axs[1,0].set_ylim(y_bottom,y_top)
axs[1,0].minorticks_on()
axs[1,0].tick_params(axis='both', which='major', top = True, right = True, labelsize=ticklabelfont, width=tickwidth, length=ticklength)
axs[1,0].tick_params(axis='both', which='minor', top = True, right = True, labelsize=0.5*ticklabelfont, width=0.5*tickwidth, length=0.5*ticklength)

sigmaline, = axs[1,1].semilogx(sigma[0],si[0],linewidth=linewidthfont)
axs[1,1].set_xlabel("Conductivity (S/m)",fontsize = labelfont)
axs[1,1].set_ylabel("Height (m)",fontsize = labelfont)
axs[1,1].set_ylim(y_bottom,y_top)
axs[1,1].minorticks_on()
axs[1,1].tick_params(axis='both', which='major', top = True, right = True, labelsize=ticklabelfont, width=tickwidth, length=ticklength)
axs[1,1].tick_params(axis='both', which='minor', top = True, right = True, labelsize=0.5*ticklabelfont, width=0.5*tickwidth, length=0.5*ticklength)


Tline, = axs[1,2].plot(T[0]/1000,si[0],linewidth=linewidthfont)
axs[1,2].set_xlabel("Temperature (kK)",fontsize = labelfont)
axs[1,2].set_ylim(y_bottom,y_top)
axs[1,2].minorticks_on()
axs[1,2].tick_params(axis='both', which='major', top = True, right = True, labelsize=ticklabelfont, width=tickwidth, length=ticklength)
axs[1,2].tick_params(axis='both', which='minor', top = True, right = True, labelsize=0.5*ticklabelfont, width=0.5*tickwidth, length=0.5*ticklength)


plt.tight_layout()

# Animation function
def animate(i):
    print(i)
    qline.set_xdata(q[i]*1e6)
    Vline.set_xdata(V[i]/1e6)
    Eline.set_xdata(E[i]/1000)
    Iline.set_xdata(I[i])
    sigmaline.set_xdata(sigma[i])
    Tline.set_xdata(T[i]/1000)
    looptext.set_text('{} ms'.format(round(t[i]*1000,3)))
    return qline,Vline,Eline,Tline,sigmaline,Tline

# Create the animation object
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=200, blit=True)

# Save the animation as an MP4 video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=500)
ani.save(datapath+'/animation.mp4', writer=writer, dpi = 72)