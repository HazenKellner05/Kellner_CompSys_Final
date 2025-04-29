# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:19:51 2025

@author: hazen
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


# gLV set up matrix form, modified from lecture to represent inhibatory network
def gLV_dynamics(t, y, n, mu, alpha, k):
    dydt = np.zeros(n)
    
    dydt = y*(mu - k * np.matmul(alpha, y))
    
    return dydt

# basic set up
N = 3
K = 1
I = 1
MU = [I, I, I]

t = np.linspace(0,1000, 1000)
tspan = [t[0], t[-1]]

y0 = [0.6, 0.6, 0.1]


'''
# conditions from dynamics of inhibatory network (Lagzi, 2019)

Three conditions (in order of index):
    a + b < 2, stable equilibrium
    a + b > 2 & a, b > 1, WTA
        a or b < 1, Osicllatory, no equilibrium
'''

a = [1.4, 1.4, 1.4]
b = [0.3, 1.1, 0.9]

for i in range(len(a)):
    # coffs. matrix
    ALPHA = np.array([[1, a[i], b[i]],
                     [b[i], 1, a[i]], 
                     [a[i], b[i], 1]])

    # numerical integration
    ode_sol = sc.integrate.solve_ivp(gLV_dynamics, tspan, y0, t_eval=t,
                                         args=(N, MU, ALPHA, K))
    
    # state labeling
    if (a[i] + b[i]) < 2:
        state = 'Stable Equilibrium'
    elif a[i] > 1 and b[i] > 1:
        state = 'WTA'
    else:
        state = 'Oscillatory'

    # plot dynamics
    [fig,axs] = plt.subplots(1,1,figsize=[12,4])

    axs.plot(ode_sol.t,ode_sol.y[0],'k-')
    axs.plot(ode_sol.t,ode_sol.y[1],'r-')
    axs.plot(ode_sol.t,ode_sol.y[2],'b-')
    
    # plot h-line if at equil point if exhbited 
    if state == 'Stable Equilibrium':
        equil_activation = 1 / (1 + a[i] + b[i])
        axs.axhline(y = equil_activation, color = 'cyan', linestyle = '--',
                    label = 'Equilibrium activation')
        
    axs.set_xlabel('time [s]')
    axs.set_ylabel('firing rate (spike / ms)')
    axs.legend(['network 1','network 2', 'network 3'])
    plt.title(f"Dynamics for {state} Network \n a = {a[i]}, b = {b[i]}")

    try:
        plt.savefig(os.path.join("plots", f"Network_Dynamics_{state}.png"))
    except FileNotFoundError:
        os.mkdir("plots")
        plt.savefig(os.path.join("plots", f"Network_Dynamics_{state}.png"))
    
    # Parameter space graph
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')

    ax.plot3D(ode_sol.y[0], ode_sol.y[1], ode_sol.y[2], c='blue')
    ax.scatter(y0[0], y0[1], y0[2], c='red')
    
    # Plot plane of solution if oscillatory
    if state == 'Oscillatory':
        xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        zz = 1 - yy - xx
        ax.plot_surface(xx, yy, zz, alpha=0.5)

    ax.set_xlabel('network 1')
    ax.set_ylabel('network 2')
    ax.set_zlabel('network 3')
    plt.title(f"Parameter Space {state} Network \n a = {a[i]}, b = {b[i]}")
        
    try:
        plt.savefig(os.path.join("plots", f"Network_Parameter_Space_{state}.png"))
    except FileNotFoundError:
        os.mkdir("plots")
        plt.savefig(os.path.join("plots", f"Network_Dynamics_{state}.png"))
    