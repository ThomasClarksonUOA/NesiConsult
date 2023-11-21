# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:19:58 2023

@author: tcla272
"""

#%% Imports
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os

#%% Hamiltonian Builder
def ThreeStateEIT(params):
    da = float(params['da'])*2*np.pi
    dea = float(params['dea'])*2*np.pi
    db = float(params['db'])*2*np.pi
    deb = float(params['deb'])*2*np.pi

    ga = float(params['ga'])*2*np.pi
    gb = float(params['gb'])*2*np.pi

    ea = 0.5*float(params['ea'])*2*np.pi
    eb = 0.5*float(params['eb'])*2*np.pi
    
    kappa = float(params['kappa'])*2*np.pi
    kappb = float(params['kappb'])*2*np.pi

    gamma = float(params['gamma'])*2*np.pi
    G = float(params['G'])*2*np.pi
    O = float(params['O'])*2*np.pi
    
    Ma = int(params['Ma'])
    Mb = int(params['Mb'])
    
    a = tensor(destroy(Ma), qeye(Mb), qeye(3))
    b = tensor(qeye(Ma), destroy(Mb), qeye(3))
    
    # Create Ceasium atomic basis
    g1 = basis(3,0)
    g2 = basis(3,1)
    
    e0 = basis(3,2)
    
    # Cavity Free Energy
    HCF = (da-dea)*a.dag()*a + (db-deb)*b.dag()*b
    
    # Atomic Populations and Free Energy
    # Ground States
    pop2g = tensor(qeye(Ma),qeye(Mb),g2*g2.dag())
    pop1g = tensor(qeye(Ma),qeye(Mb),g1*g1.dag())
    pop0e = tensor(qeye(Ma),qeye(Mb),e0*e0.dag())
    
    HAFg = dea*pop1g+deb*pop2g
    
    HAF = HAFg
    
    # Mode a Interaction
    HCApartial = ga*a.dag()*(tensor(qeye(Ma),qeye(Mb),g1*e0.dag()))
    HCA = HCApartial + HCApartial.dag()
    
    # Mode b Interaction
    HCBpartial = gb*b.dag()*(tensor(qeye(Ma),qeye(Mb),g2*e0.dag()))
    HCB = HCBpartial + HCBpartial.dag()
    
    # Laser 3 Interaction
    HLA = 1j*ea*(a.dag() - a)
    
    # Laser 4 Interaction
    HLB = 1j*eb*(b.dag() - b)
    
    # Coherent ground state
    HLGpartial = O*(tensor(qeye(Ma),qeye(Mb),g1*g2.dag()))
    HLG = HLGpartial + HLGpartial.dag()
    
    # Full Hamiltonian
    H = HCF + HAF + HCA + HCB + HLA + HLB + HLG
    
    # Create Decay Operators
    c_ops_list = []
    c_ops_list.append(np.sqrt(2*kappa)*a)
    c_ops_list.append(np.sqrt(2*kappb)*b)
    c_ops_list.append(np.sqrt(G)*tensor(qeye(Ma),qeye(Mb),g1*g2.dag()))
    c_ops_list.append(np.sqrt(gamma)*tensor(qeye(Ma),qeye(Mb),g2*e0.dag()))
    c_ops_list.append(np.sqrt(gamma)*tensor(qeye(Ma),qeye(Mb),g1*e0.dag()))
    
    # Output cavity annihilation operators and atomic population operators
    operators = [a,b]
    operators.append(pop1g)
    operators.append(pop2g)
    operators.append(pop0e)
    
    # Return Hamiltonian, Collapse Operators and System Operators
    return H, c_ops_list, operators

#%% Define Single Parameter Solve
def steadyStateExpect(params):
    # Generate Hamiltonian
    H, c_ops_list, operators = ThreeStateEIT(params)
    
    # Solve for Steady State
    rhoss = steadystate(H,c_ops_list,method = 'iterative-lgmres',use_precond = True)
    
    a = operators[0]
    b = operators[1]
    
    states = np.real(expect(operators[2:],rhoss))
    modea = np.real(expect(a.dag()*a,rhoss))
    modeb = np.real(expect(b.dag()*b,rhoss))
    
    return np.array([modea,modeb,*states])

#%% Main
if __name__ == "__main__":
    # Read in parameters from config file
    configs = configparser.ConfigParser()
    path = 'C:\\Users\\tcla272\\Python\\nonclassical_light_nesi\\configs\\default.ini'
    configs.read(path)
    params = configs['DEFAULT']
    
    # Set Solver Options
    opts = Options()
    opts.store_final_state = True
    opts.store_states = False
    opts.nsteps = 10000
    
    # Build iterable
    iterstart = float(params['iterstart'])
    iterend = float(params['iterend'])
    iternum = int(params['iternum'])
    probe = np.linspace(iterstart,iterend,iternum)
    
    # Run loop
    for idx, freq in enumerate(probe):
        # Update Parameters
        params[params['key']] = str(freq)
        # Run Solver
        result = steadyStateExpect(params)
        # Save Results to Array
        if idx == 0:
            output = result
        else:
            output = np.vstack((output,result))
        # Progress Bar
        print(f'\r{idx}', end=' ',flush=True)
        
#%% Plotting
    # Update Plot Parameters
    fontsize = 30
    plt.rcParams.update({'font.size': fontsize})
        
    fig,ax = plt.subplot_mosaic("AAABBB;AAABBB;AAACCC;DDDCCC;DDDEEE;DDDEEE", figsize=(60,60))
    ax["A"].plot(probe,output[:,0],label = 'a')
    ax["D"].plot(probe,output[:,1],label = 'b')
        
    ax["B"].plot(probe,output[:,2],label = '1')
    ax["C"].plot(probe,output[:,3],label = '2')
    ax["E"].plot(probe,output[:,4],label = '3')
    
    for axs in ax:
        ax[axs].set_xlabel(r'$\Delta_a$')
        ax[axs].set_ylabel('Population')
        ax[axs].legend()
            
    ea = np.round(float(params['ea']),0)
    eb = np.round(float(params['eb']),0)

#%% Saving Results
    plt.savefig(f'ThreeLevelEIT_{ea}_{eb}.svg',format='svg',bbox_inches='tight')