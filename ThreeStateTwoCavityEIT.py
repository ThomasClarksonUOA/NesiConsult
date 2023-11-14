# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:19:58 2023

@author: tcla272
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

def ThreeStateEIT(params):
    da = params['da']
    dea = params['dea']
    db = params['db']
    deb = params['deb']

    ga = params['ga']
    gb = params['gb']

    ea = 0.5*params['ea']
    eb = 0.5*params['eb']
    
    kappa = params['kappa']
    kappb = params['kappb']

    gamma = params['gamma']
    G = params['G']
    O = params['O']
    
    Ma = params['Ma']
    Mb = params['Mb']
    
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
    # c_ops_list.append(np.sqrt(gamma)*tensor(qeye(Ma),qeye(Mb),g2*e0.dag()))
    # c_ops_list.append(np.sqrt(gamma)*tensor(qeye(Ma),qeye(Mb),g1*e0.dag()))
    
    # Output cavity annihilation operators and atomic population operators
    operators = [a,b]
    operators.append(pop1g)
    operators.append(pop2g)
    operators.append(pop0e)
    
    return H, c_ops_list, operators


probe = np.linspace(-15,15,151)
transmission = np.zeros(len(probe))
reflection = np.zeros(len(probe))
states = np.zeros((3,len(probe)))

opts = Options()
opts.store_final_state = True
opts.store_states = False
opts.nsteps = 10000

for idx, freq in enumerate(probe):
    params = {
        'da' : 0*2*np.pi,
        'dea' : freq*2*np.pi,
        'db' : 0*2*np.pi,
        'deb' : 0*2*np.pi,
    
        'ga' : 8*2*np.pi,
        'gb' : 8*2*np.pi, #6.8 donald
    
        'ea' : 2*2*np.pi,
        'eb' : 0*2*np.pi,
    
        'kappa' : 1*2*np.pi, #15*2*np.pi, #5.1  
        'kappb' : 1*2*np.pi, #15*2*np.pi, #3.4  
            
        'gamma' : 0.1*2*np.pi, #5.234
        'G' : 0.5*2*np.pi,
        'O' : 0*2*np.pi,
            
        # Number of atoms and cavity basis. M>=3 for photon statistics
        'N' : 1,
        'Ma' : 8,
        'Mb' : 8,
        }
    
    H, c_ops_list, operators = ThreeStateEIT(params)
    rhoss = steadystate(H,c_ops_list,method = 'iterative-lgmres',use_precond = True)
    
    a = operators[0]
    b = operators[1]
    
    states[:,idx] = np.real(expect(operators[2:],rhoss))
    
    x = expect(a,rhoss)
    y = params['kappa']*x/(params['ea'])
    z = y-1
    
    transmission[idx] = np.real(expect(a.dag()*a,rhoss))#np.real(y*y.conjugate()) #
    reflection[idx] = np.real(z*z.conjugate()) #np.real(expect(b.dag()*b,rhoss))#
    print(f'\r{idx}', end=' ',flush=True)
    
#%%    
fontsize = 30
plt.rcParams.update({'font.size': fontsize})
    
fig,ax = plt.subplot_mosaic("AABB;AACC;DDEE;DD..", figsize=(60,60))
ax["A"].plot(probe,transmission,label = 'a')
ax["D"].plot(probe,reflection,label = 'b')
    
ax["B"].plot(probe,states[0,:],label = '1')
ax["C"].plot(probe,states[1,:],label = '2')
ax["E"].plot(probe,states[2,:],label = '3')
        
        
for axs in ax:
    ax[axs].set_xlabel(r'$\Delta_a$')
    ax[axs].set_ylabel('Population')
    ax[axs].legend()
        
ea = np.round(params["ea"],0)
eb = np.round(params["eb"],0)
plt.savefig(f'ThreeLevelEIT_{ea}_{eb}.svg',format='svg',bbox_inches='tight')
    


