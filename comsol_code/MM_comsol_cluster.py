#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:36:24 2025

@author: angelo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:40:06 2022

@author: mynames
"""

"Code to calculate the liner impedance by the mode matching method"
"This code uses the H1 estimator through the Cross Spectrum Density"
"Code to process PowerACOUSTICS format"

"Import python packages"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import csd
from scipy.signal import welch
import scipy.linalg as LA
import h5py
import scipy as sp
import matplotlib.ticker as ticker
from decimal import Decimal
from matplotlib.ticker import FormatStrFormatter
import sys  
import torch
from scipy.optimize import least_squares
import comsol_interface as comsol_interface
from scipy.optimize import dual_annealing
import warnings
from scipy.optimize import minimize
import mph
import comsol_interface
import logging
import gc
import time
import psutil
import os
import plotly.graph_objects as go
from IPython.display import display, clear_output
import multiprocessing as mp
import nlopt

warnings.filterwarnings("ignore", category=RuntimeWarning)

"File parameters"
frequencies         = np.linspace(700,2500,19)
ac_source           = 'up'
SPL                 = 145
Mach                = 0.296
version             = '14_1'
BCs                 = 'NoSlip'
resolution          = 'fine'
geometry            = 'real_geometry'
locations            = ['middle_channel']

"Impedance eduction parameter"

uniformflow = True
slip = True

"Case parameters"
Tc                  = 25                                                        # Temperature in Celsius
estimator           = 'H1'
if BCs == 'Slip':
    W                   = 0.02                                                      # duct width [m]
elif BCs == 'NoSlip':
    W                   = 0.04                                                      # duct width [m]
H                   = 0.01                                                      # duct height [m]                                                   # duct height [m]

"Liner parameters - NASA"
n_cavities          = 11                                                        # Number of liner cavities
POA                 = 6.3/100                                                   # Percentage of Open Area
cvt_height          = 38.1e-3                                                 # Cavity Height (m)
fsheet_thick        = 0.635e-3                                                  # Face Sheet thickness (m)
orifice_d           = 0.9906e-3                                                 # Orifice diameter (m)
orifice_d_min       = 1.05e-3
orifice_d_max       = 1.26e-3      
linerLength         = 136.906e-3                                                # Liner length in meters [m]


"Liner parameters - UFSC"
n_cavities_ufsc              = 11                                                        # Number of liner cavities
POA_ufsc                     = 8.75/100                                                   # Percentage of Open Area
cvt_height_ufsc              = 38.1e-3                                                   # Cavity Height (m)
fsheet_thick_ufsc            = 0.55e-3                                                  # Face Sheet thickness (m)
orifice_d_ufsc               = 1.169250e-3                                                 # Orifice diameter (m)
linerLength_ufsc             = 136.906e-3 
                                             # Liner length in meters [m]

"Flow parameters NASA V14"
MeanMach            = 0.293                                                     # Mean Mach Number
BLDT                = 1.338e-3                                                  # Turbulent Boundary Layer Displacement Thickness

if Mach == 0:
    BLDT = 0
    MeanMach = 0

"Fluid parameters"
Pamb                = 101325                                                    # Ambient Pressure (Pa)
sutherland_tref     = 273.15                                                    # Sutherland Ref. Temperature  
Tk                  = Tc + sutherland_tref                                      # Temperature (Kelvin)
gamma               = 1.4                                                       # Heat Capacity Ratio (dry air)
Pr                  = 0.707                                                     # Prandtl
Runiv               = 8.314462                                                  # Ideal Gas Constant [J/K.mol]
mol_weight          = 28.9645/1000                                              # Mol. Weight of Air
R                   = Runiv/mol_weight                                          # Specific Gas Constant
c0                  = np.sqrt(gamma*R*Tk)                                       # Sound Speed
rho                 = Pamb/(R*Tk)                                              # Density
nu                  = ( 1.458e-6*(Tk**1.5) / (110.4 + Tk) )/rho                 # Viscosity
cm                  = 1/2.54                                                    # Variable to plots


#%% =============================================================================
# BEGIN FUNCTIONS DEFINITION
# =============================================================================
logging.basicConfig(filename='comsol_optimization.log', level=logging.INFO)

def log_event(msg):
    logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def cleanup():
    gc.collect()
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)
    log_event(f"Memoria utilizzata: {mem:.2f} MB")

def profiled(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        log_event(f"{func.__name__} completata in {time.time()-start:.2f}s")
        return result
    return wrapper

def fungrad(fun, x, args):
    h = 1e-8  # Passo per la derivata numerica
    return (fun(x + h, args) - fun(x, args)) / h

   
def format_func(value, tick_number):
    # Formatta il valore con due cifre significative
    return f'{value:.2g}'

def computeFluidProperties(Tk):
        rho                 = Pamb/(R*Tk)
        nu                  = ( 1.458e-6*(Tk**1.5) / (110.4 + Tk) )/rho
        return rho, nu
  
def computeImpedanceGuess(testFrequency, SPL, rho, nu, c0, M, BLDT):
    
    def impedanceUTAS(rho, nu, c0, POA, L, t, d, SPL, freq, M, BLthick):
        r = d/2
        omega = 2*np.pi*freq
        k = omega/c0
        pt = 2e-5*10**(SPL/20)
        epsilon = (1 - 0.7*np.sqrt(POA))/(1 + 305*M**3)
        Sm = -0.0000207*k/POA**2
        Cd = 0.80695*np.sqrt(POA**(0.1)/np.exp(-0.5072*t/d))
        Ks = np.sqrt(-1j*omega/nu)
        F = 1 - 2*sp.special.jv(1,Ks*r)/(Ks*r*sp.special.jv(0,Ks*r))
        Zof = 1j*omega*(t + epsilon*d)/(c0*POA)/F
        Rcm = M/(POA*(2 + 1.256*BLthick/d))
        Sr = 1.336541*(1 - POA**2)/(2*c0*Cd**2*POA**2)
        
        Ra = 1;
        Xa = 1;
        Vp0 = pt/(rho*c0*np.sqrt(Xa**2+Ra**2))
        
        fun = lambda x: x - pt/(rho*c0*np.abs(Zof + Sr*x + Rcm + 1j*(Sm*x - (1/np.tan(k*L)))))
        Vp = sp.optimize.fsolve(fun,Vp0)
        Z = Zof + Sr*Vp + Rcm + 1j*(Sm*Vp - (1/np.tan(k*L)))
        
        Ra = np.real(Z)
        Xa = np.imag(Z)
        return Ra, Xa
        
    ZinitialReal = np.zeros(np.shape(testFrequency))
    ZinitialImag = np.zeros(np.shape(testFrequency))
    
    for i in range(np.size(testFrequency)):
        ZinitialReal[i], ZinitialImag[i] = impedanceUTAS(rho,nu,c0,parameters[0],parameters[1],parameters[2],parameters[3],SPL,testFrequency[i],M,BLDT)
    Zinitial  = ZinitialReal + 1j*ZinitialImag
    return Zinitial

def impedanceUTAS(rho, nu, c, POA, L, t, d, SPL, freq, M, BLthick):
    r = d/2
    omega = 2*np.pi*freq
    k = omega/c0
    pt = 2e-5*10**(SPL/20)
    epsilon = (1 - 0.7*np.sqrt(POA))/(1 + 305*M**3)
    Sm = -0.0000207*k/POA**2
    Cd = 0.80695*np.sqrt(POA**(0.1)/np.exp(-0.5072*t/d))
    Ks = np.sqrt(-1j*omega/nu)
    F = 1 - 2*sp.special.jv(1,Ks*r)/(Ks*r*sp.special.jv(0,Ks*r))
    Zof = 1j*omega*(t + epsilon*d)/(c0*POA)/F
    Rcm = M/(POA*(2 + 1.256*BLthick/d))
    Sr = 1.336541*(1 - POA**2)/(2*c0*Cd**2*POA**2)
    
    Ra = 1;
    Xa = 1;
    Vp0 = pt/(rho*c0*np.sqrt(Xa**2+Ra**2))
    
    fun = lambda x: x - pt/(rho*c0*np.abs(Zof + Sr*x + Rcm + 1j*(Sm*x - (1/np.tan(k*L)))))
    Vp = sp.optimize.fsolve(fun,Vp0)
    Z = Zof + Sr*Vp + Rcm + 1j*(Sm*Vp - (1/np.tan(k*L)))
    
    Ra = np.real(Z)
    Xa = np.imag(Z)
    return Ra, Xa

    
def costFunctionEduction(x,args,weight_real = 1):
    global iteration_counter
    iteration_counter += 1
    k,pressure,M,K0,freq,comsol = args
    print(f"{x}",flush=True)
    Z1 = x[0] + 1j*x[1]
    Z2 = np.inf
    zeta1 = 0
    zeta2 = 0
    simPressure = comsol.lnsf_noslip(freq, Z1)[0]
    
    cost = np.zeros(int(len(pressure)))  # meglio usare len(pressure)
    tf_real = np.zeros(int(len(pressure)))
    tf_phase_real = np.zeros(int(len(pressure)))
    tf_sim = np.zeros(int(len(pressure)))
    tf_phase_sim = np.zeros(int(len(pressure)))
    t = np.linspace(0, 1, 100)
    count = 0
  
    for i in range(len(pressure)):
        if i == 0:
            if ac_source == 'up' or ac_source == 'noflow':
                amp_real_ground = np.real(pressure[i])
                sim_ground = simPressure[0]
            elif ac_source == 'down':
                amp_real_ground = np.real(pressure[i])
                sim_ground = simPressure[-1]
    
        amp_real = np.abs(pressure[i])
        phase_real = 1 / (np.tan(np.imag(pressure[i]) / np.real(pressure[i])) + 1e-5)
        tf_real[i] = amp_real
        tf_phase_real[i] = phase_real
    
        simPressure[i] = simPressure[i] / sim_ground
        amp_sim = np.abs(simPressure[i])
        phase_sim = 1 / (np.tan(np.imag(simPressure[i]) / np.real(simPressure[i])) + 1e-5)
        tf_sim[i] = amp_sim
        tf_phase_sim[i] = phase_sim
    
    
    resid_real = np.real(pressure) - np.real(simPressure)
    resid_imag = np.imag(pressure) - np.imag(simPressure)
    resid = np.concatenate([weight_real*resid_real, resid_imag])
    logging.info(f"{x} | Somma quadratica media errori: {np.sum(resid**2):.6f}")
    print(f"\n Somma quadratica media dei residui: \n {np.sum(resid**2)}\n",flush=True)
    Mics = np.linspace(1, len(pressure), len(pressure))
    
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.scatter(np.linspace(0, len(pressure), len(pressure)), tf_real, color='k', label='Filtered signal')
    ax.scatter(np.linspace(0, len(pressure), len(pressure)), tf_sim, color='r', label='Analytical one')
    
    ax.set_xlabel('Microphones', fontsize=40)
    ax.set_ylabel(r'$|H_f|$', fontsize=40)
    ax.legend(loc='lower left', fontsize=25, ncols=1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=30)
    #ax.set_xticks([0, 25, 50, 150, 175, 200])
    #ax.set_xticklabels(['0', '10', '21', '22', '32', '43'])
    #ax.set_xlim([0, 50])
    
    # Salvataggio grafico
    output_dir = "./optimization_plots_subplex"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"TF_mic_optimization_freq_{int(freq)}_iter_{iteration_counter}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)  # chiude la figura per liberare memoria
    
    return resid
    
def costFunctionEduction_scalar(x, args):
    resid = costFunctionEduction(x, args)
    return np.sum(resid**2)

def Fields_differences_view(x,sim_data,args):
    k,pressure,M,K0,freq,client = args
    Z1 = x[0] + 1j*x[1]
    Z2 = np.inf
    zeta1 = 0
    zeta2 = 0
    comsol = comsol_interface.Comsol_lunch()
    simPressure = comsol.lpff(freq, Z1, client)[0]    
    cost = np.zeros(int(nMics))
    count = 0
    t = np.linspace(0, 1,200)
    tf_real = np.zeros(int(nMics))
    tf_phase_real = np.copy(tf_real)
    tf_sim = np.zeros(int(nMics))
    tf_sim_real = np.copy(tf_sim)
    for i in range(nMics):
        
        if i == 0:
            if ac_source == 'up' or ac_source =='noflow':
                amp_real_ground = np.real(pressure[i])
                sim_ground = simPressure[0]
            elif ac_source == 'down':
                amp_real_ground = np.real(pressure[i])
                sim_ground = simPressure[-1]
            
        amp_real = np.sqrt(np.real(pressure[i])**2+np.imag(pressure[i])**2)
        phase_real = np.tan(np.imag(pressure[i])/np.real(pressure[i]))**-1
        tf_real[i] = amp_real
        tf_phase_real[i]=phase_real 
        simPressure[i] = simPressure[i]/sim_ground
        
        amp_sim = np.sqrt(np.real(simPressure[i])**2+np.imag(simPressure[i])**2)
        phase_sim = np.tan(np.imag(simPressure[i])/np.real(simPressure[i]))**-1
        tf_sim[i] = amp_sim
        tf_sim_real[i] = phase_sim
        pressure_plot = amp_real * np.exp(1j * (2 * np.pi * 11 * t + phase_real)) 
        simPressure_plot = amp_sim * np.exp(1j * (2 * np.pi * 11 * t + phase_sim)) 
        
        error = (pressure[i] - simPressure[i])/pressure[i]*100
        cost[count] = np.abs(np.real(error))
        count += 1   
    Mics = np.linspace(1,nMics,nMics)

    background_path = '/home/angelo/Scrivania/image_liner_2.png'
    background_img = mpimg.imread(background_path)
    # Creazione del plot
    fig, ax = plt.subplots(figsize=(15, 9))
    
    # Imposta lo sfondo con l'immagine
    ax.imshow(background_img, aspect='auto', extent=[0, len(tf_real) * 12.2, 0, 2], alpha=0.3)
    
    # Prima parte del plot
    ax.plot(np.linspace(0,52,8), tf_real[:8], color='k',linewidth = 2,label = 'Measured signal')
    ax.plot(np.linspace(0,52,8), tf_sim[:8], color='r',linewidth = 4,alpha = 0.4, label = 'Eduction process')

    
    # Determina lo shift per la seconda parte del plot (ad esempio spostiamo di 25 unità)
    shift = 150
    
    # Seconda parte del plot (shiftata)
    ax.plot(np.linspace(shift,198,8), tf_real[8:], color='k',linewidth = 2)
    ax.plot(np.linspace(shift,198,8), tf_sim[8:], color='r',linewidth = 4,alpha = 0.4)

    
    # Configura il grafico per la seconda parte
    ax.set_xlabel('Microphones', fontsize=40)
    plt.ylabel(r'$|H_f|$', fontsize=40)
    ax.legend(loc= 'lower left', fontsize=40, ncols = 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=30)
    # Correggi i tick personalizzati
    ax.set_xticks([0, 25, 50, 150, 175, 200])
    ax.set_xticklabels(['0', '10', '21', '22','32','43'])
    ax.set_xlim([0,198])
    #plt.tight_layout()
    plt.savefig(f'/./TF_mic_{i}_SPL_{SPL}_{ac_source}_frequency_{frequencies[0]}_Mach_{Mach}.png', dpi = 300, bbox_inches = 'tight')
    # plt.show()
    return tf_real,tf_sim,cost


# =============================================================================
# END OF FUNCTIONS DEFINITION
# =============================================================================
    

#############################PATH DEFINITION###################################
#%%
## =============================================================================
# BEGIN CODE
# =============================================================================
if Mach==0.32 or Mach==0.3:
    num_Mach=0.3
elif Mach==0:
    num_Mach=0

nFreq           = len(frequencies)

#%% Main code
# NOT NECESSARY TO CHANGE
tf_real = np.ones((44, 3))
tf_sim= np.copy(tf_real)
cost = np.copy(tf_sim)
impedance_ew =[]
for k in range(len(locations)):
    
    location = locations[k]

# %%
    path_mics=      './'

    mic_positions   = np.loadtxt(path_mics + f'/eduction_points_lined_sec.txt',skiprows=0)
    nMics           = np.shape(mic_positions)[0]                                # Number of microphones
    
    complex_amp     = np.zeros((nFreq,nMics),dtype=complex)
    mic_SPLs        = np.zeros((nFreq,nMics))
    impedance       = np.zeros(nFreq,dtype=complex)
    
    
    data = np.loadtxt(
            './data_x_eduction_lined.txt',
            dtype=complex,
            comments='%',  # Ignora le righe che iniziano con '%'
            skiprows=0     # Non serve saltare righe fisse, viene ignorato ciò che inizia con '%'
    )
    
    impedance = np.zeros(nFreq,dtype = complex)
    totalCost = np.zeros(nFreq)
    frequencies = np.asarray(frequencies)
    MachVector = np.asarray([MeanMach]*nFreq)
    Mach_flow = 1*MachVector
    M = np.mean(Mach_flow)
    
    start_total = time.time()
    log_event("Inizio ottimizzazione impedenza.")
    
    parameters = [POA, cvt_height, fsheet_thick, orifice_d]
    Zinitial = computeImpedanceGuess(frequencies, SPL, rho, nu, c0, M, BLDT)
    client = mph.start(cores=8)
    client.caching(True)
    comsol = comsol_interface.Comsol_lunch(client,uniformflow,slip)
    log_event("Client COMSOL avviato.")
    
    
    for f in range(nFreq):
        start_freq = time.time()
        print(f'frequenza:{frequencies[f]}')
        logging.info(f"frequenza:{frequencies[f]}")
        sim_data            = data[:, f + 3]
        
        for mic in np.arange(nMics):
                             
            if estimator == 'H1':
                
                if ac_source == 'up' or ac_source == 'no_flow':
                    H1 = sim_data[mic]/sim_data[0]
                elif ac_source == 'down':
                    H1 = sim_data[mic]/sim_data[-1]


                complex_amp[f,mic] = H1                     

                              
        BC = 'myers1980'     

        rho, nu = computeFluidProperties(Tk)
        r = H*W/(H+W)       # duct hydraulic radius
        a = W/2             # radius of the duct
              
        if M == 0:
            BLDT = 0
        
        measurements = np.zeros((nMics),dtype=complex)
        
        measurements[:] = complex_amp[f]
        
        nMicsEachSide = int(nMics/2)
        
        testFrequency = frequencies
        omega = 2*np.pi*testFrequency
        k0 = omega*a/c0
        
        Sh = r*np.sqrt(omega/nu)
        K0 = 1 + ((1 - 1j)/(Sh*np.sqrt(2)))*(1 + (gamma - 1)/np.sqrt(Pr))
        kplus = k0*K0/(1 + K0*Mach_flow)
        kminus = -k0*K0/(1 - K0*Mach_flow)


        initialZ = np.zeros(2)
          
        initialZ[0] = np.real(Zinitial[f])
        initialZ[1] = np.imag(Zinitial[f])
       
        iteration_counter = 0

        args = k0[f],measurements,Mach_flow[0],K0[f],frequencies[f],comsol
        
        # --- BOUND ---
        lower_bounds = [0.2, -4]
        upper_bounds = [3, 3]

	# --- OTTIMIZZAZIONE GLOBALE ---
        opt_global = nlopt.opt(nlopt.GN_DIRECT_L, 2)
        opt_global.set_lower_bounds(lower_bounds)
        opt_global.set_upper_bounds(upper_bounds)
        opt_global.set_min_objective(lambda x, grad: costFunctionEduction_scalar(x, args))
        opt_global.set_maxeval(100)
        opt_global.set_xtol_rel(1e-4)    # Tolleranza meno severa

        x_global = opt_global.optimize(initialZ)
        f_global = opt_global.last_optimum_value()

        print(f"\n Fase globale: Z = {x_global[0]:.4f} + j{x_global[1]:.4f}, costo = {f_global:.2e}")
        center = x_global
        eps = 0.05  # 5%
        local_lower_bounds = [max(lower_bounds[0], center[0] - eps*abs(center[0])), max(lower_bounds[1], center[1] - eps*abs(center[1]))]
        local_upper_bounds = [min(upper_bounds[0], center[0] + eps*abs(center[0])), min(upper_bounds[1], center[1] + eps*abs(center[1]))]
        print(f"\n Fase locale:\n lower bounds = {local_lower_bounds[0]:.4f}\n Global bounds:{local_upper_bounds[0]:.4f}")
        # Usa local_lower_bounds e local_upper_bounds in SUBPLEX

        # --- OTTIMIZZAZIONE LOCALE SUBPLEX ---
        opt_local = nlopt.opt(nlopt.LN_SBPLX, 2)
        opt_local.set_lower_bounds(local_lower_bounds)
        opt_local.set_upper_bounds(local_upper_bounds)
        opt_local.set_min_objective(lambda x, grad: costFunctionEduction_scalar(x, args))
        opt_local.set_xtol_rel(1e-6)
        opt_local.set_ftol_rel(1e-6)
        opt_local.set_maxeval(100)

        x_local = opt_local.optimize(x_global)  # Parti dalla soluzione globale
        f_local = opt_local.last_optimum_value()

        print(f"\n✅ Fase locale: Z = {x_local[0]:.4f} + j{x_local[1]:.4f}, costo = {f_local:.2e}")

        impedance[f] = x_local[0] + 1j*x_local[1]
        log_event(f"Frequenza {frequencies[f]} completata in {time.time() - start_freq:.2f} secondi Impedenza {x_local[0]:.4f} + j{x_local[1]:.4f}, costo = {f_local:.2e}")

    # =============================================================================
        plot=0
    # =============================================================================
        # =============================================================================
        # Plots and prints
        # =============================================================================
        np.set_printoptions(formatter={'float': lambda x: "{:.1f}".format(x)})
        print('Microphone SPLs: {}'.format(mic_SPLs[0:,:]))
        
        np.set_printoptions(formatter={'float': lambda x: "{:.2f}".format(x)})
        realZ = np.real(impedance)
        imagZ = np.imag(impedance)
        Z = realZ + imagZ*1j
        
        print('Z: {}'.format(np.around(Z,2)))

with open('impedance.txt', 'w') as file:
    for z in impedance:
        file.write(f"{z.real:.2f} + {z.imag:.2f}j\n")

log_event(f"Ottimizzazione completata in {time.time() - start_total:.2f} secondi")
