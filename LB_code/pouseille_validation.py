#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 19:23:24 2026

@author: angelo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Script: Poiseuille Flow (Convergence & Tau dependence)
Utilizza il kernel lbm_step_modular e Half-Way Bounce-Back.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

# Importa il tuo kernel
from lbm_kernel import lbm_step_modular

# =============================================================================
# COSTANTI LBM D2Q9
# =============================================================================
q = 9
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]], dtype=np.int64)
cx_i = c[:, 0]
cy_i = c[:, 1]
w = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36], dtype=np.float64)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

# =============================================================================
# FUNZIONE SINGOLA SIMULAZIONE
# =============================================================================
def run_poiseuille(H_cells, tau,collision_op, max_steps=100000, tol=1e-10):
    """
    Esegue una simulazione Poiseuille fino a convergenza (o max_steps).
    
    H_cells: Altezza totale griglia (Ny). 
             I muri sono a j=0 e j=Ny-1.
             I nodi fluidi sono H_cells - 2.
    tau:     Tempo di rilassamento.
    """
    
    Ny = H_cells
    Nx = 50  # Bastano pochi nodi in x per Poiseuille periodico
    
    # Parametri fisici/lattice
    rho0 = 1.0
    nu = (tau - 0.5) / 3.0
    omega = 1.0 / tau
    
    # Target velocity (bassa per restare in regime incompressibile)
    u_max_target = 0.1 
    
    # Calcolo Body Force Fx per ottenere u_max_target
    # Soluzione Poiseuille canale altezza L_phys: u_max = F * L^2 / (8 * nu * rho)
    # Per Half-Way BB, i muri sono a y=0.5 e y=Ny-1.5.
    # L_phys = (Ny - 1.5) - 0.5 = Ny - 2
    L_phys = Ny - 2.0
    Fx = (8.0 * nu * rho0 * u_max_target) / (L_phys**2)
    Fy = 0.0
    
    # Inizializzazione Array
    rho = np.ones((Ny, Nx), dtype=np.float64) * rho0
    ux  = np.zeros((Ny, Nx), dtype=np.float64)
    uy  = np.zeros((Ny, Nx), dtype=np.float64)
    
    f = np.zeros((Ny, Nx, 9), dtype=np.float64)
    f_new = np.zeros_like(f)
    
    # Maschera Solida (Muri a top e bottom)
    solid_mask = np.zeros((Ny, Nx), dtype=np.uint8)
    solid_mask[0, :] = 1
    solid_mask[-1, :] = 1
    
    # Inizializzazione f all'equilibrio (v=0)
    for k in range(9):
        f[:, :, k] = w[k] * rho0
    
    # Map omega (costante)
    omega_eff = np.full((Ny, Nx), omega, dtype=np.float64)
    
    # Loop Temporale
    err = 1.0
    it = 0
    
    # Flag per il kernel
    BC_NONE = 0
    
    # Init ux_old per convergenza
    ux_center_old = 0.0
    
    while it < max_steps:
        # Step LBM (collisione + streaming)
        lbm_step_modular(
            f, f_new, rho, ux, uy,
            cx_i, cy_i, w, opp,
            omega_eff,
            solid_mask,
            BC_NONE, 0, 0, 1.0, # Inlet (non usati in periodico)
            BC_NONE, 0, 1.0,    # Outlet (non usati in periodico)
            True, True, Fx, Fy, False, # periodic_x=True, forcing=True, periodic_y=False
            collision_op, lambda_trt=3/16
        )
        
        # Swap
        f, f_new = f_new, f # ATTENZIONE: Python swap semplice, ma in lbm_kernel controlla se f_new non viene sovrascritta
        # Nota: nel tuo codice cylinder, lo swap è f, f_new = f_new, f. È corretto.
        
        # Check convergenza ogni 100 step
        if it % 100 == 0:
            ux_center = np.mean(ux[Ny//2, :])
            if abs(ux_center) > 1e-12:
                err = abs(ux_center - ux_center_old) / abs(ux_center)
            ux_center_old = ux_center
            
            if err < tol:
                break
        
        it += 1
        
    # --- Post Processing Errore ---
    # Profilo numerico (media su x, in quanto omogeneo)
    u_num = np.mean(ux, axis=1)
    
    # Profilo Analitico
    # Coordinate nodi: j = 0, 1, ..., Ny-1
    # Muri Half-Way: y_bottom = 0.5, y_top = Ny - 1.5
    y_coords = np.arange(Ny, dtype=np.float64)
    y_bot = 0.5
    y_top = Ny - 1.5
    
    # u_ana(y) = F / (2*nu) * (y - y_bot) * (y_top - y)
    # Nota: dentro i muri solidi (j=0, j=Ny-1) u_ana sarà negativa o non fisica, 
    # ma noi calcoliamo l'errore solo sui nodi fluidi.
    u_ana = (Fx / (2.0 * nu * rho0)) * (y_coords - y_bot) * (y_top - y_coords)
    
    # Calcolo L2 Error solo sui nodi fluidi (da 1 a Ny-2)
    idx_fluid_start = 1
    idx_fluid_end = Ny - 1 # range esclude l'ultimo
    plt.plot(y_coords[1:-1],u_num[idx_fluid_start:idx_fluid_end], color = 'b')
    plt.plot(y_coords[1:-1],u_ana[idx_fluid_start:idx_fluid_end], color = 'k')
    plt.show()
    diff = u_num[idx_fluid_start:idx_fluid_end] - u_ana[idx_fluid_start:idx_fluid_end]
    norm_diff = np.sqrt(np.sum(diff**2))
    norm_ana  = np.sqrt(np.sum(u_ana[idx_fluid_start:idx_fluid_end]**2))
    
    L2_error = norm_diff / norm_ana
    
    return L2_error, it

# =============================================================================
# MAIN ROUTINE
# =============================================================================
if __name__ == "__main__":
    
    # Parametri simulazione
    collision_model = 0  # 0: BGK (come nel paper classico), 1: CNBGK
    print(f"Avvio validazione Poiseuille. Modello collisione: {collision_model}")
    print("-------------------------------------------------------------")

    # -----------------------------------------------------------
    # TEST A: Convergenza Spaziale (Vary H, Fixed Tau)
    # -----------------------------------------------------------
    print("Test A: Convergenza spaziale (Tau fisso, H variabile)...")
    tau_fix = 0.8
    # H rappresenta i punti griglia totali. 
    # H_effettivo (canale) sarà H-2. Usiamo valori che diano interi comodi.
    H_values = [5,10,20,30,40,80,100,160] 
    errors_H = []
    
    for H in H_values:
        err, steps = run_poiseuille(H, tau_fix, collision_op=collision_model)
        errors_H.append(err)
        print(f"  H={H:3d}, tau={tau_fix:.2f} -> L2 Error={err:.6e} (steps={steps})")
        
    # Calcolo pendenza ordine (tra ultimi due punti)
    slope = np.log(errors_H[-2]/errors_H[-1]) / np.log(H_values[-1]/H_values[-2])
    print(f"  Ordine di convergenza stimato: {slope:.2f}")

    # -----------------------------------------------------------
    # TEST B: Dipendenza da Tau (Fixed H, Vary Tau)
    # -----------------------------------------------------------
    print("\nTest B: Dipendenza da Tau (H fisso, Tau variabile)...")
    H_fix = 11  # Nel grafico usano H=9 o simile per avere pochi punti
    # Nota: se H_total=11 -> nodi fluidi=9. 
    
    # Range di tau che copre il grafico (0.51 a 5.0)
    tau_values = [0.45,0.48,0.51, 0.52, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    errors_tau = []
    
    for t in tau_values:
        err, steps = run_poiseuille(H_fix, t, collision_op=collision_model)
        errors_tau.append(err)
        print(f"  H={H_fix}, tau={t:.3f} -> L2 Error={err:.6e}")


    # =============================================================================
    # PLOTTING
    # =============================================================================
    #%%
    plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "lines.markersize": 7.0,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    
    # --- Plot A: Error vs H (Log-Log) ---
    ax[0].loglog(H_values, errors_H, 'ko-', label=f'LBM (tau={tau_fix})')
    
    # Linea riferimento 2° ordine (O(H^-2))
    # y = C * x^-2
    H_arr = np.array(H_values, dtype=float)
    ref_line = errors_H[0] * (H_arr[0]/H_arr)**2
    ax[0].loglog(H_arr, ref_line, 'k--', label='Order 2 Reference')
    
    ax[0].set_xlabel('H (Resolution)')
    ax[0].set_ylabel('$L_2$ Error')
    ax[0].set_title('Spatial Convergence')
    ax[0].legend(fontsize=25)
    ax[0].grid(True, which="both", ls="-", alpha=0.3)
    
    # --- Plot B: Error vs Tau (Semilog Y, o Log-Log) ---
    # Il grafico (b) originale ha assi log-log per tau e L2? No, tau è lineare o log?
    # Guardando l'immagine: 1, 2, 3 sono equispaziati -> Asse X lineare.
    # Asse Y logaritmico.
    
    ax[1].semilogy(tau_values, errors_tau, 'ko-', label=f'LBM (H={H_fix})')
    
    # Linea "Magic Tau" teorica per TRT/HalfWay
    # magic_tau = sqrt(3)/16 + 0.5 approx 0.608
    magic_tau = np.abs(errors_tau - np.nanmin(errors_tau))
    magic_tau = np.nanargmin(magic_tau)
    ax[1].axvline(x=tau_values[magic_tau], color='gray', linestyle='--', label=fr'$\tau = {tau_values[magic_tau]}$')
    
    ax[1].set_xlabel(r'$\tau / \Delta t$')
    ax[1].set_ylabel('$L_2$ Error')
    ax[1].set_title(f'Viscosity Dependence (H={H_fix})')
    ax[1].legend(fontsize=25)
    ax[1].grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'poiseuille_validation_results_{collision_model}.png', dpi=300)
    plt.show()