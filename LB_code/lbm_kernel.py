#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:53:45 2025

@author: angelo
"""

# lbm_kernel.py
import numpy as np
from numba import njit, prange

# ------------------------------------------------
# Utility: vorticità (solo per post-process, no Numba)
# ------------------------------------------------
def compute_vorticity(ux, uy):
    dvdx = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
    dudy = np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)
    return dvdx - dudy

# ------------------------------------------------
# Kernel generico LBM D2Q9 con tau_eff(x,y), solid_mask, inlet/outlet
# ------------------------------------------------
@njit(parallel=True)
def lbm_step_modular(
    f, f_new, rho, ux, uy,
    cx_i, cy_i, w, opp,
    omega_eff,
    solid_mask,

    # --- inlet ---
    bc_in_type, i_in, U_in, rho_in,

    # --- outlet ---
    bc_out_type, i_out, rho_out,

    # --- periodic flags ---
    periodic_x, use_forcing, Fx, Fy, periodic_y, collision_operator
):
    Ny, Nx = rho.shape


# ==================================================
# 1. COLLISIONE
#   collision_operator == 1 : BGK 1st order
#   collision_operator == 2 : CN 2nd order (Ubertini/Asinari/Succi transformed variable)
# ==================================================
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j, i]:
                continue
    
            rho_n = rho[j, i]
            ux_n  = ux[j, i]
            uy_n  = uy[j, i]
            u2_n  = ux_n*ux_n + uy_n*uy_n
    
            tau = 1.0 / omega_eff[j, i]
            omega = 1.0 / tau
            
            omega_plus = omega_eff[j, i]   # "fisico"
            tau_plus   = 1.0 / omega_plus
            # -------------------------
            # (A) 1st order BGK (explicit)
            # -------------------------
            if collision_operator == 0:
                for k in range(9):
                    cu = cx_i[k]*ux_n + cy_i[k]*uy_n
                    feq = w[k]*rho_n*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_n)
    
                    Fk = 0.0
                    if use_forcing:
                        cF = cx_i[k]*Fx + cy_i[k]*Fy
                        uF = ux_n*Fx + uy_n*Fy
                        Fk = w[k] * (3.0*cF + 9.0*cu*cF - 3.0*uF)
    
                    # 1st order: choose one
                    # (i) simple explicit source:
                    #f[j, i, k] = f[j, i, k] - omega*(f[j, i, k] - feq) +Fk
                    # (ii) alternatively (often better): Guo prefactor even in 1st order
                    f[j, i, k] = f[j, i, k] - omega*(f[j, i, k] - feq) + (1.0 - 0.5*omega)*Fk
    
            # -------------------------
            # (B) 2nd order CN via transformed population g (UAS)
            # -------------------------
            elif collision_operator == 1:
                # CNBGK (UAS) with transformed population g
                omega_tilde = 1.0 / (tau-0.5)  # = omega / (1 + 0.5*omega)
            
                for k in range(9):
                    cu = cx_i[k]*ux_n + cy_i[k]*uy_n
                    feq_n = w[k]*rho_n*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_n)
            
                    Fk_n = 0.0
                    if use_forcing:
                        cF = cx_i[k]*Fx + cy_i[k]*Fy
                        uF = ux_n*Fx + uy_n*Fy
                        Fk_n = w[k] * (3.0*cF + 9.0*cu*cF - 3.0*uF)
            

                    # 1) Transform f -> g (HSD Eq. 15, UAS Eq. 6)
                    # Nota: HSD usa h, UAS usa f_tilde. La logica è identica.
                    g = f[j, i, k] + 0.5 * (omega_tilde) * (f[j, i, k] - feq_n) - 0.5 *Fk_n
                    
                    # 2) Collide on g (Evolution equation, HSD Eq. 16, UAS Eq. 7)
                    # L'evoluzione è un rilassamento verso l'equilibrio con il forcing pesato
                    g = g - omega * (g - feq_n) + (tau * omega) * Fk_n
                    
                    # 3) Back-transform g -> f (UAS Eq. 8 invertita con forcing HSD)
                    # CORREZIONE: Il segno di feq_n deve essere POSITIVO
                    f[j, i, k] = (g + 0.5 * omega_tilde * feq_n + 0.5 * Fk_n) / (1.0 + 0.5 * omega)

            # -------------------------
            # (C) Regularized BGK (RLBM) - Per stabilità High-Re
            # -------------------------
            elif collision_operator == 2: 
                # 1. Calcolo tensore degli sforzi di non-equilibrio (Pi_neq)
                # Approssimazione standard: f_neq = f - feq
                # Ma per efficienza calcoliamo i momenti Q direttamente da f (poiché sum feq*Q = 0 a parte 2° ordine)
                
                # Calcolo feq non serve per i momenti Qxx, Qxy, Qyy se usiamo la formula veloce,
                # ma per chiarezza calcoliamo f_neq esplicito o usiamo i momenti diretti.
                # Metodo proiettivo diretto (Latt & Chopard):
                
                Qxx = 0.0
                Qxy = 0.0
                Qyy = 0.0
                
                for k in range(9):
                    # Ricostruiamo feq locale
                    c_x, c_y = cx_i[k], cy_i[k]
                    cu = c_x*ux_n + c_y*uy_n
                    feq_k = w[k]*rho_n*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_n)
                    
                    f_neq_k = f[j, i, k] - feq_k
                    
                    Qxx += f_neq_k * c_x * c_x
                    Qxy += f_neq_k * c_x * c_y
                    Qyy += f_neq_k * c_y * c_y
                
                # 2. Collisione Regularized
                # f_new = feq + (1-omega)*f_neq_projected + Source
                
                pref_S = (1.0 - 0.5 * omega) # Correzione 2° ordine temporale
                
                for k in range(9):
                    c_x, c_y = cx_i[k], cy_i[k]
                    cu = c_x*ux_n + c_y*uy_n
                    
                    # Equilibrio
                    feq = w[k]*rho_n*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_n)
                    
                    # Forcing
                    Fk = 0.0
                    if use_forcing:
                        cF = c_x*Fx + c_y*Fy
                        uF = ux_n*Fx + uy_n*Fy
                        Fk = w[k] * (3.0*cF + 9.0*cu*cF - 3.0*uF)
                    
                    # Proiezione del non-equilibrio sui polinomi di Hermite di 2° ordine
                    # (Filtra via i modi fantasma di ordine superiore)
                    # Formula D2Q9 standard per tensor projection:
                    # R_neq = (9/2) * w_k * [ (c_a c_b - 1/3 delta_ab) * Pi_ab ]
                    
                    term_xx = (c_x*c_x - 1.0/3.0) * Qxx
                    term_xy = (2.0*c_x*c_y)       * Qxy  # il 2x viene dalla simmetria xy/yx
                    term_yy = (c_y*c_y - 1.0/3.0) * Qyy
                    
                    f_neq_reg = 4.5 * w[k] * (term_xx + term_xy + term_yy)
                    
                    # Update finale
                    f[j, i, k] = feq + (1.0 - omega) * f_neq_reg + pref_S * Fk


# ==================================================
# 2. STREAMING
# ==================================================
    for k in range(9):
        cx = cx_i[k]
        cy = cy_i[k]
        k_opp = opp[k]

        for j in prange(Ny):
            for i in range(Nx):

                j_src = j - cy
                i_src = i - cx

                # periodicità x
                if periodic_x:
                    i_src %= Nx

                # periodicità y
                if periodic_y:
                    j_src %= Ny
                else:
                    # muro (bounce-back) ai bordi y
                    if j_src < 0 or j_src >= Ny:
                        f_new[j,i,k] = f[j,i,k_opp]
                        continue


                # fuori dominio x (open)
                if not periodic_x and (i_src < 0 or i_src >= Nx):
                    continue

                # bounce-back su solido
                if solid_mask[j_src, i_src]:
                    f_new[j,i,k] = f[j,i,k_opp]
                else:
                    f_new[j,i,k] = f[j_src, i_src, k]



    
# ==================================================
# 3. INLET BC
# ==================================================
    if bc_in_type == 1:  # ZOU–HE velocity
        ii = i_in
        for j in prange(Ny):
            if solid_mask[j,ii]:
                continue

            f0 = f_new[j,ii,0]
            f1 = f_new[j,ii,1]
            f2 = f_new[j,ii,2]
            f4 = f_new[j,ii,4]
            f5 = f_new[j,ii,5]
            f8 = f_new[j,ii,8]

            rho_loc = (f0 + f2 + f4 + 2.0*(f1 + f5 + f8)) / (1.0 - U_in)

            f_new[j,ii,3] = f1 - 2.0/3.0 * rho_loc * U_in
            f_new[j,ii,6] = f5 + 1.0/6.0 * rho_loc * U_in
            f_new[j,ii,7] = f8 + 1.0/6.0 * rho_loc * U_in

# ==================================================
# 4. OUTLET BC
# ==================================================
    if bc_out_type == 2:  # CONVECTIVE / zero-gradient
        io = i_out
        i_int = io - 1

        for j in prange(Ny):
            if solid_mask[j,io]:
                continue

            for k in range(9):
                f_new[j,io,k] = f_new[j,i_int,k]

# ==================================================
# 5. MACROSCOPICHE
# ==================================================
    for j in prange(Ny):
        for i in range(Nx):

            s = 0.0
            sx = 0.0
            sy = 0.0

            for k in range(9):
                fi = f_new[j,i,k]
                s  += fi
                sx += fi * cx_i[k]  
                sy += fi * cy_i[k] 

            if s < 1e-12:
                s = 1e-12

            rho[j,i] = s
            ux[j,i]  = (sx + 0.5*Fx) / s 
            uy[j,i]  = (sy  + 0.5*Fy)/ s 

# ==================================================
# 6. NO-SLIP SOLIDI
# ==================================================
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j,i]:
                ux[j,i] = 0.0
                uy[j,i] = 0.0
