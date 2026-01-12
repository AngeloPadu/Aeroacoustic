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
    periodic_x, use_forcing, Fx, Fy
):
    Ny, Nx = rho.shape

    # ==================================================
    # 1. COLLISIONE (Crank–Nicolson BGK)
    # ==================================================
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j,i]:
                continue

            u2 = ux[j,i]**2 + uy[j,i]**2
            tau = 1.0 / omega_eff[j,i]
            omega_CN = (1.0/tau) / (1.0 + 0.5*tau)

            for k in range(9):
                cu = cx_i[k]*ux[j,i] + cy_i[k]*uy[j,i]
                feq = w[k]*rho[j,i] * (
                    1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2
                )
                Fk = 0.0
                if use_forcing:
                    cu = cx_i[k]*ux[j,i] + cy_i[k]*uy[j,i]
                    cF = cx_i[k]*Fx + cy_i[k]*Fy
                    uF = ux[j,i]*Fx + uy[j,i]*Fy
                
                    Fk = w[k] * (1.0 - 0.5/tau) * (
                          3.0 * cF
                        + 9.0 * cu * cF
                        - 3.0 * uF
                    )

                # pre-collisione CN
                f_tilde = f[j,i,k] + 0.5/tau * (feq - f[j,i,k]) + 0.5 * Fk
                
                # collisione
                f[j,i,k] = f_tilde - omega_CN * (f_tilde - feq) + 0.5 * Fk

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

                # fuori dominio y
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
            ux[j,i]  = sx / s
            uy[j,i]  = sy / s

    # ==================================================
    # 6. NO-SLIP SOLIDI
    # ==================================================
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j,i]:
                ux[j,i] = 0.0
                uy[j,i] = 0.0
