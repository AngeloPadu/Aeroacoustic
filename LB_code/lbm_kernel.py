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
def lbm_step_generic(f, f_new, rho, ux, uy,
                     cx_i, cy_i, w, opp,
                     omega_eff,
                     solid_mask,
                     # inlet options
                     use_inlet, i_in, U_in, rho_in,
                     # outlet options
                     use_outlet, i_out, rho_out):
    """
    f, f_new : (Ny, Nx, 9) distribuzioni
    rho, ux, uy : (Ny, Nx)
    cx_i, cy_i : (9,)
    w : (9,)
    opp : (9,) indici opposti
    omega_eff : (Ny, Nx) collision frequency locale
    solid_mask : (Ny, Nx) uint8, 1 = solido, 0 = fluido
    inlet: se use_inlet=True, impone velocità (U_in,0) su colonna i_in con densità rho_in
    outlet: se use_outlet=True, impone densità rho_out su colonna i_out con u ricavata da colonna interna
    """

    Ny, Nx = rho.shape

    # ---- COLLISIONE con tau_eff ----
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j,i] == 1:
                continue  # celle solide gestite da bounce-back
            u2 = ux[j,i]*ux[j,i] + uy[j,i]*uy[j,i]
            om = omega_eff[j,i]
            for k in range(9):
                cu = cx_i[k]*ux[j,i] + cy_i[k]*uy[j,i]
                feq = w[k]*rho[j,i]*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2)
                f[j,i,k] += -om * (f[j,i,k] - feq)

    # ---- STREAMING: f -> f_new ----
    for k in range(9):
        cx = cx_i[k]
        cy = cy_i[k]
        for j in prange(Ny):
            j_src = (j - cy) % Ny
            for i in range(Nx):
                i_src = (i - cx) % Nx
                f_new[j,i,k] = f[j_src,i_src,k]

    # ---- BOUNCE-BACK per tutte le celle solide ----
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j,i] == 1:
                for k in range(9):
                    f_new[j,i,k] = f_new[j,i,opp[k]]

    # ---- INLET: velocità imposta (opzionale) ----
    if use_inlet:
        ii = i_in
        for j in prange(Ny):
            if solid_mask[j,ii] == 1:
                continue
            rho[j,ii] = rho_in
            ux[j,ii]  = U_in
            uy[j,ii]  = 0.0
            u2_in = ux[j,ii]*ux[j,ii] + uy[j,ii]*uy[j,ii]
            for k in range(9):
                cu = cx_i[k]*ux[j,ii] + cy_i[k]*uy[j,ii]
                f_new[j,ii,k] = w[k]*rho[j,ii]*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_in)

    # ---- OUTLET: pressione fissata rho_out (opzionale) ----
    if use_outlet:
        io = i_out
        i_int = io - 1 if io > 0 else io + 1  # colonna interna adiacente
        for j in prange(Ny):
            if solid_mask[j,io] == 1:
                continue
            # macro sulla colonna interna i_int
            s = 0.0
            sx = 0.0
            sy = 0.0
            for k in range(9):
                fi = f_new[j,i_int,k]
                s  += fi
                sx += fi * cx_i[k]
                sy += fi * cy_i[k]
            if s < 1e-12:
                s = 1e-12
            ux_int = sx / s
            uy_int = sy / s

            u2_out = ux_int*ux_int + uy_int*uy_int
            rho[j,io] = rho_out
            ux[j,io]  = ux_int
            uy[j,io]  = uy_int
            for k in range(9):
                cu = cx_i[k]*ux_int + cy_i[k]*uy_int
                f_new[j,io,k] = w[k]*rho_out*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_out)

    # ---- MACRO GLOBALI da f_new ----
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

    # ---- NO-SLIP automatico sulle celle solide ----
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j,i] == 1:
                ux[j,i] = 0.0
                uy[j,i] = 0.0
