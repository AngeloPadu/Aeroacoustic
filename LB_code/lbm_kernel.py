#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lbm_kernel_refactored.py

Refactor of the original monolithic kernel into small Numba-friendly sub-kernels.

Main entry point (same signature as before):
    lbm_step_modular(...)

Pipeline executed inside `lbm_step_modular`:
    1) collision (BGK / CN / TRT)
    2) streaming (pull scheme + bounce-back on solids / y-walls)
    3) inlet BC (Zou–He velocity, optional)
    4) outlet BC (zero-gradient, optional)
    5) macroscopic update (rho, u)
    6) enforce no-slip on solids (u=0)
"""

import numpy as np
from numba import njit, prange


# ------------------------------------------------
# Utility: vorticity (post-process only)
# ------------------------------------------------
def compute_vorticity(ux, uy):
    dvdx = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
    dudy = np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)
    return dvdx - dudy


# ------------------------------------------------
# Small inline helpers (Numba)
# ------------------------------------------------
@njit(inline="always")
def _feq_and_cu(rho, ux, uy, u2, cx, cy, wk):
    cu = cx * ux + cy * uy
    feq = wk * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    return feq, cu


@njit(inline="always")
def _guo_force(wk, cx, cy, Fx, Fy, cu, uF):
    # Guo forcing term (D2Q9)
    cF = cx * Fx + cy * Fy
    return wk * (3.0 * cF + 9.0 * cu * cF - 3.0 * uF)


# ------------------------------------------------
# 1) COLLISION operators
#   collision_operator:
#       0 -> BGK 1st order (explicit)
#       1 -> CN 2nd order via transformed populations (UAS)
#       2 -> TRT (Two-Relaxation-Time)
# ------------------------------------------------
@njit(parallel=True)
def _collision_bgk(
    f,
    rho,
    ux,
    uy,
    cx_i,
    cy_i,
    w,
    omega_eff,
    solid_mask,
    use_forcing,
    Fx,
    Fy,
):
    Ny, Nx = rho.shape

    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j, i]:
                continue

            rho_n = rho[j, i]
            ux_n = ux[j, i]
            uy_n = uy[j, i]
            u2_n = ux_n * ux_n + uy_n * uy_n

            omega = omega_eff[j, i]
            pref = 1.0 - 0.5 * omega

            uF = ux_n * Fx + uy_n * Fy

            for k in range(9):
                feq, cu = _feq_and_cu(rho_n, ux_n, uy_n, u2_n, cx_i[k], cy_i[k], w[k])

                Fk = 0.0
                if use_forcing:
                    Fk = _guo_force(w[k], cx_i[k], cy_i[k], Fx, Fy, cu, uF)

                f[j, i, k] = f[j, i, k] - omega * (f[j, i, k] - feq) + pref * Fk


@njit(parallel=True)
def _collision_cn_uas(
    f,
    rho,
    ux,
    uy,
    cx_i,
    cy_i,
    w,
    omega_eff,
    solid_mask,
    use_forcing,
    Fx,
    Fy,
):
    """2nd order CN via transformed population g (Ubertini/Asinari/Succi style).

    Notes (kept consistent with the original kernel):
      - tau_lbm = 1/omega_eff
      - tau_pde = tau_lbm - 1/2 (avoid double +1/2 shift)
      - omega_tilde = 1/(tau_pde + 1/2) = 1/tau_lbm = omega_eff
    """
    Ny, Nx = rho.shape

    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j, i]:
                continue

            rho_n = rho[j, i]
            ux_n = ux[j, i]
            uy_n = uy[j, i]
            u2_n = ux_n * ux_n + uy_n * uy_n

            omega_lbm = omega_eff[j, i]
            tau_lbm = 1.0 / omega_lbm

            tau_pde = tau_lbm - 0.5
            uF = ux_n * Fx + uy_n * Fy

            if tau_pde <= 1e-15:
                # Degenerate (tau_lbm <= 0.5): fall back to BGK update
                pref = 1.0 - 0.5 * omega_lbm
                for k in range(9):
                    feq, cu = _feq_and_cu(rho_n, ux_n, uy_n, u2_n, cx_i[k], cy_i[k], w[k])
                    Fk = 0.0
                    if use_forcing:
                        Fk = _guo_force(w[k], cx_i[k], cy_i[k], Fx, Fy, cu, uF)
                    f[j, i, k] = f[j, i, k] - omega_lbm * (f[j, i, k] - feq) + pref * Fk
                continue

            omega_pde = 1.0 / tau_pde
            invA = 1.0 / (1.0 + 0.5 * omega_pde)  # A = 1 + 0.5*omega_pde

            omega_tilde = 1.0 / (tau_pde + 0.5)  # == omega_lbm
            force_pref = tau_pde * omega_tilde  # == tau_pde/(tau_pde+0.5)

            for k in range(9):
                feq, cu = _feq_and_cu(rho_n, ux_n, uy_n, u2_n, cx_i[k], cy_i[k], w[k])
                Fk = 0.0
                if use_forcing:
                    Fk = _guo_force(w[k], cx_i[k], cy_i[k], Fx, Fy, cu, uF)

                # 1) Transform f -> g
                g = f[j, i, k] + 0.5 * omega_pde * (f[j, i, k] - feq) - 0.5 * Fk

                # 2) Collide in g
                g = g - omega_tilde * (g - feq) + force_pref * Fk

                # 3) Back-transform g -> f
                f[j, i, k] = (g + 0.5 * omega_pde * feq + 0.5 * Fk) * invA


# D2Q9 opposite pairs excluding 0 (self-opposite)
_TRT_PAIRS = ((1, 3), (2, 4), (5, 7), (6, 8))


@njit(inline="always")
def _trt_update_pair(
    fk,
    fko,
    feq_k,
    feq_ko,
    Fk,
    Fko,
    omega_plus,
    omega_minus,
):
    # +/- decomposition
    f_plus = 0.5 * (fk + fko)
    f_minus = 0.5 * (fk - fko)
    feq_plus = 0.5 * (feq_k + feq_ko)
    feq_minus = 0.5 * (feq_k - feq_ko)
    F_plus = 0.5 * (Fk + Fko)
    F_minus = 0.5 * (Fk - Fko)

    pref_plus = 1.0 - 0.5 * omega_plus
    pref_minus = 1.0 - 0.5 * omega_minus

    f_plus = f_plus - omega_plus * (f_plus - feq_plus) + pref_plus * F_plus
    f_minus = f_minus - omega_minus * (f_minus - feq_minus) + pref_minus * F_minus

    out_k = f_plus + f_minus
    out_ko = f_plus - f_minus
    return out_k, out_ko


@njit(parallel=True)
def _collision_trt(
    f,
    rho,
    ux,
    uy,
    cx_i,
    cy_i,
    w,
    omega_eff,
    solid_mask,
    use_forcing,
    Fx,
    Fy,
    lambda_trt,
):
    Ny, Nx = rho.shape

    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j, i]:
                continue

            rho_n = rho[j, i]
            ux_n = ux[j, i]
            uy_n = uy[j, i]
            u2_n = ux_n * ux_n + uy_n * uy_n
            uF = ux_n * Fx + uy_n * Fy

            omega_plus = omega_eff[j, i]
            tau_plus = 1.0 / omega_plus

            # tau_minus from magic parameter lambda_trt:
            #   (tau_plus - 0.5)*(tau_minus - 0.5) = lambda_trt
            denom = tau_plus - 0.5
            if denom <= 1e-15:
                tau_minus = tau_plus  # fallback (degenerates to BGK)
            else:
                tau_minus = 0.5 + (lambda_trt / denom)

            omega_minus = 1.0 / tau_minus

            # k=0 (pure symmetric)
            feq0 = w[0] * rho_n * (1.0 - 1.5 * u2_n)
            F0 = 0.0
            if use_forcing:
                F0 = w[0] * (-3.0 * uF)  # cF=0, cu=0
            f[j, i, 0] = f[j, i, 0] - omega_plus * (f[j, i, 0] - feq0) + (1.0 - 0.5 * omega_plus) * F0

            # paired directions
            for k, ko in _TRT_PAIRS:

                fk = f[j, i, k]
                fko = f[j, i, ko]

                feq_k, cu_k = _feq_and_cu(rho_n, ux_n, uy_n, u2_n, cx_i[k], cy_i[k], w[k])
                feq_ko, cu_ko = _feq_and_cu(rho_n, ux_n, uy_n, u2_n, cx_i[ko], cy_i[ko], w[ko])

                Fk = 0.0
                Fko = 0.0
                if use_forcing:
                    Fk = _guo_force(w[k], cx_i[k], cy_i[k], Fx, Fy, cu_k, uF)
                    Fko = _guo_force(w[ko], cx_i[ko], cy_i[ko], Fx, Fy, cu_ko, uF)

                out_k, out_ko = _trt_update_pair(
                    fk,
                    fko,
                    feq_k,
                    feq_ko,
                    Fk,
                    Fko,
                    omega_plus,
                    omega_minus,
                )

                f[j, i, k] = out_k
                f[j, i, ko] = out_ko


# ------------------------------------------------
# 2) STREAMING (pull)
# ------------------------------------------------
@njit(parallel=True)
def _streaming_pull(
    f,
    f_new,
    cx_i,
    cy_i,
    opp,
    solid_mask,
    periodic_x,
    periodic_y,
):
    Ny, Nx = solid_mask.shape

    for k in range(9):
        cx = cx_i[k]
        cy = cy_i[k]
        k_opp = opp[k]

        for j in prange(Ny):
            for i in range(Nx):
                j_src = j - cy
                i_src = i - cx

                # periodic in x
                if periodic_x:
                    i_src %= Nx

                # periodic in y, otherwise bounce-back on y-walls
                if periodic_y:
                    j_src %= Ny
                else:
                    if j_src < 0 or j_src >= Ny:
                        f_new[j, i, k] = f[j, i, k_opp]
                        continue

                # open boundary in x if not periodic
                if (not periodic_x) and (i_src < 0 or i_src >= Nx):
                    # keep original behaviour (leave as-is)
                    continue

                # bounce-back on solids (from source cell)
                if solid_mask[j_src, i_src]:
                    f_new[j, i, k] = f[j, i, k_opp]
                else:
                    f_new[j, i, k] = f[j_src, i_src, k]


# ------------------------------------------------
# 3) INLET BC (Zou–He velocity)
# ------------------------------------------------
@njit(parallel=True)
def _inlet_zou_he_velocity(f_new, solid_mask, i_in, U_in):
    Ny, _ = solid_mask.shape
    ii = i_in

    for j in prange(Ny):
        if solid_mask[j, ii]:
            continue

        f0 = f_new[j, ii, 0]
        f1 = f_new[j, ii, 1]
        f2 = f_new[j, ii, 2]
        f4 = f_new[j, ii, 4]
        f5 = f_new[j, ii, 5]
        f8 = f_new[j, ii, 8]

        rho_loc = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 - U_in)

        f_new[j, ii, 3] = f1 - (2.0 / 3.0) * rho_loc * U_in
        f_new[j, ii, 6] = f5 + (1.0 / 6.0) * rho_loc * U_in
        f_new[j, ii, 7] = f8 + (1.0 / 6.0) * rho_loc * U_in


# ------------------------------------------------
# 4) OUTLET BC (zero-gradient / convective)
# ------------------------------------------------
@njit(parallel=True)
def _outlet_zero_gradient(f_new, solid_mask, i_out):
    Ny, _ = solid_mask.shape
    io = i_out
    i_int = io - 1

    for j in prange(Ny):
        if solid_mask[j, io]:
            continue
        for k in range(9):
            f_new[j, io, k] = f_new[j, i_int, k]


# ------------------------------------------------
# 5) MACROSCOPIC fields (rho, u)
# ------------------------------------------------
@njit(parallel=True)
def _macroscopic(f_new, rho, ux, uy, cx_i, cy_i, Fx, Fy):
    Ny, Nx = rho.shape

    for j in prange(Ny):
        for i in range(Nx):
            s = 0.0
            sx = 0.0
            sy = 0.0

            for k in range(9):
                fi = f_new[j, i, k]
                s += fi
                sx += fi * cx_i[k]
                sy += fi * cy_i[k]

            if s < 1e-12:
                s = 1e-12

            rho[j, i] = s
            ux[j, i] = (sx + 0.5 * Fx) / s
            uy[j, i] = (sy + 0.5 * Fy) / s


# ------------------------------------------------
# 6) Enforce no-slip on solids
# ------------------------------------------------
@njit(parallel=True)
def _enforce_no_slip(ux, uy, solid_mask):
    Ny, Nx = solid_mask.shape
    for j in prange(Ny):
        for i in range(Nx):
            if solid_mask[j, i]:
                ux[j, i] = 0.0
                uy[j, i] = 0.0


# ------------------------------------------------
# Main wrapper (same signature as the original)
# ------------------------------------------------
@njit
def lbm_step_modular(
    f,
    f_new,
    rho,
    ux,
    uy,
    cx_i,
    cy_i,
    w,
    opp,
    omega_eff,
    solid_mask,
    # --- inlet ---
    bc_in_type,
    i_in,
    U_in,
    rho_in,
    # --- outlet ---
    bc_out_type,
    i_out,
    rho_out,
    # --- periodic + forcing + collision ---
    periodic_x,
    use_forcing,
    Fx,
    Fy,
    periodic_y,
    collision_operator,
    lambda_trt,
):
    """One LBM time-step (D2Q9).

    Parameters are kept to match the original kernel signature.
    `rho_in` and `rho_out` are currently unused (placeholders for future BCs).
    """
    _ = rho_in
    _ = rho_out

    # 1) collision
    if collision_operator == 0:
        _collision_bgk(f, rho, ux, uy, cx_i, cy_i, w, omega_eff, solid_mask, use_forcing, Fx, Fy)
    elif collision_operator == 1:
        _collision_cn_uas(f, rho, ux, uy, cx_i, cy_i, w, omega_eff, solid_mask, use_forcing, Fx, Fy)
    else:
        _collision_trt(f, rho, ux, uy, cx_i, cy_i, w, omega_eff, solid_mask, use_forcing, Fx, Fy, lambda_trt)

    # 2) streaming
    _streaming_pull(f, f_new, cx_i, cy_i, opp, solid_mask, periodic_x, periodic_y)

    # 3) inlet
    if bc_in_type == 1:
        _inlet_zou_he_velocity(f_new, solid_mask, i_in, U_in)

    # 4) outlet
    if bc_out_type == 2:
        _outlet_zero_gradient(f_new, solid_mask, i_out)

    # 5) macros
    _macroscopic(f_new, rho, ux, uy, cx_i, cy_i, Fx, Fy)

    # 6) no-slip on solids
    _enforce_no_slip(ux, uy, solid_mask)
