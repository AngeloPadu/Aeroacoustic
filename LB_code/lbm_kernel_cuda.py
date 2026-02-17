#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-CUDA implementation of your modular D2Q9 LBM step.

Design goals:
- Keep the SAME algorithmic stages as your CPU version:
  1) collision on f (in-place)
  2) pull streaming into f_new
  3) inlet Zou–He velocity BC (bc_in_type==1)
  4) outlet zero-gradient BC (bc_out_type==2)
  5) macroscopic update from f_new
  6) enforce ux=uy=0 on solid nodes

Notes:
- This code assumes D2Q9 with the conventional ordering used by your CPU kernel.
- cx, cy, w, opp are defined as compile-time constants (tuples) for broad Numba compatibility.
- Prefer float64 on GPU (bandwidth-bound). If you need float64, it is possible but slower.
"""

import numpy as np
from numba import cuda, float64, int64

# -----------------------
# D2Q9 constants (compile-time tuples)
# -----------------------
# NOTE: We intentionally avoid cuda.const.array_like() because in newer Numba
# versions it is a device-only stub and raises NotImplementedError at import time.
CX  = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY  = (0, 0, 1, 0, -1, 1, 1, -1, -1)
# Weights stored as float64 compile-time constants.
W   = (np.float64(4.0/9.0), np.float64(1.0/9.0), np.float64(1.0/9.0), np.float64(1.0/9.0), np.float64(1.0/9.0), np.float64(1.0/36.0), np.float64(1.0/36.0), np.float64(1.0/36.0), np.float64(1.0/36.0))
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)

# ============================================================
# 1) Collision (in-place on f)
#    collision_operator:
#      0 -> BGK (explicit, Guo prefactor)
#      1 -> CN via transformed populations (UAS) (matches your code)
#      2 -> TRT (magic parameter lambda_trt)
# ============================================================
@cuda.jit
def collision_kernel(
    f, rho, ux, uy,
    omega_eff,
    solid_mask,
    use_forcing, Fx, Fy,
    collision_operator, lambda_trt
):
    y, x = cuda.grid(2)
    ny, nx, _q = f.shape
    if x >= nx or y >= ny:
        return
    if solid_mask[y, x]:
        return

    rho_n = rho[y, x]
    ux_n  = ux[y, x]
    uy_n  = uy[y, x]
    u2_n  = ux_n*ux_n + uy_n*uy_n

    omega_lbm = omega_eff[y, x]  # = 1/tau_lbm

    # ---- BGK 1st order ----
    if collision_operator == 0:
        for k in range(9):
            cu  = CX[k]*ux_n + CY[k]*uy_n
            feq = W[k]*rho_n*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_n)

            Fk = 0.0
            if use_forcing:
                cF = CX[k]*Fx + CY[k]*Fy
                uF = ux_n*Fx + uy_n*Fy
                Fk = W[k] * (3.0*cF + 9.0*cu*cF - 3.0*uF)

            f[y, x, k] = f[y, x, k] - omega_lbm*(f[y, x, k] - feq) + (1.0 - 0.5*omega_lbm)*Fk
        return

    # ---- CN 2nd order via transformed populations (your exact logic) ----
    if collision_operator == 1:
        tau_lbm = 1.0 / omega_lbm
        tau_pde = tau_lbm - 0.5

        if tau_pde <= 1e-15:
            # fall back to BGK update
            for k in range(9):
                cu  = CX[k]*ux_n + CY[k]*uy_n
                feq = W[k]*rho_n*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_n)

                Fk = 0.0
                if use_forcing:
                    cF = CX[k]*Fx + CY[k]*Fy
                    uF = ux_n*Fx + uy_n*Fy
                    Fk = W[k] * (3.0*cF + 9.0*cu*cF - 3.0*uF)

                f[y, x, k] = f[y, x, k] - omega_lbm*(f[y, x, k] - feq) + (1.0 - 0.5*omega_lbm)*Fk
        else:
            omega_pde   = 1.0 / tau_pde
            invA        = 1.0 / (1.0 + 0.5*omega_pde)
            omega_tilde = 1.0 / (tau_pde + 0.5)  # equals omega_lbm
            force_pref  = tau_pde * omega_tilde  # = tau_pde/(tau_pde+0.5)

            for k in range(9):
                cu = CX[k]*ux_n + CY[k]*uy_n
                feq_n = W[k]*rho_n*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2_n)

                Fk_n = 0.0
                if use_forcing:
                    cF = CX[k]*Fx + CY[k]*Fy
                    uF = ux_n*Fx + uy_n*Fy
                    Fk_n = W[k] * (3.0*cF + 9.0*cu*cF - 3.0*uF)

                # 1) Transform f -> g
                g = f[y, x, k] + 0.5*omega_pde*(f[y, x, k] - feq_n) - 0.5*Fk_n
                # 2) Collide in g
                g = g - omega_tilde*(g - feq_n) + force_pref*Fk_n
                # 3) Back-transform g -> f
                f[y, x, k] = (g + 0.5*omega_pde*feq_n + 0.5*Fk_n) * invA
        return

    # ---- TRT ----
    if collision_operator == 2:
        omega_plus = omega_lbm
        tau_plus   = 1.0 / omega_plus

        denom = (tau_plus - 0.5)
        if denom <= 1e-15:
            tau_minus = tau_plus
        else:
            tau_minus = 0.5 + (lambda_trt / denom)

        omega_minus = 1.0 / tau_minus

        pref_plus  = (1.0 - 0.5 * omega_plus)
        pref_minus = (1.0 - 0.5 * omega_minus)

        # k=0 (pure symmetric)
        k = 0
        feq0 = W[k]*rho_n*(1.0 - 1.5*u2_n)

        F0 = 0.0
        if use_forcing:
            uF = ux_n*Fx + uy_n*Fy
            F0 = W[k] * (-3.0*uF)

        f[y, x, k] = f[y, x, k] - omega_plus*(f[y, x, k] - feq0) + pref_plus*F0

        # helper inline: process opposite pair (k, ko)
        # pairs: (1,3), (2,4), (5,7), (6,8)
        for pair in range(4):
            if pair == 0:
                k, ko = 1, 3
            elif pair == 1:
                k, ko = 2, 4
            elif pair == 2:
                k, ko = 5, 7
            else:
                k, ko = 6, 8

            fk  = f[y, x, k]
            fko = f[y, x, ko]

            cu_k  = CX[k]*ux_n  + CY[k]*uy_n
            cu_ko = CX[ko]*ux_n + CY[ko]*uy_n

            feq_k  = W[k]*rho_n*(1.0 + 3.0*cu_k  + 4.5*cu_k*cu_k  - 1.5*u2_n)
            feq_ko = W[ko]*rho_n*(1.0 + 3.0*cu_ko + 4.5*cu_ko*cu_ko - 1.5*u2_n)

            Fk  = 0.0
            Fko = 0.0
            if use_forcing:
                cF_k  = CX[k]*Fx  + CY[k]*Fy
                cF_ko = CX[ko]*Fx + CY[ko]*Fy
                uF = ux_n*Fx + uy_n*Fy
                Fk  = W[k]  * (3.0*cF_k  + 9.0*cu_k*cF_k   - 3.0*uF)
                Fko = W[ko] * (3.0*cF_ko + 9.0*cu_ko*cF_ko - 3.0*uF)

            f_plus    = 0.5*(fk + fko)
            f_minus   = 0.5*(fk - fko)
            feq_plus  = 0.5*(feq_k + feq_ko)
            feq_minus = 0.5*(feq_k - feq_ko)
            F_plus    = 0.5*(Fk + Fko)
            F_minus   = 0.5*(Fk - Fko)

            f_plus  = f_plus  - omega_plus *(f_plus  - feq_plus)  + pref_plus  * F_plus
            f_minus = f_minus - omega_minus*(f_minus - feq_minus) + pref_minus * F_minus

            f[y, x, k]  = f_plus + f_minus
            f[y, x, ko] = f_plus - f_minus

        return


# ============================================================
# 2) Streaming (pull) into f_new
#    Matches your CPU logic, but ensures every entry is written.
# ============================================================
@cuda.jit
def streaming_kernel(
    f, f_new,
    solid_mask,
    periodic_x, periodic_y
):
    y, x = cuda.grid(2)
    ny, nx, _q = f.shape
    if x >= nx or y >= ny:
        return

    for k in range(9):
        cx = CX[k]
        cy = CY[k]
        k_opp = OPP[k]

        y_src = y - cy
        x_src = x - cx

        # periodic x
        if periodic_x:
            x_src = x_src % nx
        else:
            # open x: keep something defined (self-carry) and let BC overwrite if needed
            if x_src < 0 or x_src >= nx:
                f_new[y, x, k] = f[y, x, k]
                continue

        # periodic y or wall bounce-back
        if periodic_y:
            y_src = y_src % ny
        else:
            if y_src < 0 or y_src >= ny:
                f_new[y, x, k] = f[y, x, k_opp]
                continue

        # bounce-back on solids (same as CPU: check source cell)
        if solid_mask[y_src, x_src]:
            f_new[y, x, k] = f[y, x, k_opp]
        else:
            f_new[y, x, k] = f[y_src, x_src, k]


# ============================================================
# 3) Inlet BC: Zou–He velocity (your simplified version)
# ============================================================
@cuda.jit
def inlet_zou_he_velocity_kernel(f_new, solid_mask, i_in, U_in):
    y = cuda.grid(1)
    ny, nx, _q = f_new.shape
    if y >= ny:
        return
    x = i_in
    if x < 0 or x >= nx:
        return
    if solid_mask[y, x]:
        return

    f0 = f_new[y, x, 0]
    f1 = f_new[y, x, 1]
    f2 = f_new[y, x, 2]
    f4 = f_new[y, x, 4]
    f5 = f_new[y, x, 5]
    f8 = f_new[y, x, 8]

    rho_loc = (f0 + f2 + f4 + 2.0*(f1 + f5 + f8)) / (1.0 - U_in)

    f_new[y, x, 3] = f1 - (2.0/3.0) * rho_loc * U_in
    f_new[y, x, 6] = f5 + (1.0/6.0) * rho_loc * U_in
    f_new[y, x, 7] = f8 + (1.0/6.0) * rho_loc * U_in


# ============================================================
# 4) Outlet BC: zero-gradient (copy from interior)
# ============================================================
@cuda.jit
def outlet_zero_gradient_kernel(f_new, solid_mask, i_out):
    y = cuda.grid(1)
    ny, nx, _q = f_new.shape
    if y >= ny:
        return
    x = i_out
    if x < 0 or x >= nx:
        return
    if solid_mask[y, x]:
        return

    x_int = x - 1
    if x_int < 0:
        x_int = 0

    for k in range(9):
        f_new[y, x, k] = f_new[y, x_int, k]


# ============================================================
# 5) Macros from f_new
# ============================================================
@cuda.jit
def macroscopic_kernel(f_new, rho, ux, uy, Fx, Fy):
    y, x = cuda.grid(2)
    ny, nx, _q = f_new.shape
    if x >= nx or y >= ny:
        return

    s  = 0.0
    sx = 0.0
    sy = 0.0
    for k in range(9):
        fi = f_new[y, x, k]
        s  += fi
        sx += fi * CX[k]
        sy += fi * CY[k]

    if s < 1e-12:
        s = 1e-12

    rho[y, x] = s
    ux[y, x]  = (sx + 0.5*Fx) / s
    uy[y, x]  = (sy + 0.5*Fy) / s


# ============================================================
# 6) No-slip solids (zero velocity)
# ============================================================
@cuda.jit
def enforce_no_slip_kernel(ux, uy, solid_mask):
    y, x = cuda.grid(2)
    ny, nx = ux.shape
    if x >= nx or y >= ny:
        return
    if solid_mask[y, x]:
        ux[y, x] = 0.0
        uy[y, x] = 0.0


# ============================================================
# Host-side orchestrator
# ============================================================
def lbm_step_modular_cuda(
    f_d, f_new_d, rho_d, ux_d, uy_d,
    omega_eff_d, solid_mask_d,
    bc_in_type, i_in, U_in, rho_in,         # rho_in currently unused (same as CPU)
    bc_out_type, i_out, rho_out,            # rho_out currently unused (same as CPU)
    periodic_x, use_forcing, Fx, Fy, periodic_y,
    collision_operator, lambda_trt,
    threads=(16, 16),
    stream=0
):
    """
    Perform one LBM step on the GPU.

    Inputs are expected to be CUDA device arrays (numba.cuda.devicearray.DeviceNDArray):
      - f_d, f_new_d: (Ny, Nx, 9) float64
      - rho_d, ux_d, uy_d: (Ny, Nx) float64
      - omega_eff_d: (Ny, Nx) float64
      - solid_mask_d: (Ny, Nx) boolean or uint8

    Returns: None (updates arrays in-place). Typical usage:
        lbm_step_modular_cuda(...)
        f_d, f_new_d = f_new_d, f_d   # ping-pong swap on host
    """
    ny, nx, _q = f_d.shape
    # CORREZIONE: Invertiamo l'ordine per matchare (y, x)
    ty, tx = threads[0], threads[1] 
    blocks_y = (ny + ty - 1) // ty
    blocks_x = (nx + tx - 1) // tx
    blocks2d = (blocks_y, blocks_x)

    # ensure scalar types are GPU-friendly
    Fx32 = np.float64(Fx)
    Fy32 = np.float64(Fy)
    U32  = np.float64(U_in)
    lam32 = np.float64(lambda_trt)

    # 1) collision
    collision_kernel[blocks2d, threads, stream](
        f_d, rho_d, ux_d, uy_d,
        omega_eff_d, solid_mask_d,
        int(use_forcing), Fx32, Fy32,
        int(collision_operator), lam32
    )

    # 2) streaming
    streaming_kernel[blocks2d, threads, stream](
        f_d, f_new_d, solid_mask_d,
        int(periodic_x), int(periodic_y)
    )

    # 3) inlet BC
    if bc_in_type == 1:
        threads1d = 256
        blocks1d = (ny + threads1d - 1) // threads1d
        inlet_zou_he_velocity_kernel[blocks1d, threads1d, stream](f_new_d, solid_mask_d, int(i_in), U32)

    # 4) outlet BC
    if bc_out_type == 2:
        threads1d = 256
        blocks1d = (ny + threads1d - 1) // threads1d
        outlet_zero_gradient_kernel[blocks1d, threads1d, stream](f_new_d, solid_mask_d, int(i_out))

    # 5) macros
    macroscopic_kernel[blocks2d, threads, stream](f_new_d, rho_d, ux_d, uy_d, Fx32, Fy32)

    # 6) no-slip
    enforce_no_slip_kernel[blocks2d, threads, stream](ux_d, uy_d, solid_mask_d)


# ============================================================
# Convenience: allocate & run example (optional)
# ============================================================
def to_device_float64(a):
    """Ensure float64 + contiguous before sending to GPU."""
    a = np.asarray(a, dtype=np.float64)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return cuda.to_device(a)