#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 15:01:02 2026

@author: angelo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acoustic Gaussian Pulse in a Channel (LBM D2Q9)
No mean flow — pure acoustic propagation
"""

import numpy as np
import matplotlib.pyplot as plt
from lbm_kernel import lbm_step_modular
import pyvista as pv
from field_saver import field_saver
from helmholtz_generator import add_helmholtz_resonator, helmholtz_frequency

case_name = 'gaussian_pulse'

# =============================================================================
# D2Q9 CONSTANTS
# =============================================================================

q = 9
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1],
              [1,1], [-1,1], [-1,-1], [1,-1]], dtype=np.int64)

cx_i = c[:, 0]
cy_i = c[:, 1]

w = np.array([4/9, 1/9,1/9,1/9,1/9,
              1/36,1/36,1/36,1/36], dtype=np.float64)

opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================
def initialize_gaussian_pulse(rho, rho0, A, x0, y0, sigma):
    """Create Gaussian density perturbation"""
    Ny, Nx = rho.shape
    y = np.arange(Ny)[:, None]
    x = np.arange(Nx)[None, :]
    gaussian = A * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    rho[:, :] = rho0 + gaussian


def initialize_equilibrium(f, rho, ux, uy):
    """Initialize populations at equilibrium"""
    Ny, Nx, q = f.shape
    for k in range(q):
        cu = 3.0 * (cx_i[k]*ux + cy_i[k]*uy)
        f[:, :, k] = w[k] * rho * (
            1 + cu + 0.5*cu**2 - 1.5*(ux**2 + uy**2)
        )
        
def initialize_local_plane_wave(rho, ux, uy, rho0, A, k, x_start, x_end):
    Ny, Nx = rho.shape
    x = np.arange(Nx)[None, :]

    # finestra liscia (cosine taper)
    W = np.zeros_like(x, dtype=float)
    L = x_end - x_start

    mask = (x >= x_start) & (x <= x_end)
    W[mask] = 0.5 * (1 - np.cos(2*np.pi*(x[mask]-x_start)/L))

    rho_prime = A * np.sin(k*x) * W

    rho[:, :] = rho0 + rho_prime

    cs = 1/np.sqrt(3)
    ux[:, :] = 0.0
    uy[:, :] = 0.0

def initialize_local_plane_wave_with_flow(rho, ux, uy, rho0, A, k,
                                          x_start, x_end, U0):
    Ny, Nx = rho.shape
    x = np.arange(Nx)[None, :]

    W = np.zeros_like(x, dtype=float)
    L = x_end - x_start
    mask = (x >= x_start) & (x <= x_end)
    W[mask] = 0.5 * (1 - np.cos(2*np.pi*(x[mask]-x_start)/L))

    rho_prime = A * np.sin(k*x) * W
    rho[:, :] = rho0 + rho_prime

    cs = 1/np.sqrt(3)

    ux[:, :] = U0 + cs * rho_prime / rho0
    uy[:, :] = 0.0
    
    
# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
Nx, Ny = 20000, 80       # domain size
rho0 = 1.0
tau = 0.55              # relaxation time
omega = 1.0 / tau
U0 = 0.00577
# Acoustic pulse parameters
A = 1e-1               # amplitude (must be small)
sigma = 8.0            # width of pulse
x0 = Nx // 2
y0 = Ny // 2


sponge = True
Length_sponge = 5000

# =============================================================================
# ARRAYS
# =============================================================================
rho = np.ones((Ny, Nx)) * rho0
ux  = np.zeros((Ny, Nx))
uy  = np.zeros((Ny, Nx))

f     = np.zeros((Ny, Nx, 9))
f_new = np.zeros_like(f)

# Solid mask: channel walls top/bottom
solid_mask = np.zeros((Ny, Nx), dtype=np.uint8)
solid_mask[0, :]  = 1
solid_mask[-1, :] = 1

source_mask = np.zeros((Ny, Nx), dtype=np.bool_)
if sponge:
    source_mask[:, Length_sponge] = True
else:
    source_mask[:, 0] = True

eps = A
omega_src = (2*np.pi)/1400


# solid_mask = add_helmholtz_resonator(
#     solid_mask,
#     channel_height=Ny,
#     neck_width=6,
#     neck_length=10,
#     cavity_width=60,
#     cavity_height=100
# )

# =============================================================================
# SPONGE
# =============================================================================
if sponge:
    sponge_width = int(Length_sponge)*2/3
    tau_sponge = 4
    alpha = 0.0045
    

    tau_eff = tau * np.ones((Ny, Nx), dtype=np.float64)
    
    f_r0 = (np.exp(alpha*(Nx-sponge_width))/np.exp(alpha*(Nx-int(Length_sponge)*1/3)))
    f_l0 = (np.exp(-alpha*(sponge_width))/np.exp(-alpha*(int(Length_sponge)*1/3)))

    for j in range(Ny):
        for i in range(Nx):
           
            f_l = (np.exp(-alpha*i)/np.exp(-alpha*(int(Length_sponge)*1/3)))
            f_r = (np.exp(alpha*i)/np.exp(alpha*(Nx-int(Length_sponge)*1/3)))
            tau_eff[j, i] = tau + (tau_sponge) * (f_r-f_r0) + (tau_sponge) * (f_l-f_l0)
            

    for j in range(Ny):
        for i in range(Nx):
            dx_right = max(0, (Length_sponge-sponge_width) - (Nx - 1 - i))
            d =  dx_right
            
            if d > 0:
                tau_eff[j, i] = tau_sponge + tau
                
    for j in range(Ny):
        for i in range(Nx):
            dx_left = max(0, (Length_sponge-sponge_width) -  i)
            
            if dx_left > 0:
                tau_eff[j, i] = tau_sponge + tau
                    
    plt.plot(np.linspace(0,Nx,Nx),tau_eff[int(Ny/2),:])   
    plt.show()       
    omega_eff = 1.0 / tau_eff

else:
    tau_eff = tau * np.ones((Ny, Nx), dtype=np.float64)
    omega_eff = omega_eff = 1.0 / tau_eff
# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

# initialize_local_plane_wave_with_flow(
#     rho, ux, uy,
#     rho0=1.0,
#     A=1e-2,
#     k=2*np.pi/80,
#     x_start=0,
#     x_end=300,
#     U0=U0
# )
# uy[:, :] = 0.0

initialize_equilibrium(f, rho, ux, uy)

#omega_eff = np.full((Ny, Nx), omega)
# =============================================================================
# INITIALAZING FIELD FOR VISUALIZATION
# =============================================================================
x = np.linspace(0, (Nx - 1), Nx)
y = np.linspace(0, (Ny - 1), Ny)
X, Y = np.meshgrid(x, y, indexing='xy')   # shape (Ny, Nx)


points = np.c_[X.ravel(order='C'),
               Y.ravel(order='C'),
               np.zeros(X.size)]

grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [Nx, Ny, 1]

save_every = 20
# =============================================================================
# SIMULATION LOOP
# =============================================================================
nsteps = 2000
plot_every = 20
snap_id = 0
peak_positions = []
amplitudes = []
rho_point = []
c_s = 1/np.sqrt(3) + U0
src_pt = []
for it in range(nsteps):

    source_amp = eps * np.sin(omega_src * it)   # single-frequency

    lbm_step_modular(
        f, f_new, rho, ux, uy,
        cx_i, cy_i, w, opp,
        omega_eff,
        solid_mask,
        0, 0, 0.0, 1.0,
        0, 0, 2.0,
        True, False, 0.0, 0.0, False,   # periodic_x=True, periodic_y=False
        collision_operator=0,
        lambda_trt=3/16,
        use_source=True,
        source_mask=source_mask,
        source_amp=source_amp,
        src_nx=2.0,
        src_ny=0.0,
        cs=c_s
    )
    src_pt.append(source_amp)
    rho_point.append(rho[Ny//2,10])

    # swap
    f, f_new = f_new, f
    if it < 70:
        rho_line = rho[Ny//2, 1:]
        peak_positions.append(np.argmax(rho_line))
        amplitudes.append(np.max(np.abs(rho_line)))

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------
    if it % save_every == 0:
        field_saver(X, Y, grid, rho, rho0, ux, uy, case_name,snap_id)
        print(f"completed {np.float32((it/nsteps))*100:.2f}%")

    
    snap_id +=1



# =============================================================================
# DISPERSION ANALYSIS
# =============================================================================
time = np.linspace(0,len(peak_positions),len(peak_positions))

# linear fit of peak position vs time
coeffs = np.polyfit(time, peak_positions, 1)
c_num = coeffs[0]

c_s = 1/np.sqrt(3) + U0

print("Numerical wave speed:", c_num)
print("Lattice sound speed:", c_s)
print("Relative error:", (c_num - c_s)/c_s)

# =============================================================================
# ATTENUATION ANALYSIS
# =============================================================================
log_amp = np.log(amplitudes)
coeffs_att = np.polyfit(time, log_amp, 1)
alpha = -coeffs_att[0]

print("Attenuation coefficient alpha:", alpha)

# =============================================================================
#%%
plt.contourf(X,Y,rho,
cmap="RdBu_r",  # Colormap più contrastata
        levels=20,vmin=0.9,vmax=1.1,    # Meno livelli per aumentare il contrasto
        extend="max")# Normalizzazione
plt.show()

plt.plot(np.linspace(0,nsteps,nsteps),np.asarray(rho_point)[:])
plt.plot(np.linspace(0,nsteps,nsteps),np.asarray(src_pt))
plt.show()
  