#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:54:15 2025

@author: angelo
"""

# run_cylinder.py
import numpy as np
import matplotlib.pyplot as plt
from lbm_kernel import lbm_step_modular, compute_vorticity
from geometry_creator import GeometryCreator
from Check_Point_loader import Check_Point
from Check_Point_Maker import Maker_Ckpt
import pyvista as pv
import os
import time
# =========================================
# CASE NAME 
# =========================================

case_name = 'cylinder_second_order'

resume = False

ckpt_file = '/Users/angelo/Desktop/PhD/Aeroacoustic/LB_code/channel_coarse/channel_coarse_ckpt.npz'

ckpt_at_end = True

# ==========================================
# PARAMETRI FISICI
# ==========================================

nvx = 10                                                                        # nodi per R_m (risoluzione sul raggio)
R_m = 0.001                                                                    # raggio cilindro [m]

Cx = R_m / nvx                                                                 # [m per cella]
    
U_phys = 10.0                                                                  # [m/s]
Lx = 0.5                                                                       # dominio x [m]
Ly = 0.05                                                                       # dominio y [m]
t_final = 1                                                                    # tempo simulazione [s]

f = 2500                                                                       #frequenza minima assorbita
Length_sponge = 340/f                                 

# ==========================================
# turbulent channel paramenters
# ==========================================

Re = U_phys*Ly/1.5e-05

u_tau = 0.05*U_phys

tau_w = u_tau**2*1.225

dpdx = -tau_w/Ly/2
                                                                          
# ==========================================
# GEOMETRIA
# ==========================================
wall_y_bottom = 0
wall_y_top = Ly

# tripping sulla parete inferiore
trip_x_start = Length_sponge + 0.03                                            # [m] da inlet
trip_width   = 0.005                                                           # [m]
trip_height  = 0.003                                                           # [m]


# ==========================================
# Boundary condition
# ==========================================

BC_NONE        = 0
BC_ZOU_HE_VEL  = 1
BC_CONVECTIVE = 2
BC_PERIODIC   = 3
BC_BB_WALL    = 4

######################### DO NOT MODIFY FROM HERE #############################

# ==========================================
# GESTIONE PATH
# ==========================================

local_path = os.curdir
path = local_path+'/'+case_name

os.makedirs(path,exist_ok=True)

os.chdir(path)

# ==========================================
# DOMINIO IN LATTICE
# ==========================================
Nx = int(Lx / Cx)                                                              # celle in x
Ny = int(Ly / Cx)                                                              # celle in y

print(f"{Nx}")
print(f"{Ny}")

U_in = 0.1
rho0 = 1.0

Cu = U_phys / U_in
Ct = Cx / Cu                                                                   # dt fisico per time step

nu_phys = 1.5e-5                                                               # m^2/s
nu_lat  = nu_phys / (Cu * Cx)
tau_base = 0.6 + 3.0 * nu_lat
print("WARNING: ", tau_base)

Fy = 0.0  
Fx_phys = -dpdx / 1.225
Fx = Fx_phys * (Ct**2 / Cx)

nt = int(t_final/Ct)

save_every = 100  # Modificato da 500 per più frame
Nt_save = int(nt/save_every)

# D2Q9
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1],
              [1,1], [-1,1], [-1,-1], [1,-1]], dtype=np.int64)

cx_i = c[:,0]
cy_i = c[:,1]

w = np.array([4/9,
              1/9,1/9,1/9,1/9,
              1/36,1/36,1/36,1/36], dtype=np.float64)

opp = np.array([0,3,4,1,2,7,8,5,6], dtype=np.int64)

# ==========================================
# CONVERSIONE GEOMETRIA
# ==========================================
solid_mask = np.zeros((Ny, Nx), dtype=np.uint8)
geom = GeometryCreator(Cx)

geom.add_wall_y(solid_mask, y_m=wall_y_bottom)
geom.add_wall_y(solid_mask, y_m=wall_y_top)
geom.add_block(solid_mask,
               x_start_m=trip_x_start,
               width_m=trip_width,
               height_m=trip_height,tripper="triangle",
               position='both')

# geom.add_cylinder(solid_mask, Nx//5, Ny//1.25, R_m*2)
# geom.add_cylinder(solid_mask, Nx//5, Ny//2, R_m*2)
# geom.add_cylinder(solid_mask, Nx//5, Ny//4, R_m*2)

# ==========================================
# CREATE MESH
# ==========================================

x = np.linspace(0, (Nx-1)*Cx, Nx)
y = np.linspace(0, (Ny-1)*Cx, Ny)
X, Y = np.meshgrid(x, y, indexing='xy')   # shape (Ny, Nx)

# ==========================================
# CAMPI
# ==========================================

if resume:
    check_point = Check_Point(ckpt_file, X,Y,Cx)
    [rho,ux,uy,f] = check_point.interpolate(solid_mask,cx_i,cy_i,w)
    f_new = np.zeros_like(f)
 
    
else:
    rho = rho0 * np.ones((Ny, Nx), dtype=np.float64)
    ux  = np.zeros((Ny, Nx), dtype=np.float64)
    uy  = np.zeros((Ny, Nx), dtype=np.float64)
    
    f     = np.zeros((Ny, Nx, 9), dtype=np.float64)
    f_new = np.zeros_like(f)
    for i in range(9):
        f[:,:,i] = w[i] * rho0

# ==========================================
# SPONGE: tau_eff(x,y) esponenziale ai bordi
# ==========================================
sponge = None

if sponge:
    sponge_width = int(Length_sponge/Cx)
    tau_sponge   = 1.5
    alpha        = 5.0
    
    tau_eff = tau_base * np.ones((Ny, Nx), dtype=np.float64)
    for j in range(Ny):
        for i in range(Nx):
            dx_left   = max(0, sponge_width - i)
            dx_right  = max(0, sponge_width - (Nx-1 - i))
            # dy_bottom = max(0, sponge_width - j)
            # dy_top    = max(0, sponge_width - (Ny-1 - j))
            d = max(dx_left, dx_right)
            if d > 0:
                r = d / sponge_width
                f_r = (np.exp(alpha * r) - 1.0) / (np.exp(alpha) - 1.0)
                tau_eff[j, i] = tau_base + (tau_sponge - tau_base) * f_r
    
    omega_eff = 1.0 / tau_eff
else:
    tau_eff = tau_base * np.ones((Ny, Nx), dtype=np.float64)
    omega_eff =     omega_eff = 1.0 / tau_eff



# ==========================================
# INIZIALIZZAZIONE PyVista MULTIBLOCK
# ==========================================

multiblock = pv.MultiBlock()
times = []
snap_id = 0

print(f"Inizio simulazione: Nx={Nx}, Ny={Ny}, nt={nt}, save_every={save_every}")
start_time = time.time()
# ==========================================
# LOOP TEMPORALE
# ==========================================

for it in range(nt):
    lbm_step_modular(
            f, f_new, rho, ux, uy,
            cx_i, cy_i, w, opp,
            omega_eff,
            solid_mask,
        
            BC_NONE, 0, 0.0, rho0,      # inlet
            BC_NONE, 0, rho0,           # outlet
            True, True, Fx, Fy)                        # periodic_x

    f, f_new = f_new, f

    if it % save_every == 0:
        p = rho - rho0
        vort = compute_vorticity(ux, uy)

        # crei una surface strutturata (z=0)
        points = np.c_[X.ravel(order='C'),
                       Y.ravel(order='C'),
                       np.zeros(X.size)]
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = [Nx, Ny, 1]  
        
        # campi (usa lo stesso ordine di ravel)
        grid.point_data['rho']       = rho.ravel(order='C')
        grid.point_data['ux']        = ux.ravel(order='C')*Cu
        grid.point_data['uy']        = uy.ravel(order='C')
        grid.point_data['pressure']  = p.ravel(order='C')
        grid.point_data['vorticity'] = vort.ravel(order='C')

        filename = f"t{case_name}_{snap_id:04d}.vts"
        grid.save(filename)
        times.append(it*Ct)
        
        print(f"Saved snapshot {snap_id} at it={it}, t_phys={it*Ct:.8f}s")
        snap_id += 1

    if ckpt_at_end and it == nt-1:
        
        np.savez(f"{case_name}_ckpt.npz", f=f_new, rho=rho, ux=ux, uy=uy)

        

# ==========================================
# SALVATAGGIO FINALE
# ==========================================
print(f"Salvato lbm_simulation.vtm con {len(times)} time steps!")
end_time = time.time()
print(f"Tempo di simulazione: {end_time-start_time}")


