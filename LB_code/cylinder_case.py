#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:54:15 2025

@author: angelo
"""

# run_cylinder.py
import numpy as np
import matplotlib.pyplot as plt
from lbm_kernel import lbm_step_generic, compute_vorticity
from geometry_creator import GeometryCreator
import pyvista as pv

# ==========================================
# PARAMETRI FISICI
# ==========================================
nvx = 1                                                                        # nodi per R_m (risoluzione sul raggio)
R_m = 0.01                                                                     # raggio cilindro [m]

Cx = R_m / nvx                                                                  # [m per cella]
    
U_phys = 10.0                                                                   # [m/s]
Lx = 2                                                                         # dominio x [m]
Ly = 0.05                                                                      # dominio y [m]
t_final = 0.25                                                                 # tempo simulazione [s]

# ==========================================
# GEOMETRIA
# ==========================================
wall_y_bottom = 0
wall_y_top = Ly

# tripping sulla parete inferiore
trip_x_start = 0.03                                                             # [m] da inlet
trip_width   = 0.005                                                            # [m]
trip_height  = 0.003                                                            # [m]

# ==========================================
# DOMINIO IN LATTICE
# ==========================================
Nx = int(Lx / Cx)                                                              # celle in x
Ny = int(Ly / Cx)                                                              # celle in y

U_in = 0.1
rho0 = 1.0

Cu = U_phys / U_in
Ct = Cx / Cu                                                                   # dt fisico per time step

nu_phys = 1.5e-5                                                               # m^2/s
nu_lat  = nu_phys / (Cu * Cx)
tau_base = 0.5 + 3.0 * nu_lat

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
               height_m=trip_height,
               bottom=True)

# ==========================================
# CAMPI
# ==========================================
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
sponge_width = 60
tau_sponge   = 1.5
alpha        = 4.0

tau_eff = tau_base * np.ones((Ny, Nx), dtype=np.float64)
for j in range(Ny):
    for i in range(Nx):
        dx_left   = max(0, sponge_width - i)
        dx_right  = max(0, sponge_width - (Nx-1 - i))
        dy_bottom = max(0, sponge_width - j)
        dy_top    = max(0, sponge_width - (Ny-1 - j))
        d = max(dx_left, dx_right, dy_bottom, dy_top)
        if d > 0:
            r = d / sponge_width
            f_r = (np.exp(alpha * r) - 1.0) / (np.exp(alpha) - 1.0)
            tau_eff[j, i] = tau_base + (tau_sponge - tau_base) * f_r

omega_eff = 1.0 / tau_eff

# ==========================================
# INIZIALIZZAZIONE PyVista MULTIBLOCK
# ==========================================
multiblock = pv.MultiBlock()
times = []
snap_id = 0

print(f"Inizio simulazione: Nx={Nx}, Ny={Ny}, nt={nt}, save_every={save_every}")

# ==========================================
# LOOP TEMPORALE
# ==========================================
for it in range(nt):
    lbm_step_generic(f, f_new, rho, ux, uy,
                     cx_i, cy_i, w, opp,
                     omega_eff,
                     solid_mask,
                     True,  0,    U_in, rho0,
                     True,  Nx-1, rho0)

    f, f_new = f_new, f

    if it % save_every == 0:
        p = rho - rho0
        vort = compute_vorticity(ux, uy)

        # === CREA NUOVO GRID PER OGNI SNAPSHOT (CRUCIALE) ===
        grid = pv.ImageData()
        grid.dimensions = [Nx, Ny, 1]
        grid.origin = [0, 0, 0]
        grid.spacing = [Cx, Cx, 1.0]
        
        # === ASSEGNA DATI CON flatten(order='F') ===
        grid.point_data['ux'] = ux.flatten(order='F')
        grid.point_data['uy'] = uy.flatten(order='F')
        grid.point_data['pressure'] = p.flatten(order='F')
        grid.point_data['vorticity'] = vort.flatten(order='F')
        
        # === AGGIUNGI AL MULTIBLOCK ===
        multiblock.append(grid)
        times.append(it * Ct)
        
        print(f"Saved snapshot {snap_id} at it={it}, t_phys={it*Ct:.4f}s")
        snap_id += 1

# ==========================================
# SALVATAGGIO FINALE
# ==========================================
multiblock.save('lbm_simulation.vtm')
print(f"✅ Salvato lbm_simulation.vtm con {len(times)} time steps!")

# ==========================================
# VERIFICA DATI (opzionale)
# ==========================================
reader = pv.read('lbm_simulation.vtm')
print("📊 Campi disponibili:", list(reader[0].point_data.keys()))
print("⏱️  Numero time steps:", len(reader))
print("✅ File pronto per ParaView!")

#%%
# ==========================================
# POST-PROCESS: PLOT DOPO LA SIMULAZIONE
# ==========================================
# carica dati
data = np.load("../lbm_cylinder_data.npz")
ux_hist   = data["ux_hist"]
p_hist    = data["p_hist"]
vort_hist = data["vort_hist"]
save_every = int(data["save_every"])

Nt_save, Ny, Nx, _ = ux_hist.shape

# figura iniziale
fig, axes = plt.subplots(1, 3, figsize=(15,4))

# inizializza con il primo snapshot
k0 = 0
ux_plot = ux_hist[k0, :, :, 0]
uy_plot = ux_hist[k0, :, :, 1]
vorticity = vort_hist[k0,:,:]
p_plot  = p_hist[k0]

im0 = axes[0].imshow(ux_plot, origin="lower", cmap="turbo",vmin=-0.005, vmax=0.2)
axes[0].set_title(f"u_x, t={k0*save_every}")
plt.colorbar(im0, ax=axes[0])

umag_plot = np.sqrt(uy_plot)
im1 = axes[1].imshow(umag_plot, origin="lower", cmap="bwr")
axes[1].set_title(f"|u|, t={k0*save_every}")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(vorticity, origin="lower", cmap="bwr")
axes[2].set_title(f"p, t={k0*save_every}")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.pause(0.1)

# loop di "animazione" offline
for k in range(1, Nt_save):
    ux_plot = ux_hist[k, :, :, 0]
    uy_plot = ux_hist[k, :, :, 1]
    vorticity  = vort_hist[k]
    t_plot = k * save_every

    umag_plot = np.sqrt(ux_plot**2+uy_plot**2)

    im0.set_data(ux_plot)
    im1.set_data(umag_plot)
    im2.set_data(vorticity)

    axes[0].set_title(f"u_x, t={t_plot}")
    axes[1].set_title(f"|u|, t={t_plot}")
    axes[2].set_title(f"vort_hist, t={t_plot}")
    
    
    plt.pause(0.05)   # tempo tra frame
    plt.draw()

plt.show()
