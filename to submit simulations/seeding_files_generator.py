#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 11:29:40 2025

@author: angelo
"""

import os
import numpy as np

def calcola_parametri_onda(frequencies, source, Mach, T_celsius=25):
    # Costanti
    gamma = 1.4
    mol_weight = 28.9645
    rgas_universal = 8.314462
    sutherland_muref = 1.716e-5
    sutherland_tref = 273.15
    sutherland_s = 110.4
    inlet_pressure = 101325

    inlet_temperature = sutherland_tref + T_celsius
    rgas = rgas_universal / mol_weight * 1e3
    cp = rgas * gamma / (gamma - 1)
    inlet_density = inlet_pressure / (rgas * inlet_temperature)
    viscosity_dyn = sutherland_muref * ((inlet_temperature / sutherland_tref) ** 1.5) * \
                    (sutherland_tref + sutherland_s) / (inlet_temperature + sutherland_s)
    viscosity = viscosity_dyn / inlet_density
    soundspeed = np.sqrt(gamma * rgas * inlet_temperature)

    zigzag = 500e-3
    origin = 5

    frequencies = np.array(frequencies)
    optydB_wavelength = soundspeed/frequencies
    up_source_wavelength = soundspeed * (1 + Mach) / frequencies
    down_source_wavelength = soundspeed * (1 - Mach) / frequencies

    if source.lower() == 'up':
        OPTYDB_waves = np.floor((origin - zigzag) / up_source_wavelength)
        wavelengths = optydB_wavelength
    else:
        OPTYDB_waves = np.floor((origin - zigzag - 136.906e-3) / down_source_wavelength)
        wavelengths = optydB_wavelength

    return OPTYDB_waves.astype(int), wavelengths

# Parametri di input
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

source = 'up'  # 'up' or 'down'
frequencies = [800, 1400, 2000]
Mach = 0.3
SPL = 130

# Calcolo parametri onda
num_sin_waves, wavelengths = calcola_parametri_onda(frequencies, source, Mach)

# Loop per creazione file
for i in range(len(frequencies)):
    freq = frequencies[i]
    wl = wavelengths[i]
    nsin = num_sin_waves[i]

    if source == 'down':
        correction = 2000
    else:
        correction = 1000
    file_name = f"seed_{source}_{SPL}_{Mach}_{freq}.fnc"

    optydb_fieldmod = f"""../flow-NASA_chamfer_22_cavitie_run_fine.ckpt.fnc ....... Name of the imported solution file
109592.2 ......................... Mean-flow pressure (Pa)
1.283355 .......................... Mean-flow denisty (kg/m^3)
1.4 .............................. Gas specific heats ratio
287.057 .......................... Gas constant
0.2893652 ........................ Mean-flow Mach number along x
0.0 .............................. Mean-flow Mach number along y
0.0 .............................. Mean-flow Mach number along z
# FIELD CORRECTION SECTION =======
{file_name} ......................... Name of the exported solution file
optydb_fieldmod_param_{freq}.i .......... Name of the user-defined parameter input file
{correction} ............................. User-defined field correction index
"""

    optydb_fieldmod_param = f"""5 ....................................... Number of parameters
1 ....................................... Number of wavepackets
0.5 .............................. Central position of the truncated sinus wave at the initial time (m)
{wl:.9f} .............................. Wavelength of the truncated sinus wave (m)
{SPL} ............................ SPL of the truncated sinus wave (dB)
{nsin} .............................. Number of sinus waves
"""

    # Salvataggio file
    main_filename = f"optydb_fieldmod_{freq}.i"
    main_path = os.path.join(output_dir, main_filename)
    with open(main_path, "w") as f:
        f.write(optydb_fieldmod)

    param_filename = f"optydb_fieldmod_param_{freq}.i"
    param_path = os.path.join(output_dir, param_filename)
    with open(param_path, "w") as f:
        f.write(optydb_fieldmod_param)

    print(f"Creati: {main_path} e {param_path}")
