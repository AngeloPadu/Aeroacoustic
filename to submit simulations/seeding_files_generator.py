#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 11:29:40 2025

@author: angelo
"""

import os

# Directory di output
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

source = 'down'
frequencies = [800, 1400, 2000]
Mach = 0.3
SPL = 130

# Parametri variabili
wavelengths = [0.432688642933482, 0.247250653104847, 0.173075457173393]
num_sin_waves = [14, 25, 36]

for i in range(len(frequencies)):
    freq = frequencies[i]
    wl = wavelengths[i]
    nsin = num_sin_waves[i]
    
    if source == 'down':
       correction = 2000
    else:
       correction = 1000
    file_name = f"seed_{source}_{SPL}_{Mach}_{freq}.fnc"

    # Template del contenuto base
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
{wl} .............................. Wavelength of the truncated sinus wave (m)
{SPL} ............................ SPL of the truncated sinus wave (dB)
{nsin} .............................. Number of sinus waves
"""

    # Scrittura del file principale
    main_filename = f"optydb_fieldmod_{freq}.i"
    main_path = os.path.join(output_dir, main_filename)
    with open(main_path, "w") as f:
        f.write(optydb_fieldmod)

    # Scrittura del file parametri onda
    param_filename = f"optydb_fieldmod_param_{freq}.i"
    param_path = os.path.join(output_dir, param_filename)
    with open(param_path, "w") as f:
        f.write(optydb_fieldmod_param)

    print(f"Creati: {main_path} e {param_path}")
