#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:10:10 2025

@author: angelo
"""

import functions_pp as export 
import numpy as np
from scipy.io import savemat
import concurrent.futures
import logging
import os

delta_x = 2.4300e-05
delta_y = 2.4300e-05


script_name = 'Extract_contour'
cavity = 'first_cavity'
tipe = 'contour'
variables  = ['Pressure','XVelocity','YVelocity','ZVelocity']
# analysis='Mean'
analysis='Sample'
path_points = './'
file_name = 'prova'

path = '/media/angelo/Volume/22_cavities/sharp/145/40vx/no_flow/1400/'
plane_name = 'ac_plane1_stream_inst.snc'
save_path = '/media/angelo/Volume/22_cavities/sharp/145/40vx/no_flow/1400/'

def execute_analysis(x_min, x_max, y_min, y_max, delta_x, delta_y, lined_flow_only):
    # Creazione della directory e dello script 2D rake
    lined_flow_only.create_date_directory()
    lined_flow_only.create_script_2D_rake(tipe, x_min, x_max, y_min, y_max, z_value, delta_x, delta_y)
    
    # Esecuzione dei comandi PowerAcoustics
    lined_flow_only.execute_commands_poweracoustics()

lined_flow_only = export.ScriptGenerator(path, variables, path_points, file_name, script_name, cavity, plane_name,save_path,analysis)


x_min =  0.065454
x_max =  0.06895
y_min = -0.0023
y_max = 0.0029
z_value = 0.006224
lined_flow_only.create_date_directory()
lined_flow_only.create_script_2D_rake(tipe,x_min,x_max,y_min,y_max,z_value,delta_x,delta_y)

lined_flow_only.execute_commands_poweracoustics()

#%% SORTER
import sorting_file as sorter

save_path = '/media/angelo/Volume/22_cavities/sharp/145/40vx/no_flow/1400/'
data_files = [f'{analysis}_Pressure_{cavity}_cavity_{tipe}.txt',f'{analysis}_XVelocity_{cavity}_cavity_{tipe}.txt',f'{analysis}_YVelocity_{cavity}_cavity_{tipe}.txt',f'{analysis}_ZVelocity_{cavity}_cavity_{tipe}.txt']


data = sorter.Contour_Analysis(save_path, data_files)

# Parallelizzazione
for i, filename in enumerate(data_files):
    logging.info("Elaborazione del file %s iniziata.", filename)
    data.import_data_for_time(save_path, filename)
    logging.info("Elaborazione del file %s completata.", filename)

data.X_unique = np.flip(data.X_unique)
data.Y_unique = np.flip(data.Y_unique)

# Prepara i dati
data_to_export = {
    'X_unique': np.asarray(data.X_unique),
    'Y_unique': np.asarray(data.Y_unique),
    'Pressure': np.asarray(data.Pressure_time),
    'XVelocity':np.asarray(data.XVelocity_time),
    'YVelocity': np.asarray(data.YVelocity_time),
    'ZVelocity':np.asarray(data.ZVelocity_time)
    #'ExtraData': {'nested_key1': 1, 'nested_key2': [1, 2, 3]}  # Dizionario annidato
}

# Appiattisci eventuali dizionari annidati
flattened_data = sorter.flatten_dict(data_to_export)

filename = os.path.join(save_path, f'M0_1400_145_up_{cavity}.mat')

savemat(filename,data_to_export)