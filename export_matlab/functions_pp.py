#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:16:48 2025

@author: angelo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:59:00 2024

@author: angelo
"""

import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import scipy.optimize as spOpt
from matplotlib.patches import Rectangle
import math  # Per usare il valore NaN
import re
import csv
from scipy.signal import csd
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import subprocess
import concurrent.futures
import scipy.signal as signal
import scipy.io
from scipy.io import savemat
import datetime
from sklearn.metrics import mean_squared_error
from scipy.optimize import fsolve
from IPython.display import clear_output
from scipy import interpolate
from scipy.signal import butter, filtfilt, csd

class CrossCorrelationAnalyzer:
    def __init__(self, data_path, velocity_file, points_file):
        """
        Initialize the CrossCorrelationAnalyzer.

        :param data_path: Path to the data files.
        :param velocity_file: Name of the file containing velocity data.
        :param points_file: Name of the file containing points data.
        """
        self.data_path = data_path
        self.velocity_file = velocity_file
        self.points_file = points_file
        self.data = None
        self.punti = None
        self.correlated = None
        self.lags = None
        self.dt = None

    def load_data(self):
        """Load the velocity and point data from the specified files."""
        self.punti = np.loadtxt(self.data_path + self.points_file, skiprows=1)[:, 0]
        self.data = np.loadtxt(self.data_path + self.velocity_file)
        self.correlated = np.zeros((self.data.shape[0] * 2 - 1, self.data.shape[1]))
        self.lags = np.copy(self.correlated)
        self.dt = self.data[1, 0] - self.data[0, 0]

    def xcorr(self, x, y):
        """
        Perform Cross-Correlation on x and y.
        
        :param x: 1st signal.
        :param y: 2nd signal.
        :return: lags and coefficients of correlation.
        """
        corr = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(len(x), len(y), mode="full")
        return lags, corr

    def compute_correlations(self):
        """Compute cross-correlations for each point."""
        for i in range(1, self.correlated.shape[1], 10):
            print(f"Processing column: {i}")
            
            # Perform cross-correlation between column 1 (reference) and column i
            lags, self.correlated[:, i] = self.xcorr(
                self.data[:, i] - self.data[:, i].mean(),
                self.data[:, 1] - self.data[:, 1].mean()
            )
            punto = self.punti[i - 1] - self.punti[0]
            self.lags[:, i] = punto / (lags * self.dt)
            
            # Find the lag corresponding to the maximum correlation
            lag_index = np.argmax(self.correlated[:, i]) - (len(self.data[:, 1]) - 1)
            lag_time = lag_index * self.dt  # Convert to time units
            print(f"Lag for column {i}: {lag_time} seconds")

    def plot_correlations(self):
        """Plot the computed cross-correlations."""
        plt.ion()  # Interactive mode
        plt.figure(figsize=(10, 6))

        # Plot only the relevant correlations (skip empty ones)
        for i in range(1, self.correlated.shape[1], 10):
            plt.plot(self.lags[:, i], self.correlated[:, i], label=f'punto {i}')

        # Graph settings
        plt.xlabel('Lags (Tempo [s])')
        plt.ylabel('Correlazione')
        plt.xlim([0, 110])
        plt.ylim([0, 70100])
        plt.legend()
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        plt.show()
        
class SPLCenterline:
    def __init__(self, ac_source, geom, SPL, frequencies, resolution, time_window=(600, 5000)):
        self.ac_source = ac_source
        self.geom = geom
        self.SPL = SPL
        self.frequencies = frequencies
        self.resolution = resolution
        self.time_window = time_window
        
    def load_data(self, flow=True):
        """
        Carica i dati del file SPL_centerline_pressure in base alla presenza o meno di flusso.
        """
        if flow:
            path = '/media/angelo/results/Progetto_ANEMONE/Cases/0-TUD_UFSC/NoSlip_NoSlip Cases/Liner NASA/{}/{}/fine/{}/{}/SPL_centerline/'.format(
                self.ac_source, self.geom, self.SPL, self.frequencies[0]
            )
        else:
            path = '/media/angelo/results/Progetto_ANEMONE/Cases/0-TUD_UFSC/NoSlip_NoSlip Cases/Liner NASA/noflow/{}/fine/{}/{}/SPL_centerline/'.format(
                self.geom, self.SPL, self.frequencies[0]
            )

        try:
            P_centerline = np.loadtxt(path + 'SPL_centerline_pressure')
        except:
            P_centerline = np.loadtxt(path + 'SPL_centerline_pressure.txt')

        P_time = P_centerline[:, 0]
        
        if flow:
            P_data_fine = P_centerline[self.time_window[0]:self.time_window[1], 1:]
        else:
            P_data_fine = P_centerline[self.time_window[0]:self.time_window[1], 1:]
        
        fs = 1 / (P_time[1] - P_time[0])
        nperseg = len(P_data_fine)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.plot(P_data_fine)
        
        return P_time, P_data_fine, fs, nperseg

    def calculate_SPL(self, P_data_fine, fs, nperseg):
        """
        Calcola il livello di pressione sonora (SPL) per la linea centrale.
        """
        SPL_centerline_fine = np.zeros(np.shape(P_data_fine)[1])
        
        for i in range(len(SPL_centerline_fine)):    
            f, Sxx = csd(P_data_fine[:, i], P_data_fine[:, i], fs=fs, nperseg=nperseg, scaling='spectrum')
            f_peak_idx = np.where(Sxx == np.max(Sxx))
            SPL_centerline_fine[i] = 20 * np.log10(np.sqrt(np.abs(Sxx[f_peak_idx])) / 2e-5)
        
        return f, Sxx, f_peak_idx, SPL_centerline_fine
    
    def calculate_OASPL(self, P_data_fine, fs, nperseg):
        """
        Calcola l'Overall Sound Pressure Level (OASPL) integrando su tutto lo spettro di frequenze.
        """
        OASPL_centerline = np.zeros(np.shape(P_data_fine)[1])
        p0 = 20e-6  # Pressione di riferimento (20 μPa)

        for i in range(len(OASPL_centerline)):    
            # Calcola la densità spettrale di potenza (Sxx)
            f, Sxx = csd(P_data_fine[:, i], P_data_fine[:, i], fs=fs, nperseg=nperseg, scaling='spectrum')
            
            # Integra su tutto lo spettro per ottenere la pressione sonora totale
            p_rms_squared = np.trapz(Sxx, f)  # Integrale della densità spettrale su f
            
            # Calcola OASPL
            OASPL_centerline[i] = 10 * np.log10(p_rms_squared / p0**2)
        
        self.OASPL_centerline = OASPL_centerline

    def plot_spectrum(self, f, Sxx, f_peak_idx, title_suffix=""):
        """
        Plot dello spettro calcolato.
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.plot(f, 20 * np.log10(np.sqrt(np.abs(Sxx)) / 2e-5))
        ax.set_yscale('log')
        ax.axvline(f[f_peak_idx], color='r', linestyle='dashed')
        ax.set_xlim([100, 30000])
        ax.set_xlabel('Frequency [Hz]', fontsize=30)
        ax.set_title('Spectrum of {}'.format(self.resolution + title_suffix))
        plt.show()

    def process_case(self, flow=True):
        """
        Esegue tutte le operazioni per il caso con o senza flusso.
        """
        P_time, P_data_fine, fs, nperseg = self.load_data(flow=flow)
        f, Sxx, f_peak_idx, SPL_centerline = self.calculate_SPL(P_data_fine, fs, nperseg)
        OASPL = self.calculate_OASPL(P_data_fine, fs, nperseg)
        self.plot_spectrum(f, Sxx, f_peak_idx, title_suffix=" (Flow)" if flow else " (No Flow)")
        
    def plot_OASPL_decay(self, other_cases=[]):
        """
        Plotta il decadimento di SPL in funzione della distanza per più casi.
        """
        plt.figure(figsize=(8, 8))  # Imposta le dimensioni del grafico
        distanze_fine = np.linspace(0, 1, len(self.OASPL_centerline))
        
        # Plot del primo caso (il caso corrente)
        plt.plot(distanze_fine, self.OASPL_centerline, label='SPL = {}'.format(self.SPL), marker='o')

        # Plotta altri casi (altre istanze della classe SPLCenterline)
        for case in other_cases:
            distanze_fine = np.linspace(0, 1, len(case.OASPL_centerline))
            plt.plot(distanze_fine, case.OASPL_centerline, label='SPL = {}'.format(case.SPL), marker='s')
        
        plt.xlabel('$x/L$', fontsize=30)
        plt.ylabel('$OASPL [dB]$', fontsize=30)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.show()
               
class FileGenerator:
    def generate_line_points(self, x_value):
        """
        Genera una lista di punti (x, y, z) con x fisso e z fisso a 0.006223.
        """
        self.x_value = x_value
        y_values = [
            -0.039999999, -0.039902765, -0.039805532, -0.039708298, -0.039611064,
            -0.03951383, -0.039416596, -0.039319359, -0.039222125, -0.039124891,
            -0.039027657, -0.038930427, -0.03883319, -0.038735952, -0.038638718,
            -0.038541485, -0.038444251, -0.038347021, -0.038249783, -0.038152546,
            -0.038055312, -0.037958078, -0.037860844, -0.03776361, -0.03766638,
            -0.037471905, -0.037277438, -0.03708297, -0.036888499, -0.036694031,
            -0.036499564, -0.036305092, -0.036110625, -0.035916157, -0.035721686,
            -0.035527218, -0.035332751, -0.034943812, -0.034554876, -0.034165937,
            -0.033776999, -0.033388063, -0.032999124, -0.032610189, -0.03222125,
            -0.031832311, -0.031443376, -0.031054437, -0.0306655, -0.029887624,
            -0.02910975, -0.028331876, -0.027554, -0.026776128, -0.02599825,
            -0.025220376, -0.0244425, -0.023664625, -0.022886749, -0.022108875,
            -0.021330999, -0.019775251, -0.018219501, -0.017441625, -0.016663751,
            -0.015885875, -0.015108, -0.014330124, -0.01355225, -0.012774375,
            -0.0119965, -0.011218625, -0.01044075, -0.010051812, -0.009662875,
            -0.009273938, -0.008885, -0.008496063, -0.008107125, -0.007718187,
            -0.00732925, -0.006940313, -0.006551376, -0.006162438, -0.0057735,
            -0.005579031, -0.005384563, -0.005190094, -0.004995625, -0.004801156,
            -0.004606687, -0.004412219, -0.00421775, -0.004023281, -0.003828812,
            -0.003634344, -0.003439875, -0.003342641, -0.003245407, -0.003148172,
            -0.003050938, -0.002953703, -0.002856469, -0.002759234, -0.002662,
            -0.002564766, -0.002467531, -0.002370297, -0.002273062, -0.002224445,
            -0.002175828, -0.002127211, -0.002078594, -0.002029977, -0.001981359,
            -0.001932742, -0.001884125, -0.001835508, -0.001786891, -0.001738274,
            -0.001689656, -0.001641039, -0.001592422, -0.001543805, -0.001495188,
            -0.00144657, -0.001397953, -0.001349336, -0.001300719, -0.001252102,
            -0.001203484, -0.001154867, -0.00110625, -0.001057633, -0.001009016,
            -0.000960398, -0.000911781, -0.000863164, -0.000814547, -0.00076593,
            -0.000717313, -0.000668695, -0.000620078, -0.000571461, -0.000522844,
            -0.000474227, -0.000425609, -0.000376992, -0.000328375, -0.000279758,
            -0.000231141, -0.000182523, -0.000133906, -8.53e-05, -3.67e-05, -1.27e-09
        ]
        z_value = 0.006223

        # Crea una lista di tuple (x, y, z) con x fisso e z fisso
        points = [(self.x_value, y, z_value) for y in y_values]
        return points

    def write_line_to_file(self, filename, x_value):
        """
        Scrive i punti generati con x fisso in un file.
        """
        points = self.generate_line_points(x_value)
        with open(filename, 'w') as f:
            f.write("length_unit = m\n")
            for point in points:
                f.write(f"{point[0]}\t{point[1]}\t{point[2]}\n")
        print(f"File '{filename}' generato con successo!")


    def generate_matrix(self, x_range, y_range, z_value, delta_x, delta_y):
        """
        Genera una matrice di punti (x, y, z) in base ai range di x e z e ai delta forniti.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_value = z_value
        self.delta_x = delta_x
        self.delta_y = delta_y
    
        x_values = [self.x_range[0] + i * self.delta_x for i in range(int((self.x_range[1] - self.x_range[0]) / self.delta_x) + 1)]
        y_values = [self.y_range[0] + i * self.delta_y for i in range(int((self.y_range[1] - self.y_range[0]) / self.delta_y) + 1)]
        
        matrix = []
        for x in x_values:
            for y in y_values:
                matrix.append((x, y, self.z_value))  # Usa z, non z_range
        return matrix

    
    def save_matrix_to_txt(self, path_points, filename, x_range, y_range, z_value, delta_x, delta_y):
       # Genera la matrice
        matrix = self.generate_matrix(x_range, y_range, z_value, delta_x, delta_y)

        # Assicurati che la directory esista
        os.makedirs(path_points, exist_ok=True)

        # Scrivi i punti nel file di output
        file_path = os.path.join(path_points, filename)  # Costruisci il percorso completo
        with open(file_path, 'w') as f:
            f.write("length_unit = m\n")  # Scrivi l'unità di lunghezza
            for point in matrix:
                f.write(f"{point[0]:.8f}\t{point[1]:.8f}\t{point[2]:.6f}\n")  # Scrivi i punti (x, y, z)
        
        print(f"File '{filename}' salvato in {file_path}.")                
class ScriptGenerator:
    
    def __init__(self, path, variables,path_points, points, script_name, cavity, plane_name, save_dir, analysis):
        self.path = path
        self.variables = variables  # Una lista di variabili
        self.path_points = path_points
        self.points = points
        self.script_name = script_name
        self.cavity = cavity
        self.plane_name = plane_name
        self.save_dir = save_dir
        self.analysis = analysis
    def create_date_directory(self):
        try:
            # Ottieni la data corrente nel formato 'YYYY-MM-DD_cavity'
            today = datetime.datetime.now().strftime(f"%Y-%m-%d_{self.cavity}")
            
            # Crea il percorso completo della nuova cartella
            self.current_dir = os.path.join(self.path, today)
            
            # Crea la cartella se non esiste già
            os.makedirs(self.current_dir, exist_ok=True)
            
            # Cambia la directory corrente a quella appena creata
            os.chdir(self.current_dir)
            
            print(f"Cartella creata: {self.current_dir} e cambiata come directory corrente.")
        except Exception as e:
            print(f"Errore nella creazione della directory: {e}")
    
    def create_script(self, tipe):
        try:
            newdir = os.path.join(self.path, "BDL_samples")
            os.makedirs(newdir, exist_ok=True)  # Crea la directory se non esiste

            # Crea una sottocartella chiamata 'script_name'
            self.sub_dir = os.path.join(self.current_dir, self.script_name)
            os.makedirs(self.sub_dir, exist_ok=True)
            
            # Cambia la directory corrente alla sottocartella
            os.chdir(self.sub_dir)

            for variable in self.variables:
                script_name = f"{variable}_analysis_script.py"  # Genera uno script per ogni variabile
                
                # Prepara il contenuto dello script
                script_content = f"""
import os

workdir = "{self.path}"
path_point = "{self.path_points}"
newdir = "{newdir}/"
save_dir = "{self.save_dir}"

project0 = app.newProject()
calculation0 = project0.createCalculation()
calculation0.calcFunction = "{self.analysis}"
calculation0.name = "BDL_sample_before"
calculation0.inputTab.variables[0].delete()

variable0 = calculation0.inputTab.createVariable()
calculation0.inputTab.variables[0].name = "{variable}"  # Usa la variabile attuale
calculation0.inputTab.filename = workdir + "/{self.plane_name}"
calculation0.outputTab.filename = "{self.current_dir}" + "/BDL_samples/{variable}.calc"
calculation0.outputTab.outputFormat = "ImportPoints"
calculation0.inputTab.autoSelectFrames = False
calculation0.inputTab.start = 4196
calculation0.inputTab.end = 5196
calculation0.outputTab.ptfileOptions.filename = path_point + "/{self.points}"
calculation0.apply()

exportData0 = project0.calculations["BDL_sample_before"].createExportData()
exportData0.name = "BDL_sample_pressure_{variable}"
exportData0.filename = save_dir + "/{variable}_{self.cavity}_{tipe}.txt"
exportData0.apply()

project0.save(workdir + "/BDL_samples/BDL_sample_{variable}.pap")
project0.queueAll(True)
"""
                # Scrivi il contenuto nello script file
                with open(script_name, 'w') as file:
                    file.write(script_content)
                print(f"Script creato: {script_name}")
        except Exception as e:
            print(f"Errore nella creazione degli script: {e}")
    
    def read_cavity_coordinates(self, cavity, cavity_file_name='/home/angelo/Scrivania/PhD/02.aeroacustica/postprocessing/BDL_points/cavity_point_data.csv'):
        with open(cavity_file_name, mode='r') as file:
            reader = csv.DictReader(file)  # Usa DictReader per leggere le righe come dizionari
            for row in reader:
                if row['cavity'] == cavity:
                    # Ritorna le coordinate della cavità trovata
                    return {
                        'x_min': float(row['x_min']),
                        'x_max': float(row['x_max']),
                        'y_min': float(row['y_min']),
                        'y_max': float(row['y_max']),
                        'z_value': float(row['z_value'])
                    }

    def create_script_2D_rake(self,tipe,x_min,x_max,y_min,y_max,z_value,delta_x,delta_y ):
        try:
            newdir = os.path.join(self.path, "BDL_samples")
            os.makedirs(newdir, exist_ok=True)  # Crea la directory se non esiste

            # Crea una sottocartella chiamata 'script_name'
            self.sub_dir = os.path.join(self.current_dir, self.script_name)
            os.makedirs(self.sub_dir, exist_ok=True)
            
            # Cambia la directory corrente alla sottocartella
            os.chdir(self.sub_dir)

            for variable in self.variables:
                script_name = f"{variable}_analysis_script.py"  # Genera uno script per ogni variabile
                
                # Prepara il contenuto dello script
                script_content = f"""
import os

workdir = "{self.path}"
path_point = "{self.path_points}"
newdir = "{newdir}/"
save_dir = "{self.save_dir}"
app.jobControls.jobMemoryLimit = 2000
project0 = app.newProject()
calculation0 = project0.createCalculation()
calculation0.calcFunction = "{self.analysis}"
calculation0.name = "BDL_sample_before"
calculation0.inputTab.variables[0].delete()

variable0 = calculation0.inputTab.createVariable()
calculation0.inputTab.variables[0].name = "{variable}"  # Usa la variabile attuale
calculation0.inputTab.filename = workdir + "/{self.plane_name}"
calculation0.outputTab.filename = "./{variable}.calc"
calculation0.outputTab.outputFormat = "Rake2D"
#calculation0.inputTab.autoSelectFrames = False
#calculation0.inputTab.start = 0
#calculation0.inputTab.end = 10498
#calculation0.inputTab.incrementVia = "NumFrames"
#calculation0.inputTab.increment = 8
calculation0.outputTab.rake2DOptions.cornerA.x = {x_min}
calculation0.outputTab.rake2DOptions.cornerA.y = {y_min}
calculation0.outputTab.rake2DOptions.cornerA.z = {z_value}
calculation0.outputTab.rake2DOptions.cornerB.x = {x_min}
calculation0.outputTab.rake2DOptions.cornerB.y = {y_max}
calculation0.outputTab.rake2DOptions.cornerB.z = {z_value}
calculation0.outputTab.rake2DOptions.cornerC.x = {x_max}
calculation0.outputTab.rake2DOptions.cornerC.y = {y_max}
calculation0.outputTab.rake2DOptions.cornerC.z = {z_value}

calculation0.outputTab.rake2DOptions.ABGridsize.sizeVia = "Delta"
calculation0.outputTab.rake2DOptions.ABGridsize.delta = {delta_y}
calculation0.outputTab.rake2DOptions.BCGridsize.sizeVia = "Delta"
calculation0.outputTab.rake2DOptions.BCGridsize.delta = {delta_x}
calculation0.outputTab.rake2DOptions.maxProjectionDistance = 0.1
calculation0.apply()
exportData0 = project0.calculations["BDL_sample_before"].createExportData()
exportData0.name = "BDL_sample_pressure_{variable}"
exportData0.filename = "./{self.analysis}_{variable}_{self.cavity}_cavity_{tipe}.txt"
exportData0.apply()

project0.save("./BDL_sample_{variable}.pap")
project0.queueAll(True)
"""
                # Scrivi il contenuto nello script file
                with open(script_name, 'w') as file:
                    file.write(script_content)
                print(f"Script creato: {script_name}")
                self.tipe = tipe
        except Exception as e:
            print(f"Errore nella creazione degli script: {e}")

    def run_poweracoustics(self, variable, poweracoustics_path):
        script_name = f"{variable}_analysis_script.py"  # Script creato per ogni variabile
        try:
            # Prepara il comando PowerAcoustics
            comando = (
                f'bash -c "source ~/.bashrc && export QT_QPA_PLATFORM=xcb && '
                f'{poweracoustics_path} -force -no_gui -wait -script {script_name} '
                f'> poweracoustics_output_{variable}.log 2> poweracoustics_error_{variable}.log"'
            )
            
            # Esegui il comando e cattura l'output
            result = subprocess.run(comando, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Errore durante l'esecuzione del comando PowerAcoustics per {variable}. Controlla 'poweracoustics_error_{variable}.log' per i dettagli.")
            else:
                comando = (f'cp {self.analysis}_{variable}_{self.cavity}_cavity_{self.tipe}.txt {self.save_dir}')
                result = subprocess.run(comando,shell=True) 
                print(f"Comando PowerAcoustics eseguito con successo per {variable}. Controlla 'poweracoustics_output_{variable}.log' per l'output.")
              	
        except Exception as e:
            print(f"Errore durante l'esecuzione dei comandi PowerAcoustics per {variable}: {e}")


    def create_script_2D_rake_z(self,tipe,z_min,z_max,y_min,y_max,x_value,delta_x,delta_y ):
        try:
            newdir = os.path.join(self.path, "BDL_samples")
            os.makedirs(newdir, exist_ok=True)  # Crea la directory se non esiste

            # Crea una sottocartella chiamata 'script_name'
            self.sub_dir = os.path.join(self.current_dir, self.script_name)
            os.makedirs(self.sub_dir, exist_ok=True)
            
            # Cambia la directory corrente alla sottocartella
            os.chdir(self.sub_dir)

            for variable in self.variables:
                script_name = f"{variable}_analysis_script.py"  # Genera uno script per ogni variabile
                
                # Prepara il contenuto dello script
                script_content = f"""
import os

workdir = "{self.path}"
path_point = "{self.path_points}"
newdir = "{newdir}/"
save_dir = "{self.save_dir}"
app.jobControls.jobMemoryLimit = 2000
project0 = app.newProject()
calculation0 = project0.createCalculation()
calculation0.calcFunction = "{self.analysis}"
calculation0.name = "BDL_sample_before"
calculation0.inputTab.variables[0].delete()

variable0 = calculation0.inputTab.createVariable()
calculation0.inputTab.variables[0].name = "{variable}"  # Usa la variabile attuale
calculation0.inputTab.filename = workdir + "/{self.plane_name}"
calculation0.outputTab.filename = "{self.current_dir}" + "/BDL_samples/{variable}.calc"
calculation0.outputTab.outputFormat = "Rake2D"
# calculation0.inputTab.autoSelectFrames = False
# calculation0.inputTab.start = 4196
# calculation0.inputTab.end = 5196
calculation0.outputTab.rake2DOptions.cornerA.x = {x_value}
calculation0.outputTab.rake2DOptions.cornerA.y = {y_min}
calculation0.outputTab.rake2DOptions.cornerA.z = {z_min}
calculation0.outputTab.rake2DOptions.cornerB.x = {x_value}
calculation0.outputTab.rake2DOptions.cornerB.y = {y_max}
calculation0.outputTab.rake2DOptions.cornerB.z = {z_min}
calculation0.outputTab.rake2DOptions.cornerC.x = {x_value}
calculation0.outputTab.rake2DOptions.cornerC.y = {y_max}
calculation0.outputTab.rake2DOptions.cornerC.z = {z_max}

calculation0.outputTab.rake2DOptions.ABGridsize.sizeVia = "Delta"
calculation0.outputTab.rake2DOptions.ABGridsize.delta = {delta_y}
calculation0.outputTab.rake2DOptions.BCGridsize.sizeVia = "Delta"
calculation0.outputTab.rake2DOptions.BCGridsize.delta = {delta_x}
calculation0.outputTab.rake2DOptions.maxProjectionDistance = 0.0001
calculation0.apply()
exportData0 = project0.calculations["BDL_sample_before"].createExportData()
exportData0.name = "BDL_sample_pressure_{variable}"
exportData0.filename = save_dir + "/{self.analysis}_{variable}_{self.cavity}_cavity_{tipe}.txt"
exportData0.apply()

project0.save(workdir + "/BDL_samples/BDL_sample_{variable}.pap")
project0.queueAll(True)
"""
                # Scrivi il contenuto nello script file
                with open(script_name, 'w') as file:
                    file.write(script_content)
                print(f"Script creato: {script_name}")
        except Exception as e:
            print(f"Errore nella creazione degli script: {e}")


    def execute_commands_poweracoustics(self):
        poweracoustics_path = '/usr/local/poweracoustics/bin/poweracoustics'
        
        num_workers = 4 # Usa tutti i core disponibili nel nodo
        # num_workers = num_workers/2
        print(f"Usando {num_workers} processori")
        # Esegui i comandi in parallelo usando ThreadPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.run_poweracoustics, variable, poweracoustics_path) 
                       for variable in self.variables]

            # Recupera i risultati non appena i processi terminano
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Se ci sono eccezioni, vengono sollevate qui
                except Exception as e:
                    print(f"Errore durante l'esecuzione in parallelo: {e}")
    
    def get_current_path(self):
        return self.current_dir 
class SemiEmpiricalmodel:
    def __init__(self,k,C):
        self.k = k
        self.C = C
    def ColesFernholz(self,Reynolds_theta,k,C):
           friction_coefficiet = 2/((1/k)*np.log(Reynolds_theta)+C)**2      
           return friction_coefficiet
       
    def calcola_Cf(self, Re_b):
    # Definire l'equazione da risolvere per Cf

        def equation(Cf):
            if Cf <= 0:  # Aggiungere una condizione per evitare valori non fisici
                return np.inf
            left_side = np.sqrt(2 / Cf)
            right_side = (1 /self.k) * np.log((Re_b / 2) * np.sqrt(Cf / 2)) +C - (1 /self.k)
            return left_side - right_side

    # Valore iniziale approssimato per Cf (guida al solver)
        Cf_initial_guess = 0.001

    # Risolvere l'equazione per Cf
        Cf_solution = fsolve(equation, Cf_initial_guess)

        return Cf_solution[0]
class FlowAnalysis:
    def __init__(self, path, path_points, points_file, data_files, variables):
        self.path = path
        self.path_points = path_points
        self.points_file = points_file
        self.data_files = data_files
        self.variables = variables
        self.u = None
        self.v = None
        self.w = None
        self.var_u = None
        self.var_v = None
        self.var_w = None
        self.k = None
        self.reynolds_shear_stress = None

    def load_data(self):
        # Carica i dati dei punti
        points = np.loadtxt(self.path_points + self.points_file, skiprows=1)
        points = -points[:-1, 1]
        self.points = points
        
        # Carica i dati delle variabili
        self.load_velocity_data()

    def load_velocity_data(self):
        num_vars = len(self.data_files)
        
        # Inizializza le variabili
        self.u = self.v = self.w = None
        
        # Carica i dati per ogni variabile
        if num_vars > 0:
            self.u = np.loadtxt(self.path + self.data_files[0], skiprows=1) if 'XVelocity' in self.variables else None
            self.time = self.u[:,0]
            self.u = self.u[:,1:-1]
        if num_vars > 1:
            self.v = np.loadtxt(self.path + self.data_files[1], skiprows=1) if 'YVelocity' in self.variables else None
            self.v = self.v[:,1:-1]
        if num_vars > 2:
            self.w = np.loadtxt(self.path + self.data_files[2], skiprows=1)if 'ZVelocity' in self.variables else None
            self.w = self.w[:,1:-1]
        return self.u, self.v, self.w
    
    
    def intergral_parameters(self, num_points):
        
        self.max_u = np.max(np.mean(self.u, axis=0))
        # Crea una funzione di interpolazione usando i dati originali
        interpolation_func = interp1d(self.points, np.mean(self.u, axis=0), kind='cubic', fill_value='extrapolate')
        
        # Genera nuovi punti y interpolati (ad esempio, 10 volte più punti)
        self.y_interp = np.linspace(np.min(self.points), np.max(self.points), num=len(self.points) * num_points)
        
        # Calcola i nuovi valori interpolati di u per i nuovi y
        self.u_interp = interpolation_func(self.y_interp)
       
        # Trova il boundary layer thickness dove u_interp raggiunge il 99% di u_max
        self.boundary_layer_index = np.where(self.u_interp >= 0.99 * self.max_u)[0][0]
        self.boundary_layer_thickness = self.y_interp[self.boundary_layer_index]
        
        return self.boundary_layer_thickness, self.max_u, self.u_interp, self.boundary_layer_index, self.y_interp
    
    def calculate_statistics(self):
        # Calcolo della varianza di u, v e w per ogni punto
        if self.u is not None:
            self.var_u = np.var(self.u, axis=0, ddof=1)
        else:
            self.var_u = np.zeros(self.points.shape[0])
        if self.v is not None:
            self.var_v = np.var(self.v, axis=0, ddof=1)
        else:
            self.var_v = np.zeros(self.points.shape[0])
        if self.w is not None:
            self.var_w = np.var(self.w, axis=0, ddof=1)
        else:
            self.var_w = np.zeros(self.points.shape[0])
        
        # Calcolo dell'energia cinetica turbolenta
        self.k = 0.5 * (self.var_u + self.var_v + self.var_w)
        
        # Calcolo dello stress di taglio di Reynolds
        self.reynolds_shear_stress = np.zeros(self.u.shape[1]) if self.u is not None else np.zeros(self.v.shape[1])
        self.spectra_u = []

        for i in range(len(self.reynolds_shear_stress)):
            u_i = self.u[:, i] if self.u is not None else np.zeros(self.v.shape[1])
            v_i = self.v[:, i] if self.v is not None else np.zeros(self.v.shape[1])
            # Calcolo delle medie temporali
            mean_u = np.mean(u_i)
            mean_v = np.mean(v_i)
            # Calcolo dello stress di taglio di Reynolds
            uv = np.mean((u_i - mean_u) * (v_i - mean_v))
            self.reynolds_shear_stress[i] = uv
            
            fs = 1/(self.time[1]-self.time[0])
            # Calcolo dello spettro di densità di potenza per u
            f, pxx = csd(u_i, u_i, fs=fs, nperseg=len(u_i) // 2)
            self.spectra_u.append((f, pxx))  # Salva l'intero spettro come tupla (frequenze, densità)

        
        return self.var_u, self.var_v, self.var_w, self.spectra_u
    # Funzione per tracciare gli spettri
    

    def save_results(self, path,filename):
        self.filename = filename
        self.points = self.points
        self.points = self.points
        
        # Colonne da salvare
        data = np.column_stack((self.points, self.var_u, self.var_v, self.var_w,self.reynolds_shear_stress,self.k ))

        # Salva in un file di testo con colonne
        np.savetxt(path + filename, data, header='Points Varianza_u Varianza_v Varianza_w', fmt='%0.4f', delimiter='\t')
class MultiPlot:
    def __init__(self, nrows, ncols, figsize=(20, 15)):
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=figsize)
        self.all_labels = []  # Per tenere traccia di tutte le etichette delle curve
        self.nrows = nrows
        self.ncols = ncols
       # Se c'è solo un subplot, axs non sarà una matrice
        if nrows == 1 and ncols == 1:
           self.axs = [self.axs]  # Trasformo axs in lista per uniformità di accesso
   
             
    def plot_data(self, row, col, x_data_list, y_data_list, labels, xlabel, ylabel, xmin,xmax, ymin, ymax,name,log_scale=False):
        """
        Plotta dati multipli su un singolo subplot specificato da row e col, con x_data variabile.
        
        :param row: Indice della riga del subplot.
        :param col: Indice della colonna del subplot.
        :param x_data_list: Lista dei set di dati per l'asse x (uno per ogni curva).
        :param y_data_list: Lista dei set di dati per l'asse y.
        :param labels: Lista delle etichette per ogni curva.
        :param xlabel: Etichetta per l'asse x.
        :param ylabel: Etichetta per l'asse y.
        """
       
            
        # Configura LaTeX per il testo
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        color = ['k','b','green','r']
        
        size = 70
         # Lista per etichette (per leggenda globale)
        self.labels = []
        i = 0 
        # Accedi all'asse corretto (singolo asse o matrice di assi)
        if self.nrows == 1 and self.ncols == 1:
            ax = self.axs[0]  # Unico asse disponibile
        else:
           ax = self.axs[col] if self.nrows == 1 else self.axs[row, col]
        
        fig, ax = plt.subplots(figsize=(9, 10))  
        for i, (x_data, y_data, label) in enumerate(zip(x_data_list, y_data_list, labels)):
            ax.scatter(x_data, y_data, label=label, color=color[i], facecolor = 'none', marker='o', s=size, linewidth = 2)
            ax.set_xlabel(xlabel, fontsize=50)
            ax.set_ylabel(ylabel, fontsize=50)
            
        # Imposta la leggenda per ogni curva
            ax.legend(fontsize=30, loc='best')  # Usa fontsize più piccolo se necessario
            # Imposta la scala dell'asse x (logaritmica o lineare)
            if log_scale:
                ax.set_xscale('log')  # Imposta scala logaritmica se richiesto
                ax.set_xlim([10e-04, 1])
                 # Customize the number of ticks as needed
                y_ticks = np.linspace(ymin, ymax, num=5)
                
                ax.set_yticks(y_ticks)
                formatter = FuncFormatter(lambda x, _: f'{x:.3f}')
                ax.yaxis.set_major_formatter(formatter)
                ax.xaxis.set_tick_params(pad=20)  # Move x-axis tick labels down slightly
                ax.yaxis.set_tick_params(pad=20)
            else:
                ax.set_xscale('linear')  # Imposta scala lineare altrimenti
                ax.set_xlim([xmin, xmax])
                x_ticks = np.linspace(xmin, xmax, num=5)  # Customize the number of ticks as needed
                y_ticks = np.linspace(ymin, ymax, num=5)
                
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                # Formatter to show only 2 decimal places
                formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)
                # Slight offset of tick labels to prevent overlap
                ax.xaxis.set_tick_params(pad=20)  # Move x-axis tick labels down slightly
                ax.yaxis.set_tick_params(pad=20)  
                
            ax.set_ylim([ymin, ymax])
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            
            # Move y-axis tick labels left slightly
            
            # Grid and ticks settings
 
            ax.tick_params(axis='both', labelsize=35)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
  # Aggiungi le etichette alla lista generale per la legenda globale
            self.all_labels += labels
            plt.tight_layout()
            plt.savefig(f'/home/angelo/Scrivania/profile_{name}.png', dpi=300, bbox_inches='tight')
            
    def boundary_layer_stream_plot(self, x_data_list, y_data_list, labels, xlabel, ylabel,xmin,xmax, ymin, ymax,figure_name):
            # Configura LaTeX per il testo
            l = 0.137
            cvt_width_mm = 0.49*25.4
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
             # Lista per etichette (per leggenda globale)
            self.labels = []
            i = 0     
            color = ['k','r','green','r','c','gray']
            size = 70
            # Definisci la dimensione della figura in pollici (fig_width x fig_height)
            fig = plt.figure(figsize=(20, 10))

#            Crea un asse
            ax = fig.add_subplot(111)  # Unico asse disponibile

            for i, (x_data,y_data, label) in enumerate(zip(x_data_list,y_data_list, labels)):
                ax.scatter(x_data, y_data, label=label, color=color[i], marker='o', s=size)
            ax.set_xlabel(xlabel, fontsize=50)
            ax.set_ylabel(ylabel, fontsize=50)
            

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            
            # Custom ticks to avoid overlap
            x_ticks = np.linspace(xmin, xmax, num=5)  # Customize the number of ticks as needed
            y_ticks = np.linspace(ymin, ymax, num=5)
            
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            # Formatter to show only 2 decimal places
            formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            # Slight offset of tick labels to prevent overlap
            ax.xaxis.set_tick_params(pad=20)  # Move x-axis tick labels down slightly
            ax.yaxis.set_tick_params(pad=20)  # Move y-axis tick labels left slightly
            

            ax.tick_params(axis='both', labelsize=40)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
      
            #ax.legend(labels, loc='best', fontsize=40, colonne = 2)
           
            
# Adjust the layout and save the figure with maximum resolution
            plt.tight_layout()
            plt.savefig(f'/home/angelo/Scrivania/{figure_name}.png', dpi=300, bbox_inches='tight')
            
            
            # Now create a new figure for the legend
            fig_legend = plt.figure(figsize=(10, 5))  # Adjust the figure size for the legend
        
            # Create a legend handle from the main plot
            handles, labels = ax.get_legend_handles_labels()
        
            # Add the legend to the new figure and specify the number of columns
            fig_legend.legend(handles, labels, loc='center', fontsize=30, ncol=3)
        
            # Remove axes from the legend figure
            plt.axis('off')
        
            # Save the legend as a separate figure
            plt.savefig(f'/home/angelo/Scrivania/{figure_name}_legend.png', dpi=300, bbox_inches='tight')
        
            # Close the figures to free up memory
            plt.close(fig)
            plt.close(fig_legend)
    def show_legend(self):
    # Se axs è un singolo asse, lo trattiamo come un oggetto, altrimenti lo trattiamo come un array.
        if isinstance(self.axs, np.ndarray):
        # Caso con più subplot, prendi il primo per le etichette
            handles, labels = self.axs[0].get_legend_handles_labels()
        else:
            if hasattr(self.axs, 'get_legend_handles_labels'):
                handles, labels = self.axs.get_legend_handles_labels()
        
        # Mostra la legenda globale sulla figura
                self.fig.legend(handles, labels, loc='upper center', fontsize=30, ncol=len(self.all_labels))

            else:
                raise AttributeError("self.axs non è un oggetto Axes valido")
    def save_legend(self, save_path):
        """
        Crea una figura separata per la legenda e la salva.
    
        :param save_path: Il percorso per salvare la figura della legenda.
        """
        # Gestisci il caso di un singolo subplot (axs è un singolo oggetto, non un array)
        if isinstance(self.axs, np.ndarray):
            handles, labels = self.axs[0].get_legend_handles_labels() if self.axs.ndim == 1 else self.axs[0, 0].get_legend_handles_labels()
        else:
            handles, labels = self.axs.get_legend_handles_labels()
        
        # Controlla se ci sono etichette da visualizzare
        if len(handles) == 0 or len(labels) == 0:
            raise ValueError("Non ci sono etichette o curve da visualizzare nella legenda.")
        
        # Crea una nuova figura per la legenda
        fig_legend = plt.figure(figsize=(10, 5))  # Regola la dimensione della figura per la legenda
        fig_legend.legend(handles, labels, loc='center', fontsize=30, ncol=len(labels))
        
        # Rimuovi gli assi dalla figura della legenda
        plt.axis('off')
        
        # Salva la figura della legenda
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        # Chiudi la figura della legenda
        plt.close(fig_legend)
class Contour_Analysis:
    def __init__(self, save_path, filenames, delta, voxel_size_fine=2.43e-05):
        self.save_path = save_path
        self.filenames = filenames  # Ora accetta una lista di nomi di file
        self.voxel_size_fine = voxel_size_fine
        self.delta = delta
        self.X_unique = None
        self.Y_unique = None
        self.fluc_uu = None
        
        # Set plot parameters
        self.set_plot_parameters()

    def set_plot_parameters(self):
        plt.rcParams['text.usetex'] = True
        plt.rcParams["font.family"] = ["Latin Modern Roman"]
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['axes.formatter.use_locale'] = True
        plt.rcParams['axes.formatter.useoffset'] = False
        plt.rcParams.update({'font.size': 12})
        plt.rcParams["figure.dpi"] = 300
        plt.close('all')

    def fix_coordinate(self, x, y, z):
            x = x[~pd.isnull(x)]
            x = [x[2:] for x in x]
            x = np.asarray(x).astype(float)
    
            y = y[~pd.isnull(y)]
            z = z[~pd.isnull(z)]
            z = [z[:-1] for z in z]
            z = np.asarray(z).astype(float)
            
            return x, y, z
     
    def load_data(self, path, filename):
        return np.loadtxt(os.path.join(path, filename))

    def import_and_clean_data(self, path, filename):
        # Carica il file di punti
        self.points = pd.read_csv(os.path.join(path, filename), skiprows=6, low_memory=False)
       
        x, y, z = self.points.iloc[:, 2], self.points.iloc[:, 3], self.points.iloc[:, 4]
        self.x, self.y, self.z = self.fix_coordinate(x, y, z)

        # Carica i dati grezzi
        data_raw = self.load_data(path, filename)[1:]
        
        # Crea la mesh per il plot
        self.X_unique = np.unique(self.x)
        self.Y_unique = np.unique(self.y)
        X, Y = np.meshgrid(self.X_unique, self.Y_unique)
        X, Y = np.flip(X, axis=1), np.flip(Y, axis=0)
        
        df_points = np.full_like(X, np.nan, dtype=float)
        # Crea un DataFrame per i dati di fluttuazione
        df_points = pd.DataFrame({'x': self.x, 'y': self.y, 'fluc_uu': data_raw})
        
        # Pivota per riorganizzare i dati come matrice
        self.fluc_uu = df_points.pivot(index='y', columns='x', values='fluc_uu')\
            .reindex(index=self.Y_unique[::-1], columns=self.X_unique[::-1]).to_numpy()
            
    def import_data_for_time(self, path, filenames, flag_z = None, fallback_path=None):
        # Identifica il nome della variabile dai nomi dei file
        variable_name = filenames.split('_')[1]

        # Controlla se le coordinate sono valide
        
        if fallback_path:
                print("Coordinate non trovate, utilizzo il percorso alternativo fornito.")
                self.points = np.loadtxt(fallback_path, skiprows=1)
                x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        
                # Fissa le coordinate
                self.x, self.y, self.z = self.fix_coordinate(x, y, z)
                print("Coordinate importate e riordinate.")

        else:
            # Carica i dati dei punti, usa 'low_memory=True' per ridurre la memoria
            self.points = pd.read_csv(os.path.join(path, filenames), skiprows=6, low_memory=False)
            
            # Estrai e fissa le coordinate
            x, y, z = self.points.iloc[:, 2], self.points.iloc[:, 3], self.points.iloc[:, 4]
            self.x, self.y, self.z = self.fix_coordinate(x, y, z)
        
        # Carica i dati grezzi
        data_raw = self.load_data(path, filenames)
        
        # Applica la maschera per filtrare valori NaN nelle coordinate
        
        
        # Crea la mesh per il plot
        
        if flag_z == True:
            mask = ~np.isnan(self.y)
            self.x, self.y, self.z = self.x[mask], self.y[mask], self.z[mask]
            self.X_unique, self.Y_unique = np.unique(self.z), np.unique(self.y)
        else:
            mask = ~np.isnan(self.x)
            self.x, self.y, self.z = self.x[mask], self.y[mask], self.z[mask]
            self.X_unique, self.Y_unique = np.unique(self.x), np.unique(self.y)
            
            
        X, Y = np.meshgrid(self.X_unique, self.Y_unique)
        X, Y = np.flip(X, axis=1), np.flip(Y, axis=0)
        
        # Inizializza il contenitore per i dati temporali (usando float32 per ridurre memoria)
        time_steps = np.shape(data_raw)[0]
        fluc_time_variable = np.empty((len(self.Y_unique), len(self.X_unique), time_steps), dtype=np.float32)
        
        for i in range(time_steps):
            # Crea un DataFrame temporaneo con le fluttuazioni
            if flag_z == True:
                df_points = pd.DataFrame({'x': self.z, 'y': self.y, 'fluc_uu': data_raw[i, 1:]})
            else:
                df_points = pd.DataFrame({'x': self.x, 'y': self.y, 'fluc_uu': data_raw[i, 1:]})
            # Pivota il DataFrame e convertilo in matrice numpy con i NaN sostituiti
            self.fluc_uu = df_points.pivot(index='y', columns='x', values='fluc_uu') \
                            .reindex(index=self.Y_unique[::-1], columns=self.X_unique[::-1]) \
                            .to_numpy(dtype=np.float32)
            
            # Sostituisci NaN con zero per ridurre errori nella gestione della memoria
            # self.fluc_uu = np.nan_to_num(self.fluc_uu, nan=0.0)
            
            # Inserisci lo snapshot temporale nella matrice 3D
            fluc_time_variable[:, :, i] = self.fluc_uu
            print(f"frame {i} processato")

        # Assegna la variabile temporale
        setattr(self, f"{variable_name}_time", fluc_time_variable)
        
        # Gestisci separatamente il caso di pressione con NaN
        if variable_name == 'Pressure':
            if flag_z == True:
                df_points = pd.DataFrame({'x': self.z, 'y': self.y, 'fluc_uu': data_raw[1, 1:]})
            else:
                df_points = pd.DataFrame({'x': self.x, 'y': self.y, 'fluc_uu': data_raw[1, 1:]})
            self.fluc_uu = df_points.pivot(index='y', columns='x', values='fluc_uu') \
                            .reindex(index=self.Y_unique[::-1], columns=self.X_unique[::-1]) \
                            .to_numpy(dtype=np.float32)
        pressure_with_nan = self.fluc_uu
        setattr(self, "Pressure_with_nan", pressure_with_nan)
        
        print(f"{variable_name} riordinata")
        
    
    
    def contour_subtraction(self, path1, path2, colorbar_labels, vmin_list, vmax_list):
        
        for filename, colorbar_label, vmin, vmax in zip(self.filenames, colorbar_labels, vmin_list, vmax_list):# Importa e pulisci i dati dal primo file
            
            # Importa e pulisci i dati dal secondo file
            self.import_and_clean_data(path2, filename)
            fluc_uu2 = self.fluc_uu
            self.fluc_uu2 = fluc_uu2
            self.import_and_clean_data(path1, filename)
            fluc_uu1 = self.fluc_uu
            
            # Esegui la sottrazione
            self.fluc_diff = (fluc_uu2 - fluc_uu1)*100
            
            # self.fluc_diff = self.fluc_diff / fluc_uu1
            smooth = 'no-ac_{max}'
            
            
            # max_value = np.nanmax(self.fluc_diff)
    
            # # Trova gli indici dei massimi
            # max_indices = np.where(self.fluc_diff == max_value)
            
            # media_prova1 = np.nanmean(fluc_uu1, axis=1)
            # media_prova1_matrix = media_prova1[:, np.newaxis]  # Aggiungi una dimensione per la broadcast
            # self.fluc_diff = (self.fluc_diff/np.abs(media_prova1_matrix))*100
            # Condizione per dividere per 100 i valori maggiori di 1000

            # Crea un plot di contour per la differenza
            plot_filename = os.path.join(path2, f"contour_plot_difference_{colorbar_label}_{cavity}.png")
            self.create_contour_plot(self.fluc_diff, colorbar_label=f"$\Delta {colorbar_label}/{{{colorbar_label}}}_{{{smooth}}}  \%$", 
                                     xlabel='$x/l$', ylabel='$y/h$', 
                                     vmin=vmin, vmax=vmax, 
                                     plot_filename=plot_filename)
    
    def create_contour_plot(self, Z, colorbar_label, vmin, vmax, xlabel, ylabel, plot_filename):
       
        # max_value = np.nanmax(self.fluc_uu)
        # print(max_value)
        max_value = 114
        # max_value = 31      
        cvt_width_mm = 0.49 * 25.4
        X_dimensionalized = (np.max(self.X_unique) - self.X_unique) / (cvt_width_mm)  # Invertire l'asse x
        X_dimensionalized = X_dimensionalized * 1e03
        
        
        # Imposta il layout per il grafico e la colorbar
        fig, ax = plt.subplots(figsize=(6, 3))
        
        plt.axis([np.min(X_dimensionalized), np.max(X_dimensionalized), -0.12, 0.12])
        
        # Crea il plot di contour
        contour = ax.contourf(X_dimensionalized, self.Y_unique / 0.02, Z / max_value, cmap='RdBu_r', levels=200, vmin=vmin, vmax=vmax,extend='both')
        ax.contour(X_dimensionalized, self.Y_unique / 0.02, Z / max_value, colors='black', levels=10, linewidths=0.8)
        
        # Imposta le etichette degli assi con dimensione del font
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.invert_yaxis()
        
        # Modifica la dimensione dei numeri sugli assi
        ax.tick_params(axis='both', labelsize=25, length=6, width=2, direction='in', which='both')
        plt.savefig(plot_filename + '.png', bbox_inches='tight', dpi=300)
        # Sottoplot per la colorbar
        fig_cbar = plt.figure(figsize=(10, 1))  # Dimensione specifica per la colorbar
        ax_cbar = fig_cbar.add_axes([0.05, 0.5, 0.9, 0.4])  # Posizione della colorbar

        # # Crea la colorbar
        cbar = plt.colorbar(contour, cax=ax_cbar, orientation='horizontal')

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cbar.ax.xaxis.set_major_formatter(formatter)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))

        # Pointed ends for colorbar
        cbar_length = 0.8  # Length of the colorbar
        cbar_width = 0.01   # Width of the colorbar
    
        # Set the ticks and limits for colorbar
        cbar.set_ticks(np.linspace(vmin if vmin else np.nanmin(Z), vmax if vmax else np.nanmax(Z), 4))
    
        # Adjust ticks and labels
        cbar.ax.tick_params(labelsize=35, pad=2)  # Dimensione della label della colorbar
        for label in cbar.ax.get_yticklabels():
            label.set_horizontalalignment('left')  # Centra l'etichetta

        cbar.set_label(colorbar_label, fontsize=40)

    
        # Salva la colorbar su file
        plt.savefig(plot_filename + '_colorbar.png', bbox_inches='tight', dpi=300)
    # Funzione per creare un nome file valido
    def clean_filename(self,label):
    # Rimuovi i caratteri speciali usando una regex
            return re.sub(r'[^a-zA-Z0-9_\-]', '_', label)
    
    def plot_fluctuations(self, colorbar_labels, vmin_list, vmax_list, cavity):
        for filename, colorbar_label, vmin, vmax in zip(self.filenames, colorbar_labels, vmin_list, vmax_list):
            self.import_and_clean_data(self.save_path,filename)  # Assuming this method processes the data
            colorbar_label_cleaned = self.clean_filename(colorbar_label)
            plot_filename = os.path.join(self.save_path, f"contour_plot_{colorbar_label_cleaned}_{cavity}")
            self.create_contour_plot(self.fluc_uu, colorbar_label, xlabel='$x/l$', ylabel='$y/h$', vmin=vmin, vmax=vmax, plot_filename=plot_filename)
class BoundaryLayerAnalysis_streamwiseevolution:
    def __init__(self):
        # Fluid and flow parameters
        self.k = 0.41
        self.A = 6.1e-4
        self.B = 1.43e-3
        self.C = 5.0
        self.Lambda = (self.A + self.B) ** (1/3)

        # Additional parameters
        self.uTAU_Guess = 4
        self.Tc = 25
        self.Pamb = 101325
        self.sutherland_tref = 273.15
        self.Tk = self.Tc + self.sutherland_tref
        self.gamma = 1.4
        self.Pr = 0.707
        self.Runiv = 8.314462
        self.mol_weight = 28.9645 / 1000
        self.R = self.Runiv / self.mol_weight
        self.c0 = np.sqrt(self.gamma * self.R * self.Tk)
        self.rho = self.Pamb / (self.R * self.Tk)
        self.nu = (1.458e-6 * (self.Tk ** 1.5) / (110.4 + self.Tk)) / self.rho
        
        # Data holders
        self.y = []
        self.velocity = []
        self.delta = []
        self.theta = []
        self.delt_star = []
        self.numerical_uTau = []
        self.reynolds_theta = []
        self.shape_factor = []
        self.shape_factor_error = []
        
    def log_law(self):
        """
        Calcola il profilo logaritmico, la regione viscosa e la regione di transizione con un termine di wake.
        
        Argomenti:
        Lambda -- Parametro del modello per la regione di transizione.
        B -- Parametro del modello per la regione di transizione.
        
        Ritorna:
        y -- Coordinate trasversali.
        U_plus -- Profilo completo di U^+.
        """
        # Parametri
        y = np.linspace(0, 0.2, 1000000)  # Coordinate trasversali
        delta = 17e-03                   # Spessore dello strato limite
        u_tau = 4.13                     # Velocità di attrito
        
        # Condizione eta < 1
        eta = y / delta
        mask_eta = eta < 1
        y = y[mask_eta]
        
        # Calcolo di y^+
        y_plus = (y * u_tau) / self.nu
        
        # Inizializzazione del profilo U^+
        U_plus = np.zeros_like(y_plus)
        
        # Regione viscosa: y^+ < 15
        mask_viscosa = y_plus < 5
        U_plus[mask_viscosa] = y_plus[mask_viscosa]
        
        # Regione di transizione: 15 <= y^+ <= 70
        mask_transizione = (y_plus >= 5) & (y_plus <= 70)
        yPlus_transizione = y_plus[mask_transizione]
        U_plus[mask_transizione] = (1 / self.Lambda) * (
            1 / 3 * np.log((self.Lambda * yPlus_transizione + 1) / 
                           np.sqrt((self.Lambda * yPlus_transizione) ** 2 - self.Lambda * yPlus_transizione + 1)) +
            1 / np.sqrt(3) * (np.arctan((2 * self.Lambda * yPlus_transizione - 1) / np.sqrt(3)) + np.pi / 6)
        ) + 0.25 / self.k * np.log(1 + self.k * self.B * yPlus_transizione ** 4)
        
        # Regione logaritmica con wake: y^+ > 70
        mask_logaritmica = y_plus > 70
        y_log = y[mask_logaritmica]
        U_plus[mask_logaritmica] = (1 / self.k) * np.log(y_plus[mask_logaritmica]) + self.C + \
                                   (0.55 / self.k) * 2 * np.sin(np.pi / 2 * (y_log / delta))**2
    
        return y_plus, U_plus

    def schlichting_fit(self, x, y, nu, karman, eta, Lambda, B, C, idx):
        yPlus = np.abs(y[:idx] * x / nu)
        uPlus = np.zeros(len(yPlus))
        for i in range(len(yPlus)):
            if yPlus[i] <= 5 and eta[i] <= 1:
                uPlus[i] = yPlus[i]
            elif yPlus[i] > 5 and yPlus[i] <= 70 and eta[i] <= 1:
                uPlus[i] = (1 / Lambda) * (1 / 3 * np.log((Lambda * yPlus[i] + 1) / np.sqrt((Lambda * yPlus[i]) ** 2 - Lambda * yPlus[i] + 1)) +
                            1 / np.sqrt(3) * (np.arctan((2 * Lambda * yPlus[i] - 1) / np.sqrt(3)) + np.pi / 6)) + 0.25 / karman * np.log(1 + karman * B * yPlus[i] ** 4)
            elif yPlus[i] > 70 and eta[i] <= 1:
                uPlus[i] = (1 / karman) * np.log(yPlus[i]) + C + \
                                           (0.55 / self.k) * 2 * np.sin(np.pi / 2 * (eta[i]))**2
        return uPlus * x

    def scientific_formatter(self, x, pos):
        """Convert the axis values to scientific notation."""
        return f'{x:.1e}'


    def calculate_errors(self, y_unique, u_unique, y_interp, u_interp, idx_u_99_percento, u_99_percento, theta_local, delt_star_local, reynolds_theta, interpolation_func):
        """Calcola gli errori di interpolazione, integrazione e parametri derivati."""
        
        # 1. Errore di interpolazione (RMSE)
        rmse_interpolation = np.sqrt(mean_squared_error(u_unique, interpolation_func(y_unique)))
    
        # 2. Errore di integrazione per delt_star
        # Calcolo con un numero maggiore di punti
        y_interp_more_points = np.linspace(np.min(y_interp), np.max(y_interp), num=len(y_interp) * 10)  # Maggior risoluzione
        u_interp_more_points = interpolation_func(y_interp_more_points)
    
        idx_max_velocita = np.argmax(u_interp_more_points)
        u_99_percento = 0.99 * u_interp_more_points[idx_max_velocita]
        idx_u_99_percento = np.abs(u_interp_more_points[:idx_max_velocita] - u_99_percento).argmin()
        
        delt_star_more_points = np.abs(np.trapz(1 - (u_interp_more_points[:idx_u_99_percento] / np.max(u_interp_more_points[:idx_u_99_percento])), y_interp_more_points[:idx_u_99_percento]))
        error_delt_star = np.abs(delt_star_more_points - delt_star_local)
    
        # 3. Errore di integrazione per theta
        theta_more_points = np.abs(np.trapz(u_interp_more_points[:idx_u_99_percento] / np.max(u_interp_more_points[:idx_u_99_percento]) *
                                            (1 - (u_interp_more_points[:idx_u_99_percento] / np.max(u_interp_more_points[:idx_u_99_percento]))), y_interp_more_points[:idx_u_99_percento]))
        error_theta = np.abs(theta_more_points - theta_local)
    
        # 4. Errore su Re_theta
        reynolds_theta_more_points = (theta_more_points * u_99_percento) / self.nu
        error_reynolds_theta = np.abs(reynolds_theta_more_points - reynolds_theta)
    
        # 5. Errore relativo sullo shape factor H = delta_star / theta
        H_local = delt_star_local / theta_local
        mean_u_interp = np.mean(u_interp[:idx_u_99_percento])  # Valore medio di u (per normalizzare RMSE)
        error_H = H_local * np.sqrt(
            (error_delt_star / delt_star_local) ** 2 +
            (error_theta / theta_local) ** 2 +
            (rmse_interpolation / mean_u_interp) ** 2  # Aggiungi l'errore di interpolazione
        )

        return rmse_interpolation, error_delt_star, error_theta, error_reynolds_theta, error_H

    def run_analysis(self, path, num_points, source_dw=False):
        """Main method to run the analysis and calculate boundary layer parameters"""
        for i in range(len(os.listdir(path))-1):
            data = np.asarray(pd.read_csv(path + f'sample_{i+1}.csv', skiprows=0))[:, :]
            
            filtered_array = np.array([row for row in data if not any('*' in str(item) for item in row)])
            
            data = filtered_array.astype(np.float64)
            y = np.abs(data[:, 0])  # Distance from the wall
            u = data[:, 1]
            
            # Rimuovi i duplicati in y e ottieni i valori corrispondenti in u
            y_unique, idx_unique = np.unique(y, return_index=True)
            u_unique = u[idx_unique]
            interpolation_func = interp1d(y_unique, u_unique, kind='cubic', fill_value='extrapolate')
            
            # Genera nuovi punti y interpolati
            y_interp = np.linspace(np.min(y), np.max(y), num=len(y) * num_points)
            u_interp = interpolation_func(y_interp)
            
            # Salva i dati
            self.velocity.append(u_interp)
            self.y.append(y_interp)
    
            # Calcolo spessore dello strato limite
            idx_max_velocita = np.argmax(u_interp)
            u_99_percento = 0.99 * u_interp[idx_max_velocita]
            idx_u_99_percento = np.abs(u_interp[:idx_max_velocita] - u_99_percento).argmin()
            self.delta.append(np.abs(y_interp[idx_u_99_percento]))
    
            # Calcolo delta_star
            delt_star_local = np.abs(np.trapz(1 - (u_interp[:idx_u_99_percento] / np.max(u_interp[:idx_u_99_percento])), y_interp[:idx_u_99_percento]))
            self.delt_star.append(delt_star_local)
    
            # Calcolo theta
            theta_local = np.abs(np.trapz(u_interp[:idx_u_99_percento] / np.max(u_interp[:idx_u_99_percento]) *
                                           (1 - (u_interp[:idx_u_99_percento] / np.max(u_interp[:idx_u_99_percento]))), y_interp[:idx_u_99_percento]))
            self.theta.append(theta_local)
            reynolds_theta = (theta_local * u_99_percento) / self.nu
            self.reynolds_theta.append(reynolds_theta)
            
            # Calcolo degli errori
            rmse_interpolation, error_delt_star, error_theta, error_reynolds_theta, error_H = self.calculate_errors(
                y_unique, u_unique, y_interp, u_interp, idx_u_99_percento, u_99_percento, theta_local, delt_star_local, reynolds_theta, interpolation_func
                )
       
            # Shape factor H
            H_local = delt_star_local / theta_local
            self.shape_factor.append(H_local)
            self.shape_factor_error.append(error_H)
            
            # print(f"Sample {i+1}:")
            # print(f"Errore di interpolazione (RMSE): {rmse_interpolation}")
            # print(f"Errore di integrazione per delta_star: {error_delt_star}")
            # print(f"Errore su theta: {error_theta}")
            # print(f"Errore su Re_theta: {error_reynolds_theta}")
    
            
            numerical_eta = np.abs(y_interp) / y_interp[idx_u_99_percento]
            costFun = lambda x: np.sum(np.abs(self.schlichting_fit(x, y_interp, self.nu, self.k, numerical_eta, self.Lambda, self.B, self.C, idx_u_99_percento) - u_interp[:idx_u_99_percento]))
            self.numerical_uTau.append(spOpt.least_squares(costFun, self.uTAU_Guess, method='trf')['x'][0])    
    
        self.x = np.arange(-0.07, 0.2, 0.001)
    
        # Positions for the black lines
        x_positions = [0.0044 + i * 0.01243 for i in range(11)]
        x_positions += [0.0055 + i * 0.01243 for i in range(11)]
        x_positions += [0.0069 + i * 0.01243 for i in range(11)]
        x_positions += [0.0080 + i * 0.01243 for i in range(11)]
        x_positions = sorted(x_positions)
    
        self.process_shape_factors(self.x, x_positions)
    
    def process_shape_factors(self, x, x_positions):
        """Applica le condizioni per eliminare i valori in base alla posizione."""
        for i in range(len(x)):
            for j in range(len(x_positions) - 1):
                if (x_positions[j] <= x[i] <= x_positions[j + 1]) and ((x_positions[j + 1] - x_positions[j]) < 1.18e-03):
                    # Settiamo i valori su NaN se la condizione è verificata
                    self.delta[i] = math.nan
                    self.delt_star[i] = math.nan
                    self.theta[i] = math.nan
                    self.reynolds_theta[i] = math.nan
    
    def fitting_velocity(self,path,name):
        # Calcolo spessore dello strato limite
        points = pd.read_csv(os.path.join(path, name), skiprows=6, low_memory=False)
        y = points.iloc[:, 3]
        y = y[~pd.isnull(y)]
        data = np.asarray(np.loadtxt(path + f'{name}', skiprows=0))[:, :]
        filtered_array = np.array([row for row in data if not any('*' in str(item) for item in row)])
        
        data = filtered_array.astype(np.float64)

    
        u_temp = []
        convergence = []
        j = len(data[:3646, 1]) // 10  # Un quarto dei dati
        
        # Calcolo delle medie per segmenti
        for i in np.arange(1, 10):  # Itera su 1, 2, 3, 4
            
            u_temp.append(np.mean(data[:j * i, 1:], axis=0))
            if i > 1:  # Inizia il calcolo della convergenza dalla seconda iterazione
        # Calcolo della variazione percentuale
                convergence.append(np.mean(np.abs((np.array(u_temp)[i-1] - np.array(u_temp)[i-2]) / np.array(u_temp)[i-2]) * 100))

        # Conversione in array per facilitare il calcolo
        u = np.flip(np.mean(data[:,1:],axis=0))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 10), convergence, '-o', label='Variazione Percentuale', linewidth=2)
        plt.title('Convergenza della Media in Funzione dei Segmenti', fontsize=16)
        plt.xlabel('Numero di Segmenti', fontsize=14)
        plt.ylabel('Variazione Percentuale (\%)', fontsize=14)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        
        y = -y  
        y = np.array(np.sort(y))
        idx_max_velocita = np.argmax(u)
        u_99_percento = 0.995 * u[idx_max_velocita]
        idx_u_99_percento = np.abs(u[:idx_max_velocita] - u_99_percento).argmin()
        numerical_eta = np.abs(y) / y[idx_u_99_percento]
        costFun = lambda x: np.sum(np.abs(self.schlichting_fit(x, y, self.nu, self.k, numerical_eta, self.Lambda, self.B, self.C, idx_u_99_percento) - u[:idx_u_99_percento]))
        numerical_utau = spOpt.least_squares(costFun, self.uTAU_Guess, method='trf')['x'][0]    
        u_plus = u[1:]
        y_plus = y[1:]
        delta =  y[idx_u_99_percento]
        
        return numerical_utau,u_plus,y_plus,delta,u_99_percento   
        
def plot_spectra(flow_analysis_obj, label):
        # Estrai gli spettri (frequenze, densità di potenza) per ogni oggetto
        data = np.array(flow_analysis_obj.spectra_u)
        
        # Estrai l'ultimo spetto (ad esempio, l'ultimo caso) per il tracciamento
        freq = data[-3, 0]  # Frequenze
        pxx = data[-3, 1]   # Densità di potenza spettrale
        
        # Grafico logaritmico
        plt.loglog(freq, pxx, label=label) 
        
def process_mass_flow(data_path, start, end, cutoff=10000, order=4):
    # Caricamento dei dati
    data = np.loadtxt(data_path)
    mass_flow = data[start:end, 2]
    time = data[start:end, 0]

    # Parametri per il filtro passa-basso
    fs = 1 / (time[1] - time[0])  # Frequenza di campionamento

    # Funzione per creare un filtro passa-basso
    def lowpass_filter(data, cutoff, fs, order):
        nyquist = 0.5 * 421000
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    # Applicazione del filtro passa-basso
    filtered_mass_flow = lowpass_filter(mass_flow, cutoff, fs, order)

    # Plot dei dati originali e filtrati
    plt.figure(figsize=(10, 6))
    plt.plot(time[:], mass_flow, label='Original', alpha=0.7)
    plt.plot(time[:], filtered_mass_flow, label='Filtered', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Mass Flow')
    plt.legend()
    plt.title('Mass Flow with Lowpass Filter')
    plt.grid()
    plt.show()

    # Calcolo dei parametri per l'autocorrelazione
    nperseg = len(data[:, 1])

    # Autocorrelazione con il segnale filtrato
    ref_f, Sxx = csd(filtered_mass_flow, filtered_mass_flow, fs=fs, nperseg=nperseg)

    a = 1 / ref_f[np.where(np.abs(Sxx) == np.max(np.abs(Sxx)))][0]
    print('The number of frames per cycle is: {}'.format(a))

    b = ref_f[np.where(np.abs(Sxx) == np.max(np.abs(Sxx)))][0]
    print('The frequency is: {:.2f}'.format(b))

    # Calcolo della media del segnale raw su intervalli di "a"
    interval = int(a)  # Numero di frame per intervallo
    num_intervals = len(mass_flow) // interval
    rms = np.sqrt(np.mean(np.square(mass_flow)))
    averaged_signal = np.zeros(interval)
    for i in range(num_intervals):
        mass_flow_temp = (mass_flow[i * interval:(i + 1) * interval])
        averaged_signal += mass_flow_temp
    
    averaged_signal = averaged_signal/num_intervals
    time =np.linspace(0,360,interval)
    
    # Plot dei dati originali e filtrati
    plt.figure(figsize=(10, 6))
    plt.plot(averaged_signal, label='Phased', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Mass Flow')
    plt.legend()
    plt.title('Mass Flow with Lowpass Filter')
    plt.grid()
    plt.show()


    return time, averaged_signal, rms
