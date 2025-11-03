import os
import numpy as np
import pandas as pd
import logging
from scipy.io import savemat
import h5py
import numpy as np

# Configurazione del logger
logging.basicConfig(
    filename="contour_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class sorter:
    
    def __init__(self, save_path,data_files):
        self.client = client
        
        # Scelta modello in base a uniformflow e slip
        if uniformflow and slip:
            mph_path = '/home/apaduano/comsol_eduction/Impedance_bc/UFSC_channel_CFD_030_MF_LNSF_up_slip_ac.mph'
        elif slip:
            mph_path = '/home/apaduano/comsol_eduction/Impedance_bc/UFSC_channel_CFD_030_BL_resolved_fine_up_noslip.mph'
        else:
            mph_path = '/home/apaduano/comsol_eduction/Impedance_bc/UFSC_channel_CFD_030_BL_resolved_fine_up_noslip.mph'

        self.model_wrapper = self.client.load(mph_path)
        self.model = self.model_wrapper.java
        
    def save_large_mat(filename, data):
        """
        Salva un dizionario di dati in un file .mat compatibile con MATLAB v7.3.
        Supporta dataset di grandi dimensioni con compressione gzip.

        Args:
            filename (str): Nome del file .mat da salvare.
            data (dict): Dizionario dei dati da salvare. Le chiavi diventano i nomi delle variabili MATLAB.
        """
        print(f"Salvataggio in formato MATLAB v7.3: {filename}")

        def save_recursive(group, data):
            """
            Funzione ricorsiva per salvare dati annidati in HDF5.

            Args:
                group: Il gruppo HDF5 corrente.
                data: Il dizionario o array da salvare.
            """
            for key, value in data.items():
                try:
                    if isinstance(value, dict):
                        # Se il valore è un dizionario, crea un sottogruppo e salva ricorsivamente
                        subgroup = group.create_group(key)
                        save_recursive(subgroup, value)
                    else:
                        # Converti il valore in array NumPy
                        value = np.asarray(value)

                        # Per valori scalari
                        if value.size == 1:
                            group.create_dataset(key, data=value.item())  # Salva come scalare
                        else:
                            # Salva array di grandi dimensioni con compressione gzip
                            group.create_dataset(key, data=value, compression="gzip", chunks=True)
                except Exception as e:
                    print(f"Errore nel salvataggio di {key}: {e}")

        # Creazione del file HDF5
        with h5py.File(filename, 'w') as f:
            save_recursive(f, data)

    def flatten_dict(d, parent_key='', sep='_'):
        """
        Appiattisce un dizionario annidato.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
                
        return dict(items)

    class Contour_Analysis:
        def __init__(self, save_path, filenames, delta, voxel_size_fine=2.43e-05):
            self.save_path = save_path
            self.filenames = filenames
            self.voxel_size_fine = voxel_size_fine
            self.delta = delta
            self.X_unique = None
            self.Y_unique = None
            self.fluc_uu = None
            logging.info("Inizializzato Contour_Analysis con %d file.", len(filenames))

        def fix_coordinate(self, x, y, z):
            x = x[~pd.isnull(x)]
            x = [x[2:] for x in x]
            x = np.asarray(x).astype(float)

            y = y[~pd.isnull(y)]
            z = z[~pd.isnull(z)]
            z = [z[:-1] for z in z]
            z = np.asarray(z).astype(float)

            return x, y, z

        def load_data(self, path, filename, dtype=np.float32):
            file_path = os.path.join(path, filename)
        
            print(f'Caricamento file con np.loadtxt: {file_path}...')
        
            try:
                # Carica i dati, saltando l'header
                data = np.loadtxt(file_path, dtype=dtype, skiprows=1)
        
                print(f'Dati caricati con forma: {data.shape}')
                return data
        
            except Exception as e:
                print(f'Errore durante il caricamento del file: {e}')
                return None

        	
        def import_and_clean_data(self, path, filename, chunksize=500000):
            logging.info("Importazione dati da %s iniziata.", filename)
            chunk_iter = pd.read_csv(os.path.join(path, filename), skiprows=6, chunksize=chunksize, low_memory=False)

            x_list, y_list, z_list = [], [], []
            data_raw_list = []

            for chunk in chunk_iter:
                x_chunk, y_chunk, z_chunk = chunk.iloc[:, 2], chunk.iloc[:, 3], chunk.iloc[:, 4]
                x_clean, y_clean, z_clean = self.fix_coordinate(x_chunk, y_chunk, z_chunk)

                x_list.append(x_clean)
                y_list.append(y_clean)
                z_list.append(z_clean)
                data_raw_list.append(chunk.iloc[:, 5:].values)

            self.x = np.concatenate(x_list)
            self.y = np.concatenate(y_list)
            self.z = np.concatenate(z_list)
            data_raw = np.vstack(data_raw_list)

            self.X_unique = np.unique(self.x)
            self.Y_unique = np.unique(self.y)

            df_points = pd.DataFrame({'x': self.x, 'y': self.y, 'fluc_uu': data_raw.flatten()})
            self.fluc_uu = df_points.pivot(index='y', columns='x', values='fluc_uu')
            self.fluc_uu = self.fluc_uu.reindex(index=self.Y_unique[::-1], columns=self.X_unique[::-1]).to_numpy()

            logging.info("Importazione e pulizia dati da %s completata.", filename)

        def import_data_for_time(self, path, filename, chunksize=50):
            logging.info("Importazione coordinate da %s iniziata.", filename)
            variable_name = filename.split('_')[1]
            if variable_name == 'Pressure':

                    chunk_iter = pd.read_csv(os.path.join(path, filename), skiprows=6, chunksize=chunksize, low_memory=False)

                    x_list, y_list, z_list = [], [], []
                    data_list = []

                    for chunk in chunk_iter:
                        x_chunk, y_chunk, z_chunk = chunk.iloc[:, 2], chunk.iloc[:, 3], chunk.iloc[:, 4]  
                        x_clean, y_clean, z_clean = self.fix_coordinate(x_chunk, y_chunk, z_chunk)
                        x_list.append(x_clean)
                        y_list.append(y_clean)
                        z_list.append(z_clean)


                    self.x = np.concatenate(x_list)
                    self.y = np.concatenate(y_list)
                    self.z = np.concatenate(z_list)
                    
                    
                    x_clean = []
                    y_clean = []
                    z_clean = []
                    chunk_iter = []
                    data_raw_list = []
    	
                    self.X_unique = np.unique(self.x)
                    self.Y_unique = np.unique(self.y)

                    logging.info("Importazione coordinate da %s terminata.", filename)
            logging.info("Importazione dati temporali da %s iniziata.", filename)
            data_raw = self.load_data(path, filename)

            time_steps = np.shape(data_raw)[0]
            fluc_time_variable = np.empty((len(self.Y_unique), len(self.X_unique), 4741), dtype=np.float32)
            logging.info("Riordino %s...", filename)
            for i in range(4741):
                df_points = pd.DataFrame({'x': self.x, 'y': self.y, 'fluc_uu': data_raw[i, 1:]})
                self.fluc_uu = df_points.pivot(index='y', columns='x', values='fluc_uu')
                self.fluc_uu = self.fluc_uu.reindex(index=self.Y_unique[::-1], columns=self.X_unique[::-1]).to_numpy(dtype=np.float32)
                self.fluc_uu = np.nan_to_num(self.fluc_uu, nan=0.0)
                fluc_time_variable[:, :, i] = self.fluc_uu

            setattr(self, f"{variable_name}_time", fluc_time_variable)
            logging.info("Importazione dati temporali da %s completata.", filename)
