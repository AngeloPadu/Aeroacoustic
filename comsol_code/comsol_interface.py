#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:39:29 2025
@author: angelo
"""
class Comsol_lunch:
    
    def __init__(self, client, uniformflow=None, slip=None):
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
    def lnsf_mf_slip(self, freq, impedance, flag = None):
        import numpy as np
        import gc
    
        resistance = np.real(impedance)
        reactance = np.imag(impedance)
    
        try:
            # Imposta condizioni al contorno
            self.model.component("comp1").physics("lnsf").feature("imp1").selection().set(19)
            self.model.component("comp1").physics("lnsf").feature("imp1").set("Zn", f"({resistance}+{reactance}i)*lnsf.rho0*lnsf.c0")
            self.model.component("comp1").physics("lnsf").feature("imp1").set("TangentialVelocity", "Slip")
    
            # Esegui studio
            #model.study("std2").run()
            self.model.study("std3").feature("freq").set("plist", f"{freq}")
            self.model.study("std3").run()
    
            self.model.sol("sol4").feature("st1").set("study", "std3")
            self.model.sol("sol4").feature("st1").set("studystep", "freq")
    
            if "dset1" not in self.model.result().dataset().tags():
                self.model.result().dataset().create("dset1", "Solution")
                self.model.result().dataset("dset1").set("solution", "sol4")
    
            self.model.result().dataset("cpt1").set("method", "file")
            self.model.result().dataset("cpt1").set("filename", "./eduction_points_lined_sec.txt")
    
            # Rimuovi interp esistente se già presente
            if "gev1" in self.model.result().numerical().tags():
                self.model.result().numerical().remove("gev1")
    
            interp = self.model.result().numerical().create("gev1", "Interp")
            interp.set("expr", ["real(p2)", "imag(p2)"])
            interp.set("data", "cpt1")
            interp.run()
    
            data = interp.getData()
            data_np = np.array([[val for val in row] for row in data])
            pressure_complex = data_np[0, :, :] + 1j * data_np[1, :, :]
            
            if flag == True:
                 self.model.save(f'/home/angelo/Scrivania/comsol_lnsf_noslip_{freq}.mph')
                
            return pressure_complex
    
        except Exception as e:
            print(f"Errore COMSOL: {e}")
            return np.nan
            
    def lnsf_slip(self, freq, impedance, flag = None):
        import numpy as np
        import gc
    
        resistance = np.real(impedance)
        reactance = np.imag(impedance)
    
        try:
            # Imposta condizioni al contorno
            self.model.component("comp1").physics("lnsf").feature("imp1").selection().set(19)
            self.model.component("comp1").physics("lnsf").feature("imp1").set("Zn", f"({resistance}+{reactance}i)*lnsf.rho0*lnsf.c0")
            self.model.component("comp1").physics("lnsf").feature("imp1").set("TangentialVelocity", "Slip")
    
            # Esegui studio
            #model.study("std2").run()
            self.model.study("std3").feature("freq").set("plist", f"{freq}")
            self.model.study("std3").run()
    
            self.model.sol("sol4").feature("st1").set("study", "std3")
            self.model.sol("sol4").feature("st1").set("studystep", "freq")
    
            if "dset1" not in self.model.result().dataset().tags():
                self.model.result().dataset().create("dset1", "Solution")
                self.model.result().dataset("dset1").set("solution", "sol4")
    
            self.model.result().dataset("cpt1").set("method", "file")
            self.model.result().dataset("cpt1").set("filename", "./eduction_points_lined_sec.txt")
    
            # Rimuovi interp esistente se già presente
            if "gev1" in self.model.result().numerical().tags():
                self.model.result().numerical().remove("gev1")
    
            interp = self.model.result().numerical().create("gev1", "Interp")
            interp.set("expr", ["real(p2)", "imag(p2)"])
            interp.set("data", "cpt1")
            interp.run()
    
            data = interp.getData()
            data_np = np.array([[val for val in row] for row in data])
            pressure_complex = data_np[0, :, :] + 1j * data_np[1, :, :]
            
            if flag == True:
                 self.model.save(f'/home/angelo/Scrivania/comsol_lnsf_noslip_{freq}.mph')
                
            return pressure_complex
    
        except Exception as e:
            print(f"Errore COMSOL: {e}")
            return np.nan
    
    def lnsf_noslip(self, freq, impedance, flag = None):
        import numpy as np
        import gc
    
        resistance = np.real(impedance)
        reactance = np.imag(impedance)
    
        #model = None  # dichiarato fuori dal try per poterlo chiudere nel finally
        #model_wrapper = None
        try:
            # model_wrapper = client.load('/home/apaduano/comsol_eduction/Impedance_bc/UFSC_channel_CFD_030_BL_resolved_fine_up_noslip.mph')
            # model = model_wrapper.java
    
            # Imposta condizioni al contorno
            self.model.component("comp1").physics("lnsf").feature("imp1").selection().set(19)
            self.model.component("comp1").physics("lnsf").feature("imp1").set("Zn", f"({resistance}+{reactance}i)*lnsf.rho0*lnsf.c0")
            self.model.component("comp1").physics("lnsf").feature("imp1").set("TangentialVelocity", "NoSlip")
    
            # Esegui studio
            #model.study("std2").run()
            self.model.study("std3").feature("freq").set("plist", f"{freq}")
            self.model.study("std3").run()
    
            self.model.sol("sol4").feature("st1").set("study", "std3")
            self.model.sol("sol4").feature("st1").set("studystep", "freq")
    
            if "dset1" not in self.model.result().dataset().tags():
                self.model.result().dataset().create("dset1", "Solution")
                self.model.result().dataset("dset1").set("solution", "sol4")
    
            self.model.result().dataset("cpt1").set("method", "file")
            self.model.result().dataset("cpt1").set("filename", "./eduction_points_lined_sec.txt")
    
            # Rimuovi interp esistente se già presente
            if "gev1" in self.model.result().numerical().tags():
                self.model.result().numerical().remove("gev1")
    
            interp = self.model.result().numerical().create("gev1", "Interp")
            interp.set("expr", ["real(p2)", "imag(p2)"])
            interp.set("data", "cpt1")
            interp.run()
    
            data = interp.getData()
            data_np = np.array([[val for val in row] for row in data])
            pressure_complex = data_np[0, :, :] + 1j * data_np[1, :, :]
            
            if flag == True:
                 self.model.save(f'/home/angelo/Scrivania/comsol_lnsf_noslip_{freq}.mph')
                
            return pressure_complex
    
        except Exception as e:
            print(f"Errore COMSOL: {e}")
            return np.nan
    

