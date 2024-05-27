import os
import numpy as np

# Assicura che la directory di output esista
output_dir = "./export"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Apri il progetto e i file necessari
project0 = app.currentProject
project0.openPowerflowData(
    CDIFilename="/media/angelo/scanned_geom/Simulations/145/scanned_geom/flow/40vx/1400/ac-NASA_UFSC-V16_2chW-M0.3-NoSlip_NoSlip_run_fine.cdi",
    surfaceMeasFilename="/media/angelo/scanned_geom/Simulations/145/scanned_geom/flow/40vx/1400/ac_plane1_stream_inst.snc",
    scale=1.0, rotateX=0.0, rotateUnitX="deg", rotateY=0.0, rotateUnitY="deg", rotateZ=0.0, rotateUnitZ="deg",
    translateX=0.0, translateUnitX="m", translateY=0.0, translateUnitY="m", translateZ=0.0, translateUnitZ="m"
)
project0.openTargetMesh(
    targetFilename="/media/angelo/scanned_geom/Simulations/145/scanned_geom/flow/40vx/1400/rectangular_mesh.nas",
    fileType="NASTRAN", scale=1.0, scaleUnitName="m",
    rotateX=90.0, rotateUnitX="deg", rotateY=90.0, rotateUnitY="deg", rotateZ=0.0, rotateUnitZ="deg",
    translateX=0.0, translateUnitX="m", translateY=0.0, translateUnitY="m", translateZ=0.006223, translateUnitZ="m",
    coordinateSysName="default_csys"
)

# Ottieni il numero di frame
n_ts = project0.numFrames

# Seleziona la regione e la faccia target per l'export
targetFile0 = project0.targetFile
targetRegion0 = targetFile0.getRegion(name="target_region")
targetFace0 = targetRegion0.getFace(name="1")
targetFace0.selectedForExport = True
targetFace0.exportSelection = "Back"

# Inizializza le variabili
i = 0
project0.openFrame(frame=0)

# Itera sui frame ed esegui l'export dei dati
while project0.currentFrame < n_ts :
    interpolation0 = project0.interpolation
    interpolation0.interpolate()
    
    outputFileParameterSet0 = interpolation0.outputFileParameterSet
    outputFileParameter0 = outputFileParameterSet0.getParameter("Emit Element Centroids")
    
    # Imposta il nome del file di output correttamente
    output_file_path = os.path.join(output_dir, "frame_{}.dat".format(i))
    outputFileParameter0.value = output_file_path
    interpolation0.exportFile(output_file_path)
    # Stampa il nome del file di output per il debug
    print("Exporting frame {} to file: {}".format(i, output_file_path))
    

    
    i += 1
    project0.openFrame(frame=i)

