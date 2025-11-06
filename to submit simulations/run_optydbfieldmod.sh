#!/bin/bash
#SBATCH --job-name=seeding 
#SBATCH --partition=lining_cpu 
#SBATCH --time=140:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48 
#SBATCH --mail-type=ALL 
#SBATCH --mem-per-cpu=4096M 
#SBATCH --mail-user=angelo.paduano@polito.it 
#SBATCH --exclusive

echo "== Run starting at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

# Loading the license 
module load powerflow/2025

export EXACORP_LICENSE=29000@pcdmavallonelic.polito.it;
export PATH=/home/favallone/Solvers/PowerFLOW/6-2024-R1/bin:$PATH;

EXA_QSYSTEM_DIR=slurm; export EXA_QSYSTEM_DIR
EXA_QSYSTEM_NAME=MySlurm; export EXA_QSYSTEM_NAME
EXA_PRINT_QSUB_CMD=1; export EXA_PRINT_QSUB_CMD

source /home/apaduano/post_pro/bin/activate

python seeding_files_generator.py


# Attiva ambiente
source /home/apaduano/optydb-2025-R1/bin/LINUX-gcc-10.2.1/zoptydb_env.sh 

# Frequenze
frequencies=(800 1400 2000)

# Numero di CPU / 2
NUM_CPUS=$(nproc)
NP=$((NUM_CPUS / 2))

# Loop
for freq in "${frequencies[@]}"; do
    /home/apaduano/optydb-2025-R1/mpi/LINUX-gcc-10.2.1/bin/mpirun -np "$NP" optydb_fieldmod -i "optydb_fieldmod_${freq}.i" -o "optydb_fieldmod_${freq}.i"
done
