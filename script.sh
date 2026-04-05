#!/bin/bash
#SBATCH --job-name=qiskit_sim
#SBATCH --account=btalbot
#SBATCH --gres=gpu:h100:1
#SBATCH --gres=gpu:h100:1
#SBATCH --gpus=rtx6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=75G
#SBATCH --time=160:00:00
#SBATCH --output=qiskit_sim.out
#SBATCH --error=qiskit_sim.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load cuda/12.9
module load cuquantum/25.06.0.10

source ~/venvs/cuQ-aer-0172/bin/activate

python FWHT_matrixfree.py


#!/bin/bash
#SBATCH --job-name=qaoa_mps
#SBATCH --account=def-youraccount   # <-- change this
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=750G
#SBATCH --time=04:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Load modules
module load python/3.11

# Activate your virtual environment
source ~/qenv/bin/activate

# Threading controls (VERY important for EPYC)
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Optional: improve performance consistency
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run your script
python sample.py