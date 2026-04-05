#!/bin/bash
#SBATCH --job-name=qaoa_mps
#SBATCH --account=btalbot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Load modules
module load python/3.11
module load cuda/12.9

# Activate your virtual environment
source ~/envs/qubo/bin/activate

# Threading controls (VERY important for EPYC)
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Optional: improve performance consistency
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run your script
python sample.py