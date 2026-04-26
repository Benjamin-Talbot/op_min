#!/bin/bash
#SBATCH --job-name=qaoa_mps
#SBATCH --account=def-stijn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=5900G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/btalbot/jobs/h2o/job_%j.out
#SBATCH --error=/scratch/btalbot/jobs/h2o/job_%j.err

# Move to scratch (recommended)
cd /home/btalbot/scratch

# Load environment
module load StdEnv/2023
module load python/3.11

# Activate venv
source /home/btalbot/scratch/envs/qubo/bin/activate

# Threading
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run
python /home/btalbot/scratch/honours_code/no-qubo/h2o.py