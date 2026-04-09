#!/bin/bash
#SBATCH --job-name=qaoa_mps
#SBATCH --account=def-stijn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=/scratch/btalbot/jobs/job_%j.out
#SBATCH --error=/scratch/btalbot/jobs/job_%j.err

# Move to scratch (recommended)
cd $SCRATCH

# Load environment
module load StdEnv/2023
module load python/3.11

# Activate venv
source /home/btalbot/scratch/envs/qenv/bin/activate

# Threading
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run
python /home/btalbot/scratch/op_min/h2.py