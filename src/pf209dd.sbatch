#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf209dd
#SBATCH --output=pf209dd.out
#SBATCH --error=pf209dd.err
#SBATCH --time=0-23:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Load Python modules
module load python/booth/3.8/3.8.5

srun python3 Run_Aso1_0_Non_parallel209dd.py 