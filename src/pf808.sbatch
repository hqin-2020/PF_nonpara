#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf808
#SBATCH --output=pf808.out
#SBATCH --error=pf808.err
#SBATCH --time=1-20:00:00
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G

# Load Python modules
module load python/booth/3.8/3.8.5

srun python3 Run_Aso1_0_Non_parallel808.py 