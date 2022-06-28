#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf209wc
#SBATCH --output=pf209wc.out
#SBATCH --error=pf209wc.err
#SBATCH --time=0-8:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Load Python modules
module load python/booth/3.8/3.8.5

srun python3 Run_Aso1_0_Non_parallel209wc.py 