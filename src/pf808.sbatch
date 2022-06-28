#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf808
#SBATCH --output=pf808.out
#SBATCH --error=pf808.err
#SBATCH --time=0-20:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Load Python modules
module load python/booth/3.8/3.8.5

python Run_Aso1_0_Non_parallel808.py 