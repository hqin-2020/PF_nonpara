#!/bin/bash

#SBATCH --job-name=pf1007b
#SBATCH --output=pf1007b.out
#SBATCH --error=pf1007b.err
#SBATCH --time=12:00:00
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_Non_parallel7b.py 