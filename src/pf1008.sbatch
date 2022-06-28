#!/bin/bash

#SBATCH --job-name=pf1008
#SBATCH --output=pf1008.out
#SBATCH --error=pf1008.err
#SBATCH --time=12:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60000

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_Non_parallel8.py 