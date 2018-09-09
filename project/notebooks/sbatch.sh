#!/bin/bash

#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-12:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nhoernle@g.harvard.edu

# source new-modules.sh

module load python/3.6.0-fasrc01

source activate essil
export PYTHONUNBUFFERED=1
./dispatch_script.py out_file.csv 12 4 1000 1000
source deactivate essil

echo "finished"
