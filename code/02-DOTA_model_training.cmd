#!/bin/bash
#SBATCH -J DOTA_model_training
#SBATCH -o /dss/dsshome1/0A/di38tac/DOTA-Net/FasterRCNN/logfiles/stdout.logfile
#SBATCH -e /dss/dsshome1/0A/di38tac/DOTA-Net/FasterRCNN/logfiles/stderr.logfile
#SBATCH -D /dss/dsshome1/0A/di38tac
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_testgpu
#SBATCH --cpus-per-task=40    
#SBATCH --mem=300gb
#SBATCH --mail-type=all
#SBATCH --mail-user=laura.obrecht@stud-mail.uni-wuerzburg.de
#SBATCH --export=NONE
#SBATCH --time=2:00:00
#SBATCH --account=pn39ju-c

# Load necessary modules (if required by the cluster)
module load slurm_setup
module load python
module load uv

# Activate the uv virtual environment
#source .venv/bin/activate

# Python script
script="02-DOTA_FasterRCNN.py"

# Run your Python script
uv run --no-project -p /dss/dsshome1/0A/di38tac/DOTA-Net/.venv python /dss/dsshome1/0A/di38tac/DOTA-Net/code/$script