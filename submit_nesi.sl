#!/bin/bash -e
#SBATCH --job-name=nesi_test
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --account=uoa04634
#SBATCH --output=nesi_test_%j.out
#SBATCH --error=nesi_test_%j.err

# Load Julia module (adjust version if needed, 1.10 is standard now)
module load julia/1.10.3

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Project: $SLURM_JOB_ACCOUNT"

# Move to submission directory
cd $SLURM_SUBMIT_DIR

# Instantiate and run
# The --project=. flag tells Julia to use the Project.toml in the current folder
julia --project=. --threads $SLURM_CPUS_PER_TASK nesi_test.jl
