#!/bin/bash -e
#SBATCH --job-name=iir_50x50
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=72
#SBATCH --account=uoa04634
#SBATCH --output=iir_50x50_%j.out
#SBATCH --error=iir_50x50_%j.err

# 50x50 grid profiling with current settings

module load Julia/1.11.3-GCC-12.3.0-VTune

echo "======================================"
echo "IIR 50x50 Profile (current settings)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "======================================"

cd $SLURM_SUBMIT_DIR

julia --project=. -t 1 run_repressilator_profile.jl --nuisance=16 --grid=50

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
