#!/bin/bash -e
#SBATCH --job-name=iir_50x50
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=36
#SBATCH --account=uoa04634
#SBATCH --output=iir_50x50_%j.out
#SBATCH --error=iir_50x50_%j.err

# 100x100 grid profiling for 18-param repressilator
# Uses 36 cores (standard Mahuika node)
# Estimated time: 2-3 hours (6h buffer for safety)

module load Julia/1.11.3-GCC-12.3.0-VTune

echo "======================================"
echo "IIR 100x100 Profiling Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Started: $(date)"
echo "======================================"

cd $SLURM_SUBMIT_DIR

# Run with parallel flag and 100x100 grid
# N_WORKERS will be set to SLURM_CPUS_PER_TASK - 1 (leave 1 for main)
julia --project=. -t 1 run_nesi_50x50.jl

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
