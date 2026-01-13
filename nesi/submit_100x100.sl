#!/bin/bash -e
#SBATCH --job-name=iir_100x100_improved
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=72
#SBATCH --account=uoa04634
#SBATCH --output=iir_100x100_improved_%j.out
#SBATCH --error=iir_100x100_improved_%j.err

# 100x100 grid profiling with improved optimization settings

module load Julia/1.11.3-GCC-12.3.0-VTune

echo "======================================"
echo "IIR 100x100 IMPROVED (wider bounds, better optimization)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "======================================"

cd $SLURM_SUBMIT_DIR

julia --project=. -t 1 run_repressilator_profile.jl --nuisance=16 --grid=100

# Replot locally - ScatteredInterpolation not available on NeSI
# julia --project=. -t 1 replot_profile_results.jl repressilator_16nuisance_100x100_results.jls

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
