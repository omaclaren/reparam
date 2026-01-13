#!/bin/bash -e
#SBATCH --job-name=iir_20x20_improved
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=72
#SBATCH --account=uoa04634
#SBATCH --output=iir_20x20_improved_%j.out
#SBATCH --error=iir_20x20_improved_%j.err

# 20x20 with improved optimization settings (more restarts, longer timeouts)

module load Julia/1.11.3-GCC-12.3.0-VTune

echo "======================================"
echo "IIR 20x20 IMPROVED (wider bounds, better optimization)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "======================================"

cd $SLURM_SUBMIT_DIR

julia --project=. -t 1 run_repressilator_profile.jl --nuisance=16 --grid=20

# Replot locally - ScatteredInterpolation not available on NeSI
# julia --project=. -t 1 replot_profile_results.jl repressilator_16nuisance_20x20_results.jls

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
