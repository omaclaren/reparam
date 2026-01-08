#!/bin/bash -e
#SBATCH --job-name=iir_50x50
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=36
#SBATCH --account=uoa04634
#SBATCH --output=iir_50x50_%j.out
#SBATCH --error=iir_50x50_%j.err

# 50x50 grid profiling with unified script

module load Julia/1.11.3-GCC-12.3.0-VTune

echo "======================================"
echo "IIR 50x50 Profiling (unified script)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "======================================"

cd $SLURM_SUBMIT_DIR

julia --project=. -t 1 run_repressilator_profile.jl --nuisance=16 --grid=50
julia --project=. -t 1 replot_profile_results.jl repressilator_16nuisance_50x50_results.jls

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
