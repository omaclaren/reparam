#!/bin/bash -e
#SBATCH --job-name=iir_test_20x20
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=36
#SBATCH --account=uoa04634
#SBATCH --output=iir_test_20x20_%j.out
#SBATCH --error=iir_test_20x20_%j.err

# Quick 20x20 test with wider bounds

module load Julia/1.11.3-GCC-12.3.0-VTune

echo "======================================"
echo "IIR 20x20 TEST (wider bounds)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "======================================"

cd $SLURM_SUBMIT_DIR

julia --project=. -t 1 run_repressilator_profile.jl --nuisance=16 --grid=20
julia --project=. -t 1 replot_profile_results.jl repressilator_16nuisance_20x20_results.jls

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
