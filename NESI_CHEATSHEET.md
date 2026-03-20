# NeSI OnDemand Cheatsheet

> **Project workflow note:** for this project, use **NeSI OnDemand only**.
> Upload/update files via **OnDemand Files** and submit/monitor jobs via
> **OnDemand Shell Access**.

## 1. Upload Files
- Go to https://ondemand.nesi.org.nz
- Login with NeSI credentials + 2FA
- Files → Home Directory
- Upload `nesi_upload.zip`

## 2. Open Terminal
- Clusters → Mahuika Shell Access

## 3. Setup (first time only)
```bash
cd ~
unzip nesi_upload.zip -d reparam
cd reparam
```

## 4. Updating Files (ongoing workflow)

After initial setup, upload individual changed files as needed:
- Files → Home Directory → Navigate to `~/reparam/`
- Upload only the files that changed (e.g., `core.jl`, `run_repressilator_profile.jl`)
- For files in subdirectories, navigate there first (e.g., `~/reparam/nesi/`)

**Do NOT re-upload the entire zip** - just upload changed files individually.

## 5. Test Job (first time only)
```bash
sbatch submit_nesi.sl
```

Check status:
```bash
squeue --me
```

View output when done:
```bash
cat nesi_test_*.out
```

Should say "All key packages loaded successfully."

## 6. Run Profiling Jobs
Before submitting, make sure any changed local files have been uploaded to the matching paths under `~/reparam/` via **OnDemand Files**.

Then in **OnDemand Shell Access** run:
```bash
cd ~/reparam/nesi
sbatch submit_50x50.sl       # ~2-3 hours
sbatch submit_100x100.sl     # ~8-11 hours
sbatch submit_test_20x20.sl  # ~30 min (for testing)
```

Monitor:
```bash
squeue --me              # job status
sacct -j JOBID           # detailed info
tail -f iir_100x100_*.out  # live output (once running)
```

## 7. Get Results
When done, download via Files app:
- `repressilator_16nuisance_NxN_results.jls` (binary data)
- Output logs: `iir_NxN_*.out`, `iir_NxN_*.err`

Results go to `~/reparam/nesi/` directory. Download to local `nesi/` folder for plotting.

## Useful Commands
```bash
squeue --me                    # your jobs
scancel JOBID                  # cancel job
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS  # job stats
module avail julia             # available Julia versions
```

## Project Code
`uoa04634`
