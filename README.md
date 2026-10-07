








### Batch Jobs


```bash
#!/bin/bash -l
SBATCH --account=lp_deeplearning_gt_2026b
SBATCH --cluster=wice
SBATCH --partition=gpu_v100
SBATCH --nodes=1 --ntasks=1
SBATCH --gres=shard:1
SBATCH --time=10:00
SBATCH –output=logs-%j.out
SBATCH –error=slurm-%j.err
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
cd $VSC_DATA/dl-lab1
python lab-session-1-part1.py --skip-demos --skip-plots
```

- Specify the resources as needed
- Outside lab hours, remove: `--gres=shard:1`
- Submit the compute job to the (Slurm) scheduler: `sbatch example.slurm`
- Slurm will schedule and execute the script without your intervention
- Slurm saves an output file `logs-<jobID>.out` in the folder that you submitted your job from.


--- end