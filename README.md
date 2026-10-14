
# Overview





### Init
Create SSH session
```bash
# Create and access directory
mkdir -p "$VSC_DATA/deep_learning/Lab1"
cd $VSC_DATA/deep_learning/Lab1


# Copy from KUL shared lab files
cp -r /lustre1/project/stg_00220/Labs/Lab1/Code .
```

Create VSCode Tunnel
```bash

```
Laptop-VSCode :
- Install extension : Remote-Tunnel (Microsoft).
- Command palette : Connect to tunnel
- then connect with github account
- select the tunnel that you started on 

Open folder :
```bash
/data/leuven/394/vsc39441/deep_learning
```

Load modules :
```bash
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl
module load matplotlib/3.10.3-gfbf-2025a
module load Seaborn/0.13.2-gfbf-2025a
module load tqdm/4.67.1-GCCcore-14.2.0
module load Pillow/11.3.0-GCCcore-14.2.0

# Manually install scikit-learn
pip install scikit-learn

# run test code
cd /data/leuven/394/vsc39441/deep_learning/Lab1/Code
python 01_check_dataset_access.py
```


## L1 - Image Classification
See [[1-image-classification/L1.md]]







---

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