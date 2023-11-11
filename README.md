# SimpleHPCHparamsSearch
Simple Hyper-parameter search example with wandb for offline HPC setup.
This includes:
- how to sample hyper-parameters and run sweeps with wandb offline
- how to run multiple jobs on a slurm HPC

## Setup

1. load the anaconda module `module load anaconda/3.9`
2. setup a wandb account and get your API key
2. setup a conda environment with wandb and pytorch

## Running experiments on CoE HPC

`s_sweep.sh` is an example slurm batch script for running a wandb sweep on the CoE HPC. Since the HPC workstations don't have internet access, wandb has to run in offline mode. 

start multiple jobs:
`sbatch --array=0-9 s_sweep.sh [experiment_name]`

After the jobs are finished, you can sync the results with an online dashboard
using `wandb sync --sync-all`

## Structure

- `s_sweep.sh` example slurm batch script for running a wandb sweep    
- `sweep.py` hyperparameter sweep script with example experiments   
- `trainings.py` training initialization and model loading/saving
- `MyModel.py` an arbitrary VAE model to sweep parameters for
