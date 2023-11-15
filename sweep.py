import wandb
import argparse
import random
import numpy as np

from training import train


def sample_wandb_hyperparams(params):
    sampled = {}
    for k, v in params.items():
        if 'values' in v:
            sampled[k] = random.choice(v['values'])
        elif 'distribution' in v:
            if v['distribution'] == 'uniform' or v['distribution'] == 'uniform_values':
                sampled[k] = random.uniform(v['min'], v['max'])
            elif v['distribution'] == 'normal':
                sampled[k] = random.normalvariate(v['mean'], v['std'])
            elif v['distribution'] == 'log_uniform_values':
                emin, emax = np.log(v['max']), np.log(v['min'])
                sample = np.exp(random.uniform(emin, emax))
                sampled[k] = sample
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    return sampled


default_config = {
    "name": "test-mnist-proj",
    "method": "random",
    "parameters": {
        "epochs": { "values": [10] },
        "beta": { "values": [0.0] },
        "rz_mode": { "values": ['standard'] },
        "z_variance_mode": { "values": ['sigma_diag'] },
        "batch_size": { "values": [128] },
        "lr": { "values": [1e-4] },
    }
}
# beta effect on reconstruction error
beta_vs_recon = {
    "metric": {
        "name": "reconstruction error",
        "goal": "minimize"
    },
    "parameters": {
        "beta": {
            "distribution": "log_uniform_values",
            "min": 1e-2,
            "max": 5e2,
        },
    }
}
# fix beta and check effect of other parameters
multiple_params = {
    "metric": {
        "name": "reconstruction error",
        "goal": "minimize"
    },
    "parameters": {
        "beta": { "values": [0.1] },
        "rz_mode": { "values": ['standard', 'parametrized'] },
        "z_variance_mode": { "values": ['sigma_diag', 'sigma_chol', 'logvar_diag'] },
        "batch_size": { "values": [32, 64, 128, 256] },
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-1,
        },
    }
}

exp_to_sweepconfig = {
    "beta_vs_recon": beta_vs_recon,
    "multiple_params": multiple_params,
}

project = None
experiment_name = None
dataset = None

def get_sweep_config(experiment_name):
    sweepcfg = exp_to_sweepconfig[experiment_name]
    cfg = default_config
    params = cfg['parameters']
    params.update(sweepcfg['parameters'])
    cfg.update(sweepcfg)
    cfg['parameters'] = params
    cfg['name'] = project
    return cfg

def wandb_train(local=False):
    wandb_kwargs = {"project":project, "group":experiment_name}
    if local:
        config = get_sweep_config(experiment_name)
        config["controller"] = {'type': 'local'}
        sampled_params = sample_wandb_hyperparams(config["parameters"])
        config["parameters"] = sampled_params
        print(f"locally sampled params: {sampled_params}")
        wandb_kwargs['config'] = config
    with wandb.init(**wandb_kwargs) as run:
        config = wandb.config.as_dict()
        train(
            usesaved=False,
            savemod=False,
            wandb_run=run,
            **config['parameters']
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sweep", type=str, default=None)
    args.add_argument("--n_runs", type=int, default=1)
    args.add_argument("--proj", type=str, default="test-mnist-proj")
    args.add_argument("--local-wandb", type=bool, default=True)
    args.add_argument("--exp-name", type=str, default="beta_vs_recon")
    args = args.parse_args()
    project = args.proj
    experiment_name = args.exp_name
    if args.sweep is None and not args.local_wandb:
        sweepcfg = get_sweep_config(experiment_name)
        sweepcfg['name'] = project
        sweep_id = wandb.sweep(sweepcfg, project=project)
        print(f"created new sweep {sweep_id}")
        wandb.agent(sweep_id, project=args.proj, count=args.n_runs, function=wandb_train)
    elif args.local_wandb:
        wandb_train(local=True)
    else:
        print(f"continuing sweep {args.sweep}")
        wandb.agent(args.sweep, project=args.proj, count=args.n_runs, function=wandb_train)
