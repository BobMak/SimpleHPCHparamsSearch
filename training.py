import argparse
import os

import numpy as np
import torch
import torch.utils.data
import torchvision
import wandb
from torch.nn import functional as F

from MyModel import PMVIB


def train(
    usesaved=False,
    savemod=True,
    calculate_error=True,
    beta=None,
    epochs=None,
    latent_dims=None,
    wandb_run=None,
    batch_size=None,
    lr=None,
    rz_mode=None,
    **kwargs
):
    print(locals())
    if kwargs:
        print("WARNING: ignoring kwargs:")
        for k, v in kwargs.items():
            print(f"\t{k}: {v}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_workers = 4
    train_data = torchvision.datasets.MNIST(
        'datasets/',
        download=True,
        train=True,
        transform=torchvision.transforms.ToTensor()
    )

    test_data = torchvision.datasets.MNIST(
        'datasets/',
        download=True,
        train=False,
        transform=torchvision.transforms.ToTensor()
    )

    cnn_enc_arch = [
        ('Conv2d', (1, 32, 4, 2, 1)),
        ('ReLU', (True,)),
        ('Conv2d', (32, 64, 4, 2, 1)),
        ('ReLU', (True,)),
        ('Conv2d', (64, 64, 4, 2)),
        ('ReLU', (True,)),
    ]
    cnn_dec_arch = [
        ('ConvTranspose2d', (64, 64, 4, 2)),
        ('ReLU', (True,)),
        ('ConvTranspose2d', (64, 32, 4, 2)),
        ('ReLU', (True,)),
        ('ConvTranspose2d', (32, 1, 4, 2, 1)),
        ('Sigmoid', ()),
    ]

    model = PMVIB(
        beta=beta,
        in_channels=1,
        in_size=(28, 28),
        out_size=(1,28, 28),
        latent_dims=latent_dims,
        enc_arch=cnn_enc_arch,
        dec_arch=cnn_dec_arch,
        rz_mode=rz_mode,
        device=device,
        lr=lr,
        out_type='image',
        wandb_run=wandb_run
    )
    path = f"MNIST_{str(model)}"

    model.to(device)
    model_parameters = model.get_model_size()
    print(f"model size: {model_parameters}")
    model_datasample = np.prod(train_data.train_data[0].size())
    print(f"datasample size: {model_datasample}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"available gpu memory: {gpu_memory}")
    if usesaved:
        model = PMVIB.load('trained/'+path)
    else:
        print(f"batch size: {batch_size}")
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        model.fit(epochs, train_data_loader)
        if savemod:
            os.makedirs('trained', exist_ok=True)
            model.save('trained/' + path)

    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    # calculate accuracy
    if calculate_error:
        model.to(device)
        # simple count error
        error = 0
        with torch.no_grad():
            for xs, ys in test_data_loader:
                res, inp, mu, logsigma = model(xs.to(device))
                error += F.binary_cross_entropy_with_logits(res, xs.to(device), reduction='sum').item()
        deterr = error/len(test_data)
        print(f"reconstruction error: {deterr:.2f}")
        if wandb_run:
            wandb_run.log({ "reconstruction error": deterr })


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--proj", type=str, default="test-mnist-proj")
    argparser.add_argument("--usesaved", '-u', type=bool, default=False, help="use saved model instead of training new one")
    argparser.add_argument("--savemod", type=str, default=True)
    argparser.add_argument("--calculate_error", '-cr', type=bool, default=True)
    argparser.add_argument("--num-runs", type=int, default=1)
    argparser.add_argument("--wandb", "-wb", type=bool, default=False)
    # hyperparameters
    argparser.add_argument("--epochs", '-e', type=int, default=10)
    argparser.add_argument("--beta", '-b', type=float, default=10.0)
    argparser.add_argument("--rz_mode", "-rz", type=str, default="standard")
    argparser.add_argument("--batch_size", "-bs", type=int, default=64)
    argparser.add_argument("--lr", "-lr", type=float, default=1e-4)

    args = argparser.parse_args()
    for i in range(args.num_runs):
        if args.wandb:
            run = wandb.init(project=args.proj, entity="vladmak", reinit=True, config=vars(args))
        else:
            run = None
        try:
            train(
                latent_dims=[2,2],
                wandb_run=run,
                **vars(args)
            )
        except Exception as e:
            print(e)
            if args.wandb:
                wandb.alert(title="Training failed", text=str(e))
            raise e
        finally:
            if args.wandb:
                run.finish()
