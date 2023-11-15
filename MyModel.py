import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


def kldmg(mu_1, sigma_1, mu_2, sigma_2, z_dim):
    """
    KL divergence between the multivariate gaussians N(mu_next_calculated, logvar_next_calculated) and N(mu_next, logvar_next),
    where mu_next, logvar_next = Q(x+1)
    """
    mu_diff_ = (mu_1 - mu_2).unsqueeze(2)
    sigma_1_inverse = sigma_1.inverse()
    mu_diff_ = torch.bmm(mu_diff_.transpose(1, 2), sigma_1_inverse).bmm(mu_diff_).sum(dim=(1, 2))
    sigma_trace = torch.diagonal(torch.bmm(sigma_1_inverse, sigma_2), dim1=-2, dim2=-1).sum(dim=1)
    logdet_diff = torch.logdet(sigma_2).nan_to_num(0) - torch.logdet(sigma_1).nan_to_num(0)
    _kld = 0.5 * (mu_diff_ + sigma_trace - logdet_diff - z_dim).mean()
    return _kld


def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)


str_to_nnmodule = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Linear': nn.Linear,
    'Conv2d': nn.Conv2d,
    'ConvTranspose2d': nn.ConvTranspose2d,
    'Sigmoid': nn.Sigmoid,
    'Flatten': nn.Flatten,
    'Softmax': nn.Softmax,
    'LogSoftmax': nn.LogSoftmax,
    'Unflatten': nn.Unflatten,
    'Tanh': nn.Tanh,
    'MaxPool2d': nn.MaxPool2d,
}

def build_model(architecture):
    encoder = nn.Sequential()
    for i, (layer_type, layer_params) in enumerate(architecture):
        encoder.append(str_to_nnmodule[layer_type](*layer_params))
    return encoder

def get_lower_triangular(triangular_matrix, z_dim:int, device:torch.device):
    lower_triangular = torch.zeros((triangular_matrix.shape[0], z_dim, z_dim), device=device)
    indices = torch.tril_indices(row=z_dim, col=z_dim, offset=0)
    lower_triangular[:, indices[0], indices[1]] = triangular_matrix
    return lower_triangular


class PMVIB(nn.Module):
    def __init__(self,
                 beta: float,
                 in_channels: int,
                 in_size: [int],
                 latent_dims: [int],
                 enc_arch: [(str, tuple)]=None,
                 dec_arch: [(str, tuple)]=None,
                 out_size=10,
                 out_type='categorical',
                 device=torch.device('cpu'),
                 rz_mode='standard',
                 lr=1e-4,
                 wandb_run=None,
                 epochs=0,
                 ) -> None:
        """
        Parallel Multivariate Variational Information Bottleneck
        :param latent_dims: (list) determines number and size of independent latent T variables
        :param rz_mode: {'standard', 'parametrized',} determines the type of p(z) to use
        as a latent prior. 'standard' uses a standard normal distribution, 'parametrized' uses a
        multivariate gaussian with learnable parameters.
        """
        super(PMVIB, self).__init__()
        self.device = device
        self.latent_dims = tuple(latent_dims)
        self.enc_arch = enc_arch
        self.dec_arch = dec_arch
        self.out_size = out_size
        assert out_type in {'categorical', 'image', 'continuous'}
        self.out_type = out_type
        self.in_channels = in_channels
        self.in_size = in_size
        assert rz_mode in {'standard', 'parametrized'}
        self.rz_mode = rz_mode
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        # this will allow us to easily load the model
        self.kwargs = locals()
        self.kwargs.pop('self')
        self.kwargs.pop('__class__')
        self.encoder = None
        self.encoders = []
        encoder_sizes = []
        for i in range(len(latent_dims)):
            encoder = build_model(enc_arch)
            self.encoders.append(encoder)
        self.encoders = nn.ModuleList(self.encoders)

        # find the flat size of the encoder output
        with torch.no_grad():
            out = encoder(torch.zeros((1, in_channels, *in_size)))
            enc_out_shape = out.shape
            encoder_size = np.prod(enc_out_shape[1:]).item()

        self.fc_mus = []
        self.fc_vars = []
        self.encode = self.logvar_encode
        self.reparameterize = self.logvar_reparameterize
        self.mixz = self.logvar_mixz
        for i, ld in enumerate(latent_dims):
            fc_mu = nn.Sequential(
                nn.Linear(encoder_size, ld),   # for full matrix
            ).to(device)
            sig_out = ld
            fc_var = nn.Sequential(
                nn.Linear(encoder_size, sig_out),
            ).to(device)
            self.fc_mus.append(fc_mu)
            self.fc_vars.append(fc_var)
        self.fc_mus = nn.ModuleList(self.fc_mus)
        self.fc_vars = nn.ModuleList(self.fc_vars)

        if rz_mode == 'parametrized':
            self.r_mus = []
            self.r_logvars = []
            for i, ld in enumerate(latent_dims):
                r_mu = nn.Parameter(torch.randn(ld))
                r_logvar = nn.Parameter(torch.randn(ld))
                self.r_mus.append(r_mu)
                self.r_logvars.append(r_logvar)
            self.r_mus = nn.ParameterList(self.r_mus)

        def build_decoder(inpt):
            if out_type == 'image':
                decoder = nn.Sequential(
                    nn.Linear(inpt, encoder_size),  # 2048
                    nn.ReLU(True),
                    nn.Unflatten(1, enc_out_shape[1:]),
                )
                dec = build_model(dec_arch)
                decoder.extend(dec)
            elif out_type == 'categorical':
                if dec_arch:
                    decoder = build_model(dec_arch)
                else:
                    decoder = nn.Sequential(
                        nn.Linear(inpt, self.out_size),
                        nn.LogSoftmax(dim=1)
                    )
            elif out_type == 'continuous':
                if dec_arch:
                    decoder = build_model(dec_arch)
                else:
                    decoder = nn.Sequential(
                        nn.Linear(inpt, self.out_size),
                    )
            else:
                raise ValueError('out_type must be one of {categorical, image}')
            return decoder
        self.decoders = []
        for i in range(len(latent_dims)):
            dec = build_decoder(latent_dims[i])
            self.decoders.append(dec[:-1])
            self.decoder_activation = dec[-1]
        self.decoders = nn.ModuleList(self.decoders)
        self.decode = self._decode_separate

        # apparently, orthogonal weight initialization improves performance in VAEs
        self.apply(weights_init)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # wandb parameters
        self.wandb_run = wandb_run
        self.logging_interval = None

        # for IB logging
        self.hY = None

    def calc_hy(self, train_data_loader, device):
        entropy = 0
        # calculate entropy of a discrete output random variable
        for x, y in train_data_loader:
            y = y.to(device)
            y_one_hot = torch.zeros((y.size(0), self.out_size), device=device)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1)
            p_y = torch.sum(y_one_hot) / (y.size(0) * self.out_size)
            entropy += -torch.sum(p_y * torch.log(p_y + 1e-8))
        self.hY = entropy
        return entropy

    def set_hy(self, hy):
        self.hY = hy

    def log(self, data):
        if self.wandb_run is not None:
            self.wandb_run.log(data)
            if self.hY and 'iyz' in data:
                self.wandb_run.log({'I(Y;Z) lower bound': self.hY - data['iyz']})

    def decode(self, z):
        result = self.decoder(z)
        return result

    def _decode_separate(self, z):
        """decode with separate decoders if share_decoder is False"""
        result = torch.zeros((z.shape[0], *self.out_size), device=self.device)
        for i in range(len(self.latent_dims)):
            result += self.decoders[i](z[:, sum(self.latent_dims[:i]):sum(self.latent_dims[:i+1])])
        return self.decoder_activation(result)

    def encode(self, input):
        raise NotImplementedError()

    def logvar_encode(self, input):
        mus = []
        logvars = []
        for i, encoder in enumerate(self.encoders):
            enc_out = encoder(input).flatten(start_dim=1)
            mu = self.fc_mus[i](enc_out)
            logvar = self.fc_vars[i](enc_out)
            mus.append(mu)
            logvars.append(logvar)
        return [mus, logvars]

    def reparameterize(self, mus, logvars):
        raise NotImplementedError()

    def logvar_reparameterize(self, mus, logvars):
        """
        Reparameterization trick to sample from N(0,1)
        while preserving the log-scale.
        :param mus: [Tensor] [[B x D], [B x D]]
        :param logvars: [(Tensor)] [[B x D], [B x D]]
        :return: (Tensor) [B x D]
        """
        z_all = []
        for mu, logvar in zip(mus, logvars):
            eps = torch.randn_like(mu)
            std = torch.exp(0.5 * logvar)
            z = mu + eps * std
            z_all.append(z)
        return torch.cat(z_all, dim=1)

    def forward(self, inp):
        mus, logvars = self.encode(inp)
        z = self.reparameterize(mus, logvars)
        return [self.decode(z), inp, mus, logvars]

    def ablated_forward(self, inp, enc_idx, deterministic=False):
        mus, logvars = self.encode(inp)
        for i in range(len(mus)):
            if i != enc_idx:
                mus[i] = torch.zeros_like(mus[i])
                logvars[i] = torch.zeros_like(logvars[i])
        # z = torch.cat(mus, dim=1)
        # if not deterministic:
        z = self.reparameterize(mus, logvars)
        return [self.decode(z), inp, mus, logvars]

    def mixz(self, mus, logvars):
        raise NotImplementedError()

    def logvar_mixz(self, mus, logvars):
        """approximate analytical mutual information between r(Z) and p(z|x);
        r(Z) is assumed to be an identity Gaussian"""
        kl_pZX_all = []
        for i, (mu, logvar) in enumerate(zip(mus, logvars)):
            tr_cv = torch.exp(logvar).sum(axis=1)
            log_det = torch.prod(torch.exp(logvar), axis=1).log()
            mumu = torch.bmm(mu.unsqueeze(1), mu.unsqueeze(2)).squeeze()
            kl_pZX = 0.5 * (mumu + tr_cv - self.latent_dims[i] - log_det).mean()  # self.latent_dims[i]
            kl_pZX_all.append(kl_pZX)
        return sum(kl_pZX_all)

    def logsigma_mixz(self, mus, logsigmas):
        """approximate analytical KLD between r(Z) and p(z|x);
        z is a re-parametrized p(z|f(mu),f(sigma))"""
        kl_pZX_all = torch.zeros(len(mus), device=self.device)
        for i, (mu, logsigma) in enumerate(zip(mus, logsigmas)):
            if self.rz_mode == 'parametrized':
                mu_r = self.r_mus[i]
                sigma_r = self.r_sigmas[i]
                kl_pZX = kldmg(
                    mu_r, sigma_r,
                    mu, torch.exp(logsigma),
                    self.latent_dims[i]
                )
            elif self.rz_mode == 'standard':
                pzx_covariance = torch.exp(logsigma)  # torch.exp() ?
                tr_cv = pzx_covariance.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # .mean(axis=0)
                log_det_cv = torch.det(logsigma)
                mudot = torch.bmm(mu.unsqueeze(1), mu.unsqueeze(2))
                kl_pZX = 0.5 * (mudot + tr_cv - self.latent_dims[i] - log_det_cv).mean()

            kl_pZX_all[i] = kl_pZX
        return kl_pZX_all.sum()

    def miyz(self, outputs, input, labels):
        if self.out_type == 'categorical':
            iYZ = F.nll_loss(outputs, labels)
        elif self.out_type == 'image':
            iYZ = -torch.mean(torch.sum(input * torch.log(1e-5 + outputs)
                                           + (1 - input) * torch.log(1e-5 + 1 - outputs), dim=(1,2,3)))
        elif self.out_type == 'continuous':
             iYZ = F.mse_loss(outputs, labels)
        else:
            raise NotImplementedError()
        return iYZ

    def loss_function(self, outputs, input, mus, logsigmas, labels, beta) -> dict:
        iYZ = self.miyz(outputs, input, labels)
        # analytical KL divergence between a parametrized gaussian and a normal gaussian N(0,I)
        kl_pZX = self.mixz(mus, logsigmas)
        res = {'iyz': iYZ, 'izx': kl_pZX}
        loss = iYZ + beta * kl_pZX
        res['loss'] = loss
        return res

    def fit(self, epochs, train_data_loader):
        # trains for one epoch and calculates/logs optional metrics
        avg_loss = 0.0
        for i in range(epochs):
            t0 = time.time()
            for xs, ys in train_data_loader:
                x_ = xs.to(self.device)
                y_ = ys.to(self.device)
                res, inp, mu, logsigma = self(x_)
                loss = self.loss_function(res, inp, mu, logsigma, y_, self.beta)
                avg_loss += loss['loss'].item()
                self.optimizer.zero_grad()
                loss['loss'].backward()
                self.optimizer.step()
                if self.wandb_run:
                    self.log(loss)
            avg_loss /= len(train_data_loader)
            print(f'epoch {i+1}/{epochs} loss: {avg_loss:.4f} ({time.time() - t0:.2f}s)')

    @staticmethod
    def load(path):
        kwargs, state_dict = torch.load(path)
        model = PMVIB(**kwargs)
        model.load_state_dict(state_dict)
        return model

    def save(self, path='./model'):
        state = self.state_dict()
        torch.save([self.kwargs, state], path)

    def get_model_size(self):
        size = 0
        for param in self.parameters():
            size += param.nelement()
        return size

    def sample(self, z=None, num_samples=1):
        if z is None:
            z = torch.randn(num_samples, sum(self.latent_dims))
        return self.decode(z)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def __str__(self):
        var_char = 'l'
        return \
            (f'PMIBVAE_K{",".join([str(i) for i in self.latent_dims])}'
            f'beta{self.beta}x{self.epochs}'
            f'{var_char}')