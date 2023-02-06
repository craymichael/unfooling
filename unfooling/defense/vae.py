import warnings

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F  # noqa
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence
from torch.distributions import Normal

from unfooling.utils import prod


def idx2onehot(idx, n):
    if isinstance(idx, np.ndarray):
        assert np.max(idx) < n

        if idx.ndim == 1:
            idx = idx[:, None]
        onehot = np.zeros((idx.shape[0], n))
        np.put_along_axis(onehot, idx, 1, axis=1)
    else:
        assert torch.max(idx).item() < n

        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        onehot = torch.zeros(idx.size(0), n).to(idx.device)
        onehot.scatter_(1, idx, 1)

    return onehot


class VAE(nn.Module):
    """With thanks to https://github.com/Michedev/VAE_anomaly_detection

    Learning Structured Output Representation using Deep Conditional Generative
     Models
    https://papers.nips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf
    """

    def __init__(
            self,
            epsilon=0.1,
            encoder_layer_sizes=(256,),
            latent_size=16,
            reparameterize_samples=10,
            decoder_layer_sizes=(256,),
            conditional=False,
            num_labels=0,
            epochs=10,
            batch_size=64,
            learning_rate=0.001,
            print_every=100,
            condition_on_y=True,
            alt_loss=True,
            log_var=False,
    ):
        super().__init__()

        if conditional:
            assert num_labels > 0
        elif num_labels != 0:
            warnings.warn(f'num_labels={num_labels} but conditional=False, '
                          'setting num_labels=0')
            num_labels = 0

        self.encoder = self.decoder = None

        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.num_labels = num_labels
        self.latent_size = latent_size
        self.conditional = conditional
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.epsilon = epsilon
        self.condition_on_y = condition_on_y
        self.reparameterize_samples = reparameterize_samples
        self.alt_loss = alt_loss
        self.log_var = log_var

        if self.alt_loss:
            self.prior = Normal(0, 1)
        else:
            self.prior = None  # N/A

        self.threshold_ = None

    def init_modules(self, X, y):
        if y is None:
            input_size = 0
        else:
            input_size = prod(y.shape[1:])
        # named for compat
        num_labels_X = prod(X.shape[1:])
        if self.condition_on_y:
            input_size, num_labels_X = num_labels_X, input_size
        encoder_layer_sizes = [input_size, *self.encoder_layer_sizes]
        self.encoder = VAEEncoder(
            encoder_layer_sizes, self.latent_size, self.conditional,
            num_labels_X
        )
        decoder_layer_sizes = [*self.decoder_layer_sizes, input_size]

        if self.alt_loss:
            # times 2 because this is the concatenated vector of reconstructed
            #  mean and variance
            decoder_layer_sizes[-1] *= 2
            decoder_kwargs = {'last_activation': None}
        else:
            decoder_kwargs = {}
        self.decoder = VAEDecoder(
            decoder_layer_sizes, self.latent_size, self.conditional,
            num_labels_X, **decoder_kwargs
        )

    # noinspection PyPep8Naming
    def _handle_Xy(self, X, y, shuffle=False):
        if self.conditional:
            assert X.ndim == 2
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        if y is not None:
            if y.ndim == 2:
                y = y.squeeze(axis=1)
            else:
                assert y.ndim == 1, 'y.ndim needs to be 1 or 2'

            if str(y.dtype).lower().startswith('int') and y.ndim == 1:
                y = idx2onehot(y, n=self.num_labels)

            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()

            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)
        data_loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader, X, y

    def fit(self, X, y=None):
        y_is_None = y is None
        if y_is_None:
            assert self.condition_on_y, 'invalid options'
            assert not self.conditional, 'invalid options'
        data_loader, X, y = self._handle_Xy(X, y, shuffle=True)
        self.init_modules(X, y)
        y_orig = y
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def loss_fn(recon_y_, y_, mean_, log_var_, dist_):
            if self.alt_loss:
                # https://github.com/Michedev/VAE_anomaly_detection
                recon_mu, recon_sigma = recon_y_.chunk(2, dim=1)
                if self.log_var:
                    recon_sigma = torch.exp(0.5 * recon_sigma)
                else:
                    recon_sigma = F.softplus(recon_sigma)
                recon_mu = recon_mu.view(self.reparameterize_samples,
                                         *y_.shape)
                recon_sigma = recon_sigma.view(self.reparameterize_samples,
                                               *y_.shape)

                # average over sample dimension
                y_ = y_.unsqueeze(0)
                # log_lik = Normal(recon_mu, recon_sigma).log_prob(y_).mean(dim=0)
                log_lik = Normal(
                    recon_mu, 1e-16 + torch.maximum(
                        recon_sigma,
                        torch.zeros_like(recon_sigma,
                                         dtype=recon_sigma.dtype))
                ).log_prob(y_).mean(dim=0)
                log_lik = log_lik.mean(dim=0).sum()
                kl = kl_divergence(dist_, self.prior).mean(dim=0).sum()
                loss_ = kl - log_lik
            else:
                y_ = torch.repeat_interleave(y_, self.reparameterize_samples,
                                             dim=0)
                BCE = F.binary_cross_entropy(
                    recon_y_.view(recon_y_.size(0), -1),
                    y_.view(y_.size(0), -1),
                    reduction='sum',
                )
                if self.log_var:
                    KLD = -0.5 * torch.sum(
                        1 + log_var_ - mean_.pow(2) - log_var_.exp())
                else:
                    KLD = -0.5 * torch.sum(
                        1 + torch.log(log_var_) - mean_.pow(2) - log_var_)
                loss_ = (BCE + KLD) / y_.size(0)
            return loss_

        vae = self.to(device)

        optimizer = torch.optim.Adam(vae.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            for iteration, xy in enumerate(data_loader):
                if y_is_None:
                    x, y = xy[0].to(device), None
                else:
                    x, y = xy[0].to(device), xy[1].to(device)

                if self.condition_on_y:
                    x, y = y, x

                if self.conditional:
                    recon_y, mean, log_var, z, dist = vae(y, x)
                else:
                    recon_y, mean, log_var, z, dist = vae(y)

                loss = loss_fn(recon_y, y, mean, log_var, dist)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iteration % self.print_every == 0 or
                        iteration == len(data_loader) - 1):
                    print('Epoch {:02d}/{:02d} Batch {:04d}/{:d}, '
                          'Loss {:9.4f}'.format(epoch, self.epochs, iteration,
                                                len(data_loader) - 1,
                                                loss.item()))

        # determine threshold
        scores = self.score_samples(X, y_orig)
        scores.sort()

        thresh_idx = round(self.epsilon * len(X))
        self.threshold_ = scores[thresh_idx]

    def score_samples(self, X, y=None):
        y_is_None = y is None
        data_loader, X, y = self._handle_Xy(X, y, shuffle=False)
        with torch.no_grad():
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

            vae = self.to(device)

            all_probs = []
            for iteration, xy in enumerate(data_loader):
                if y_is_None:
                    x, y = xy[0].to(device), None
                else:
                    x, y = xy[0].to(device), xy[1].to(device)

                if self.condition_on_y:
                    x, y = y, x

                if self.conditional:
                    recon_y, mean, log_var, z, dist = vae(y, x)
                else:
                    recon_y, mean, log_var, z, dist = vae(y)

                if self.alt_loss:
                    recon_mu, recon_sigma = recon_y.chunk(2, dim=1)
                    if self.log_var:
                        recon_sigma = torch.exp(0.5 * recon_sigma)
                    else:
                        recon_sigma = F.softplus(recon_sigma)
                    recon_mu = recon_mu.view(self.reparameterize_samples,
                                             *y.shape)
                    recon_sigma = recon_sigma.view(self.reparameterize_samples,
                                                   *y.shape)

                    # average over sample dimension
                    recon_dist = Normal(
                        recon_mu, 1e-16 + torch.maximum(
                            recon_sigma,
                            torch.zeros_like(recon_sigma,
                                             dtype=recon_sigma.dtype)))
                    y = y.unsqueeze(0)
                    probs = recon_dist.log_prob(y).exp().mean(dim=0).mean(
                        dim=-1)
                else:
                    recon_y = recon_y.view(self.reparameterize_samples,
                                           *y.shape)
                    recon_y = torch.mean(recon_y, dim=0)
                    probs = -torch.mean(
                        F.mse_loss(recon_y, y, reduction='none'), dim=1)
                all_probs.append(probs.cpu().detach().numpy())

        return np.concatenate(all_probs, axis=0)

    def forward(self, x, c=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if self.conditional:
            if c.dim() > 2:
                c = c.view(c.size(0), -1)

        means, log_var = self.encoder(x, c)
        z, dist = self.reparameterize(means, log_var)
        if self.conditional:
            c = torch.repeat_interleave(c, self.reparameterize_samples, dim=0)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z, dist

    def reparameterize(self, mu, log_var):
        if self.log_var:
            std = torch.exp(0.5 * log_var)
        else:
            std = F.softplus(log_var)
        dist = Normal(mu, 1e-16 + torch.maximum(
            std, torch.zeros_like(std, dtype=std.dtype)))
        # shape: [reparameterize_samples, batch_size, latent_size]
        z = dist.rsample([self.reparameterize_samples])
        z = z.view(-1, self.latent_size)
        return z, dist

    def inference(self, z, c=None):
        if self.conditional:
            c = torch.repeat_interleave(c, self.reparameterize_samples, dim=0)
        recon_x = self.decoder(z, c)
        return recon_x


class VAEEncoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name='L{:d}'.format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name='A{:d}'.format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class VAEDecoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, num_labels,
                 last_activation=nn.Sigmoid()):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        self.num_labels = num_labels
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(
                zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name='L{:d}'.format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name='A{:d}'.format(i), module=nn.ReLU())
            elif last_activation is not None:
                self.MLP.add_module(name='sigmoid', module=last_activation)

    def forward(self, z, c):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
