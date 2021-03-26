# encoding: utf-8
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from fsvae.utils import init_weights
    from fsvae.config import DEVICE

except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class Generator(nn.Module):

    def __init__(self, latent_dim=50, x_shape=(1, 28, 28), cshape=(128, 7, 7), verbose=False):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ishape = cshape
        self.iels = int(np.prod(self.ishape))
        self.x_shape = x_shape
        self.output_channels = x_shape[0]
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(True),

            nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.ReLU(True),

            Reshape(self.ishape),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.output_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, z1, z2):

        z = torch.cat((z1, z2), dim=1)
        gen_img = self.model(z)
        return gen_img.view(z.size(0), *self.x_shape)


class Encoder(nn.Module):

    def __init__(self,
                 input_channels=1,
                 output_channels=64,
                 cshape=(128, 7, 7),
                 c_feature=7,
                 r=9,
                 adding_outlier=False,
                 verbose=False):
        super(Encoder, self).__init__()

        self.cshape = cshape
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.c_feature = c_feature
        self.r = r if not adding_outlier else 25
        self.verbose = verbose

        self.first_layer = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.model = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(1024, self.output_channels)
        self.con = nn.Linear(1024, self.output_channels)
        self.v = nn.Linear(1024, 1)

        init_weights(self)
        self.v.weight.data.uniform_(0.01, 0.03)
        if self.verbose:
            print(self.model)

    def _s_re(self, mu, log_sigma, v):

        sigma = torch.exp(log_sigma * 0.5)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(DEVICE)
        e = torch.tensor(np.random.normal(0, 1, sigma.size()), dtype=torch.float).to(DEVICE)
        # sample from Gamma distribution
        z1 = (v / 2 - 1 / 3) * torch.pow(1 + (e / torch.sqrt(9 * v / 2 - 3)), 3)
        # reparameterization trick
        z2 = torch.sqrt(v / (2 * z1))
        z = mu + sigma * z2 * std_z

        return z, mu, log_sigma - torch.log(z1)

    def _forward(self, x):

        x = self.model(x)
        mu = self.mu(x)
        log_sigma = self.con(x)

        mu1 = mu[:, :self.c_feature]
        mu2 = mu[:, self.c_feature:]
        log_sigma1 = log_sigma[:, :self.c_feature]
        log_sigma2 = log_sigma[:, self.c_feature:]
        v = F.softplus(self.v(x)) + self.r

        z1, mu1, log_sigma1 = self._s_re(mu1, log_sigma1, v)
        # z1 = mu1 + (torch.exp(log_sigma1 * 0.5) * torch.randn_like(mu1))
        z2 = mu2 + (torch.exp(log_sigma2 * 0.5) * torch.randn_like(mu2))
        # z2, mu2, log_sigma2 = self._s_re(mu2, log_sigma2, v)

        return z1, z2, mu1, mu2, log_sigma1, log_sigma2

    def moex(self, x1, x2, dim=1):

        mean1 = x1.mean(dim=dim, keepdim=True)
        mean2 = x2.mean(dim=dim, keepdim=True)

        var1 = x1.var(dim=dim, keepdim=True)
        var2 = x2.var(dim=dim, keepdim=True)

        std1 = (var1 + 1e-5).sqrt()
        std2 = (var2 + 1e-5).sqrt()

        x1 = std2 * ((x1 - mean1) / std1) + mean2
        x2 = std1 * ((x2 - mean2) / std2) + mean1
        return x1, x2

    def forward(self, x, a_x=None, argument=False):
        # x.shape is [64, 1, 28, 28]
        x = self.first_layer(x)
        argumented_z = None
        if a_x is not None:
            a_x = self.first_layer(a_x)
            if argument:
                x, _ = self.moex(x, a_x)
            argumented_z = self._forward(a_x)

        original_z = self._forward(x)
        return original_z, argumented_z


class GMM(nn.Module):

    def __init__(self, n_cluster=10, n_features=64):
        super(GMM, self).__init__()

        self.n_cluster = n_cluster
        self.n_features = n_features

        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_cluster, ).fill_(1) / self.n_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.n_cluster, self.n_features).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.n_cluster,
                                                           self.n_features).fill_(0), requires_grad=True)

    def predict(self, z):

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def sample_by_k(self, k, num=10, latent_dim=30):

        mu = self.mu_c[k].data.cpu().numpy()
        sigma = self.log_sigma2_c[k].exp().data.cpu().numpy()
        z1 = mu + np.random.randn(num, self.n_features) * np.sqrt(sigma)

        z2 = torch.normal(0, 1, size=(num, latent_dim - self.n_features)).to(DEVICE)
        return torch.from_numpy(z1).to(DEVICE).float(), z2

    def get_asign(self, z):

        det = 1e-10
        pi = self.pi_
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c

        yita = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

        yita_c = yita / (yita.sum(1).view(-1, 1))  # batch_size*Clusters

        pred = torch.argmax(yita_c, dim=1, keepdim=True)
        oh = torch.zeros_like(yita_c)
        oh = oh.scatter_(1, pred, 1.)
        return yita_c, oh

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):

        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c+1, :], log_sigma2s[c:c+1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):

        return -0.5*(torch.sum(torch.tensor(np.log(np.pi*2), dtype=torch.float).to(DEVICE) +
                               log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2), 1))
