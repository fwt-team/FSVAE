# encoding: utf-8
try:
    import math
    import torch
    import argparse
    import numpy as np
    import torch.nn.functional as F
    import torch.nn as nn

    from torchvision.utils import save_image
    from scipy.optimize import linear_sum_assignment as linear_assignment

except ImportError as e:
    print(e)
    raise ImportError


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_params(data, a=0, mode='fan_in', nonlinearity='relu', type='1', std=0.02, gain=1):

    if type == 'k':
        fan = nn.init._calculate_correct_fan(data, mode)
        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)

    if type == 'x':
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(data)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    # print("std: {}".format(std))
    with torch.no_grad():
        return data.normal_(0, std)


def init_weights(net):

    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            init_params(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            init_params(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            init_params(m.weight)
            m.bias.data.zero_()


def save_images(gen_imgs, imags_path, images_name, nrow=5):

    save_image(gen_imgs.data[:nrow * nrow],
               '%s/%s.png' % (imags_path, images_name),
               nrow=nrow, normalize=True)


def cluster_acc(Y_pred, Y):

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total*1.0/Y_pred.size, w


def gmm_Loss(z, z_mu, z_sigma2_log, gmm):

    det = 1e-10
    pi = gmm.pi_
    mu_c = gmm.mu_c
    log_sigma2_c = gmm.log_sigma2_c

    yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + gmm.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

    yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

    Loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                          torch.exp(z_sigma2_log.unsqueeze(1) -
                                                                    log_sigma2_c.unsqueeze(0)) +
                                                          (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) /
                                                          torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

    Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * \
            torch.mean(torch.sum(1 + z_sigma2_log, 1))
    return Loss


def mse(origin, target):

    loss = F.mse_loss(origin, target)

    return loss
