# encoding: utf-8
try:
    import os
    import argparse

    import torch
    import numpy as np
    import torch.nn.functional as F

    from sklearn.mixture import GaussianMixture
    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    from itertools import chain
    from sklearn.metrics import normalized_mutual_info_score as NMI

    from fsvae.datasets import dataset_list, get_dataloader
    from fsvae.config import RUNS_DIR, DATASETS_DIR, DEVICE, DATA_PARAMS
    from fsvae.model import Generator, GMM, Encoder
    from fsvae.utils import save_images, cluster_acc, gmm_Loss, mse

except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="fsvae", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=500, type=int, help="Number of epochs")
    parser.add_argument("-o", "--outlier", dest="outlier_rate", default=0, type=float, help="ratios of outlier")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="v1")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name, args.version_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')
    log_path = os.path.join(run_dir, 'logs')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # -----train-----
    # train detail var
    n_epochs = args.n_epochs
    b1 = 0.5
    b2 = 0.99

    data_params = DATA_PARAMS[dataset_name]
    train_batch_size, latent_dim, c_feature, picture_size, cshape, data_size, train_lr, test_num, n_cluster = data_params
    print(data_params)

    # net
    gen = Generator(latent_dim=latent_dim, x_shape=picture_size, cshape=cshape)
    gmm = GMM(n_cluster=n_cluster, n_features=c_feature)
    encoder = Encoder(
        input_channels=picture_size[0],
        output_channels=latent_dim,
        cshape=cshape,
        c_feature=c_feature,
        r=9,
        adding_outlier=(args.outlier_rate > 0.05)
    )

    # parallel
    if torch.cuda.device_count() > 1:
        print("this GPU have {} core".format(torch.cuda.device_count()))

    # set device: cuda or cpu
    gen.to(DEVICE)
    encoder.to(DEVICE)
    gmm.to(DEVICE)

    # optimization
    gen_enc_gmm_ops = torch.optim.Adam(chain(
        gen.parameters(),
        encoder.parameters(),
        gmm.parameters(),
    ), lr=train_lr, betas=(b1, b2))
    lr_s = StepLR(gen_enc_gmm_ops, step_size=10, gamma=0.95)

    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                batch_size=train_batch_size, outlier_rate=args.outlier_rate)
    i_dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                batch_size=30000, shuffle=False, outlier_rate=0)

    # =============================================================== #
    # =========================initialization======================== #
    # =============================================================== #
    best_model = None
    best_score = 0
    print('searching best Gaussian mixture prior...')
    datas, labels = i_dataloader.dataset.data, i_dataloader.dataset.targets
    if dataset_name in ['cifar10', 'ytf', 'gtsrb']:
        datas = datas.permute((0, 3, 1, 2)).to(DEVICE) / 255.0
    else:
        datas = datas.unsqueeze(1).to(DEVICE) / 255.0

    Z = []
    Y = []
    with torch.no_grad():
        z, _ = encoder(datas)
        Z.append(z[2])
        Y.append(labels)

    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().numpy()
    for i in range(30):
        _gmm = GaussianMixture(n_components=n_cluster, covariance_type='diag')
        pre = _gmm.fit_predict(Z)
        acc = cluster_acc(pre, Y)[0] * 100

        if best_score < acc:
            best_score = acc
            best_model = _gmm
    datas = datas.cpu()
    torch.cuda.empty_cache()
    del datas

    print('best accuracy is: {:.4f}'.format(best_score))
    gmm.pi_.data = torch.from_numpy(best_model.weights_).to(DEVICE).float()
    gmm.mu_c.data = torch.from_numpy(best_model.means_).to(DEVICE).float()
    gmm.log_sigma2_c.data = torch.log(torch.from_numpy(best_model.covariances_).to(DEVICE).float())

    # =============================================================== #
    # ============================training=========================== #
    # =============================================================== #
    print('begin training...')
    epoch_bar = tqdm(range(0, n_epochs))
    best_acc, best_nmi, best_ite = 0, 0, 0
    gen_weight = 0.15
    for epoch in epoch_bar:

        g_t_loss = 0
        for index, (real_images, augmented_images, target) in enumerate(dataloader):
            real_images, augmented_images, target = real_images.to(DEVICE), \
                                                     augmented_images.to(DEVICE), \
                                                     target.to(DEVICE)

            gen.train()
            gmm.train()
            encoder.train()
            gen_enc_gmm_ops.zero_grad()

            original_z, augmented_z = encoder(real_images, augmented_images, argument=(epoch > 50))
            fake_images = gen(original_z[0], gen_weight * original_z[1])

            rec_loss = torch.mean(
                torch.sum(
                    F.binary_cross_entropy(fake_images, real_images, reduction='none'), dim=[1, 2, 3]
                )
            )

            augmented_loss = mse(original_z[2], augmented_z[2]) + \
                             mse(original_z[3], augmented_z[3])

            c_loss = mse(gmm.get_asign(original_z[0])[0], gmm.get_asign(augmented_z[0])[0])

            kl_loss = torch.mean(
                -0.5 * torch.sum(
                    1 + original_z[5] - original_z[3] ** 2 - original_z[5].exp(), dim=1
                ), dim=0
            )

            kls_loss = gmm_Loss(original_z[0], original_z[2], original_z[4], gmm)

            # the loss weight of augmentation module is used for all data sets.
            g_loss = 10 * (augmented_loss + c_loss) + kls_loss + rec_loss + kl_loss

            g_loss.backward()

            gen_enc_gmm_ops.step()
            g_t_loss += g_loss

        # save cheekpoint model
        if (epoch + 1) % 20 == 0:
            cheek_path = os.path.join(models_dir, "cheekpoint_{}".format(epoch))
            os.makedirs(cheek_path, exist_ok=True)
            torch.save(gen.state_dict(), os.path.join(cheek_path, 'gen.pkl'))
            torch.save(encoder.state_dict(), os.path.join(cheek_path, 'enc.pkl'))
            torch.save(gmm.state_dict(), os.path.join(cheek_path, 'gmm.pkl'))

        print(
            'rec_loss: {:.4f}, a_loss: {:.4f}, c_loss: {:.4f}, kls_loss: {:.4f}, kl_loss: {:.4f}'.format(
                rec_loss, augmented_loss, c_loss, kls_loss, kl_loss
            )
        )
        lr_s.step()

        # =============================================================== #
        # ==============================test============================= #
        # =============================================================== #
        gen.eval()
        encoder.eval()
        gmm.eval()

        with torch.no_grad():
            Z = []
            Y = []
            for _data, _, _target in i_dataloader:
                z, _ = encoder(_data.to(DEVICE))
                Z.append(z[0])
                Y.append(_target)

            Z = torch.cat(Z, 0)
            Y = torch.cat(Y, 0).detach().numpy()
            _pred = gmm.predict(Z)
            _acc = cluster_acc(_pred, Y)[0] * 100
            _nmi = NMI(_pred, Y)

            t_pred = _pred[:-test_num]
            t_acc = cluster_acc(t_pred, Y[:-test_num])[0] * 100
            t_nmi = NMI(t_pred, Y[:-test_num])

            if best_acc < t_acc:
                best_acc, best_nmi, best_ite = t_acc, t_nmi, epoch

            stack_images = None
            for k in range(n_cluster):

                z1, z2 = gmm.sample_by_k(k, latent_dim=latent_dim)
                fake_images = gen(z1, gen_weight * z2)

                if stack_images is None:
                    stack_images = fake_images[:n_cluster].data.cpu().numpy()
                else:
                    stack_images = np.vstack((stack_images, fake_images[:n_cluster].data.cpu().numpy()))
            stack_images = torch.from_numpy(stack_images)
            save_images(stack_images, imgs_dir, 'test_gen_{}'.format(epoch), nrow=n_cluster)

            logger = open(os.path.join(log_path, "log.txt"), 'a')
            logger.write(
                "[FSVAE]: epoch: {}, g_loss: {:.4f}, acc: {:.4f}%, nmi: {:.4f}, t_acc: {:.4f}%, t_nmi: {:.4f}\n".format
                (
                    epoch, g_t_loss / len(dataloader), _acc, _nmi, t_acc, t_nmi
                )
            )
            logger.close()
            print("[FSVAE]: epoch: {}, g_loss: {:.4f}, acc: {:.4f}%, nmi: {:.4f}, t_acc: {:.4f}%, t_nmi: {:.4f}".format
                  (
                epoch, g_t_loss / len(dataloader), _acc, _nmi, t_acc, t_nmi
            ))

    print('complete training...best_acc is: {:.4f}, best_nmi is: {:.4f}, iteration is: {}'.format(
        best_acc, best_nmi, best_ite,
    ))
    torch.save(gen.state_dict(), os.path.join(models_dir, 'gen.pkl'))
    torch.save(encoder.state_dict(), os.path.join(models_dir, 'enc.pkl'))
    torch.save(gmm.state_dict(), os.path.join(models_dir, 'gmm.pkl'))


if __name__ == '__main__':

    main()
