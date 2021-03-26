# encoding: utf-8
import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Local directory of CypherCat API
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR = os.path.split(ROOT_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')

# Local directory for runs
RUNS_DIR = os.path.join(REPO_DIR, 'runs')

# difference datasets config
# train_batch_size, latent_dim, picture_size, cshape, all_data_size, train_lr, test_num, n_cluster
DATA_PARAMS = {
    'mnist': (64, 10, 7, (1, 28, 28), (128, 7, 7), 60000, 2e-3, 10000, 10),
}

