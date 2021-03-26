# encoding: utf-8
try:
    import torch
    import torchvision.datasets as dset
    import numpy as np
    import torchvision.transforms as transforms

    from torch.utils.data import Dataset
    from PIL import ImageChops, Image, ImageEnhance

except ImportError as e:
    print(e)
    raise ImportError


class MNIST(dset.MNIST):

    def __init__(self, dataset_path, download=True, train=True, transform=None, angle=25, offset_value=3.5):
        super(MNIST, self).__init__(dataset_path,
                                    download=download,
                                    train=train,
                                    transform=transform)
        self.angle = angle
        self.offset_value = offset_value

    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        augmented_image = img.copy().rotate(np.random.random() * 2 * self.angle - self.angle)
        offset1 = np.random.randint(-self.offset_value, self.offset_value)
        offset2 = np.random.randint(-self.offset_value, self.offset_value)
        augmented_image = ImageChops.offset(augmented_image, offset1, offset2)
        augmented_image = ImageEnhance.Contrast(augmented_image).enhance(1.2)

        if self.transform is not None:
            img = self.transform(img)
            augmented_image = torch.tensor(np.array(augmented_image)).unsqueeze(0) / 255.0

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, augmented_image, target

DATASET_FN_DICT = {
    'mnist': MNIST,
}

dataset_list = DATASET_FN_DICT.keys()


def _get_dataset(dataset_name='mnist'):

    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


def get_all_data(train, test):

    if not isinstance(train.data, torch.Tensor):
        train.data, train.targets, test.data, test.targets \
            = \
            torch.tensor(train.data), \
            torch.tensor(train.targets), \
            torch.tensor(test.data), \
            torch.tensor(test.targets)

    train.data = torch.cat((train.data, test.data))
    train.targets = torch.cat((train.targets, test.targets))

    return train


def inject_outlier(data, outlier_rate):

    assert outlier_rate >= 0

    if outlier_rate == 0:
        return data
    else:
        size = data.data.shape
        outlier_count = int(60000 * outlier_rate)
        black = np.random.uniform(0, 10, size=(outlier_count // 2, *size[1:]))
        white = np.random.uniform(245, 255, size=(outlier_count // 2, *size[1:]))

        outlier = torch.tensor(np.vstack((black, white)), dtype=torch.uint8)
        labels = torch.tensor(np.ones(outlier.shape[0], dtype=np.int) * -1)
        data.data = torch.cat((data.data, outlier))
        data.targets = torch.cat((data.targets, labels))

        return data


# get the loader of all datas
def get_dataloader(dataset_path='../datasets/mnist',
                   dataset_name='mnist', shuffle=True, batch_size=50, outlier_rate=0):
    dataset = _get_dataset(dataset_name)

    transform = [
        transforms.ToTensor(),
    ]
    train_data = dataset(dataset_path, download=True, train=True, transform=transforms.Compose(transform))
    test_data = dataset(dataset_path, download=True, train=False, transform=transforms.Compose(transform))
    dataset = get_all_data(train_data, test_data)
    dataset = inject_outlier(dataset, outlier_rate)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader
