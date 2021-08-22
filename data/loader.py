import os

from torch.utils import data
from torchvision import transforms

from data.dataset import CaptchaDataset


def get_dataloader(dataset_path, batch_size, dataloader_mode='train', num_workers=4, **kwargs):
    transform_list = []
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if dataloader_mode == 'train':
        if kwargs['dataset_augmentation']:
            transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    transform = transforms.Compose(transform_list)
    dataset = CaptchaDataset(os.path.join(dataset_path, dataloader_mode),
                             num_chars=kwargs['num_chars'],
                             alphabet=kwargs['alphabet'], transform=transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)
