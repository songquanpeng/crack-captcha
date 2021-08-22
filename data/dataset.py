import os

from PIL import Image
from torch.utils import data
import torch
from utils.file import list_all_images


class CaptchaDataset(data.Dataset):
    def __init__(self, root, num_chars, alphabet, transform=None):
        self.samples = list_all_images(root)
        self.samples.sort()
        self.num_chars = num_chars
        self.alphabet = alphabet
        self.char2index = {}
        for i, char in enumerate(alphabet):
            self.char2index[char] = i
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = os.path.splitext(os.path.basename(path))[0]
        assert len(label) == self.num_chars
        label = list(map(lambda char: self.char2index[char], label))
        label = torch.LongTensor(label)
        return img, label

    def __len__(self):
        return len(self.samples)
