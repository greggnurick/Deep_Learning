import torch
from torch.utils import data
import numpy as np
import pickle


class Salicon(data.Dataset):
    def __init__(self, dataset_path):
        self.dataset = pickle.load(open(dataset_path, 'rb'))

    def __getitem__(self, index):
        img = self.dataset[index]['X']
        img = torch.from_numpy(img.astype(np.float32))
        sal_map = self.dataset[index]['y']
        sal_map = torch.from_numpy(sal_map.astype(np.float32)).view(48*48)
        return img, sal_map

    def __len__(self):
        return len(self.dataset)
