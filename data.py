
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def get_labels(fname):
    with open(fname,'r') as f:
        labels = [t.strip() for t in f.read().split(',')]
    labels2idx = {t:i for i,t in enumerate(labels)}
    idx2labels = {i:t for i,t in enumerate(labels)}
    return labels,labels2idx,idx2labels

class PlanetData(Dataset):

    def __init__(self, csv_file, root_dir, labels_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.labels, self.labels2idx, self.idx2labels = get_labels(labels_file)
        self.n_labels = len(self.labels)
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.ix[idx, 0])
        img = Image.open(img_name + '.jpg').convert('RGB')
        labels = self.data.ix[idx, 1]
        target = torch.zeros(self.n_labels)
        label_idx = torch.LongTensor([self.labels2idx[tag] for tag in labels.split(' ')])
        target[label_idx] = 1
        if self.transform:
            img = self.transform(img)
        return img, target
