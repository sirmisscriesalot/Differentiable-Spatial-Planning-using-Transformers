from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.types import Device
from tqdm import tqdm
import argparse
import copy
import sklearn
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

RANDOM_SEED = 42
device = torch.device("cpu")
config = {}


def random_seed(seed_value, use_cuda):
    pl.seed_everything(seed_value)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# class SyntheticNavigationDataset(Dataset):
#     def __init__(self, x_file, y_file, n):
#         self.x_list = []
#         self.y_list = []
#         with open(x_file, 'rb') as fx, open(y_file, 'rb') as fy:
#             for i in range(n):
#                 self.x_list.append(np.load(fx))
#                 self.y_list.append(np.load(fy))

#     def __len__(self):
#         return len(self.x_list)

#     def __getitem__(self, idx):
#         x_tensor = torch.from_numpy(self.x_list[idx].astype(np.float32))
#         y_tensor = torch.from_numpy(self.y_list[idx].astype(np.float32))
#         sample = {'x': x_tensor, 'y': y_tensor}

#         return sample

x_file = '/data/navdata_input_30.npy', y_file = '/data/navdata_output_30.npy', n = 33000


class Dataset():
    def __init__(self, x_file, y_file, train=False, batch_size=20):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.label_dict = {'0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2}
        self.count_dic = {}
        self.inputs, self.labels = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.labels)

    def process_data(self, data):
        vecs, labels = [], []
        print(len(data))
        for label, vec in zip(list(data['label']), list(data['vec'])):
            if label not in self.label_dict:
                self.label_dict[label] = len(self.label_dict)
            label = self.label_dict[label]
            vec = vec[1:-1].split(' ')
            vec = [s.strip() for s in vec]
            vec = list(filter(None, vec))
            vec = np.array(vec).astype('float64')
            vec[np.isnan(vec)] = 0
            if np.isnan(vec).any():
                print("has nans")
            vecs.append(vec)
            labels.append(label)
        inputs = vecs
        return torch.Tensor(inputs), torch.Tensor(labels)

    def get_dataloader(self, inputs, labels, train=True):
        data = TensorDataset(inputs, labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams.update({'font.size': 8})
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    random_seed(RANDOM_SEED, True)
    config = {
        "train_output_file_id": '10WwaIDmBo2cfdHJz4O0aABN0mIbF8ByH',
        "train_output_dest_path": '/data/train_navdata_output_30.npy',
        "train_input_file_id": '1lORbh70-sTXvY48ARifexxLsYepDcHSx',
        "train_input_dest_path": '/data/train_navdata_input_30.npy',
        
        "test_output_file_id": "",
        "test_output_dest_path":'/data/test_navdata_output_30.npy',
        "test_input_file_id": "",
        "test_input_dest_path":'/data/test_navdata_input_30.npy',

        "val_output_file_id": "",
        "val_output_dest_path":'/data/val_navdata_output_30.npy',
        "val_input_file_id": "",
        "val_input_dest_path":'/data/val_navdata_input_30.npy',
        
        "batch_size": 20,
        
        "train_dataset_size": 33000,
        "test_dataset_size": 33000,
        "val_dataset_size": 33000,
    }
    gdd.download_file_from_google_drive(file_id=config["train_input_file_id"],
                                        dest_path=config["train_input_dest_path"],
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id=config["train_output_file_id"],
                                        dest_path=config["train_output_dest_path"],
                                        unzip=False)

    gdd.download_file_from_google_drive(file_id=config["test_input_file_id"],
                                        dest_path=config["test_input_dest_path"],
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id=config["test_output_file_id"],
                                        dest_path=config["test_output_dest_path"],
                                        unzip=False)

    gdd.download_file_from_google_drive(file_id=config["val_input_file_id"],
                                        dest_path=config["val_input_dest_path"],
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id=config["val_output_file_id"],
                                        dest_path=config["val_output_dest_path"],
                                        unzip=False)