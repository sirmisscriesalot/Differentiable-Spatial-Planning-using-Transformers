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

from model.pos_encod import PositionalEncoding
from model.DSPT import DSPT

device = torch.device("cpu")
config = {
    "RANDOM_SEED": 42,
    "train_output_file_id": '10WwaIDmBo2cfdHJz4O0aABN0mIbF8ByH',
    "train_output_dest_path": './data/train_navdata_output_30.npy',
    "train_input_file_id": '1lORbh70-sTXvY48ARifexxLsYepDcHSx',
    "train_input_dest_path": './data/train_navdata_input_30.npy',

    "test_output_file_id": "",
    "test_output_dest_path": './data/test_navdata_output_30.npy',
    "test_input_file_id": "",
    "test_input_dest_path": './data/test_navdata_input_30.npy',

    "val_output_file_id": "",
    "val_output_dest_path": './data/val_navdata_output_30.npy',
    "val_input_file_id": "",
    "val_input_dest_path": './data/val_navdata_input_30.npy',

    "batch_size": 20,

    "train_dataset_size": 33000,
    "test_dataset_size": None,
    "val_dataset_size": None,
}


def random_seed(seed_value, use_cuda):
    pl.seed_everything(seed_value)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Dataset():
    def __init__(self, x_file, y_file, dataset_size, train=False, batch_size=20):
        self.xfile = x_file
        self.yfile = y_file
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.train = train
        print("Loading data...")
        self.inputs, self.labels = self.process_data()
        self.DataLoader = self.get_dataloader(self.inputs, self.labels)
        print("Data loaded")

    def process_data(self):
        self.x_list = []
        self.y_list = []
        with open(self.x_file, 'rb') as fx, open(self.y_file, 'rb') as fy:
            for _ in tqdm(range(self.dataset_size)):
                self.x_list.append(np.load(fx))
                self.y_list.append(np.load(fy))
        return torch.Tensor(self.x_list), torch.Tensor(self.y_list)

    def get_dataloader(self, inputs, labels, train=True):
        data = TensorDataset(inputs, labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)

# complete this  and move it to utils
##  use this https://torchmetrics.readthedocs.io/en/stable/pages/implement.html 
def custom_accuracy(preds, labels):
    pass 
class MLP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DSPT()
        self.criterion = nn.MSELoss()
        print("Init Done")

    def setup(self, stage):
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
        print("Setup Done")
    
    def train_dataloader(self):
        train_dataset = Dataset(x_file=self.config['train_input_dest_path'],y_file=self.config['train_output_dest_path'],dataset_size=self.config['train_dataset_size'],batch_size=self.config['batch_size'],train=True)
        print("Train DataLoader Done")
        return train_dataset.DataLoader

    def val_dataloader(self):
        val_dataset = Dataset(x_file=self.config['val_input_dest_path'],y_file=self.config['val_output_dest_path'],dataset_size=self.config['val_dataset_size'],batch_size=self.config['batch_size'],train=True)
        print("Validation DataLoader Done")
        return val_dataset.DataLoader
    
    def test_dataloader(self):
        test_dataset = Dataset(x_file=self.config['test_input_dest_path'],y_file=self.config['test_output_dest_path'],dataset_size=self.config['test_dataset_size'],batch_size=self.config['batch_size'],train=False)
        print("Test DataLoader Done")
        return test_dataset.DataLoader

    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.config)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y=y.long()
        loss = self.criterion(y_hat, y)
        y_pred= torch.argmax(y_hat, 1)
        acc = custom_accuracy(y_pred, y)
        self.log("train_loss",loss,on_step=False, on_epoch=True)
        self.log("train_acc",acc,on_step=False,on_epoch=True)
        return loss

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams.update({'font.size': 8})
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    random_seed(config["RANDOM_SEED"], True)
