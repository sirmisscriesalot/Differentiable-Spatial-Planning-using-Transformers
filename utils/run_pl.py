from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import random
import math
from torch import autograd,Tensor
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
from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# from model.pos_encod import PositionalEncoding
# from model.DSPT import DSPT

device = torch.device("cpu")
config = {
    "RANDOM_SEED": 42,
    "train_output_file_id": '10WwaIDmBo2cfdHJz4O0aABN0mIbF8ByH',
    "train_output_dest_path": './data/train_navdata_output_30.npy',
    "train_input_file_id": '1lORbh70-sTXvY48ARifexxLsYepDcHSx',
    "train_input_dest_path": './data/train_navdata_input_30.npy',

    "test_output_file_id": "1xq51a8cKLQvu4CYUmuuMoAxX7OBJackO",
    "test_output_dest_path": './data/test_navdata_output_30.npy',
    "test_input_file_id": "1Du8uQYHUZ34cF7IqQso6l-et_7LBm1TY",
    "test_input_dest_path": './data/test_navdata_input_30.npy',

    "val_output_file_id": "1GEBSvmh9P3NbaY_DmHMEnUNlgPovqJaJ",
    "val_output_dest_path": './data/val_navdata_output_30.npy',
    "val_input_file_id": "1UwEjF4asdcFwLJpijztFHAVq_dkWlie8",
    "val_input_dest_path": './data/val_navdata_input_30.npy',

    "batch_size": 20,
    "epochs": 40,
    "learning_rate": 1,

    "train_dataset_size": 33000,
    "test_dataset_size": 5000,
    "val_dataset_size": 5000,
}


def random_seed(seed_value, use_cuda):
    pl.seed_everything(seed_value)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Dataset():
    def __init__(self, x_file, y_file, dataset_size, train=False, batch_size=20):
        self.x_file = x_file
        self.y_file = y_file
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
        self.x_list=np.array(self.x_list)
        self.y_list=np.array(self.y_list)
        return torch.Tensor(self.x_list), torch.Tensor(self.y_list)

    def get_dataloader(self, inputs, labels, train=True):
        data = TensorDataset(inputs, labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)

# complete this  and move it to utils
# use this https://torchmetrics.readthedocs.io/en/stable/pages/implement.html

class PositionalEncoding(nn.Module):

  def __init__(self,d_model=64,max_len=900):
    super().__init__()
  
  def forward(self,x,d_model=64):
    max_len = x.size(2)
    pe = torch.zeros(1,d_model,max_len)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(0)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_len) / d_model)).unsqueeze(1)
    pe[0,0::2,:] = torch.sin(torch.matmul(div_term,position))
    pe[0,1::2,:] = torch.cos(torch.matmul(div_term,position))

    self.register_buffer('pe',pe)
    x = x.to(device) + self.pe[:x.size(0)].to(device)
    return x

class CNNEncoding(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=1)
    self.conv2 = nn.Conv2d(in_channels = 64,out_channels=64, kernel_size=1)
    self.flatten = nn.Flatten(start_dim=2)

  def init_weights(self) -> None:
    nn.init.kaiming_uniform_(self.conv1.weight.data,nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight.data,nonlinearity='relu')

  def forward(self,x) -> Tensor:
    x = F.relu(self.conv1(x))
    x = self.conv2(x)
    x = self.flatten(x)
    
    return x

class DSPT(nn.Module):
  def __init__(self,d_model=64,nhead=8,d_hid=512,nlayers=5,dropout=0.1):
    super().__init__()
    self.model_type = 'DSPT'
    self.pos_encoder = PositionalEncoding()
    self.conv_encoder = CNNEncoding()
    encoder_layers = nn.TransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers,nlayers)
    self.d_model = d_model
    self.decoder = nn.Linear(d_model,1)

  def init_weights(self) -> None:
    # initrange = 0.1
    # self.encoder.weight.data.uniform_(-initrange, initrange)
    nn.init.kaiming_uniform_(self.encoder.weight.data,nonlinearity='relu')
    self.decoder.bias.data.zero_()
    # self.decoder.weight.data.uniform_(-initrange, initrange)
    nn.init.kaiming_uniform_(self.decoder.weight.data,nonlinearity='relu')

  def forward(self, src) -> Tensor:
    src = self.conv_encoder(src)
    src = self.pos_encoder(src)
    src = torch.transpose(src,1,2)
    output = self.transformer_encoder(src)
    output = self.decoder(output)

    return output


def custom_accuracy(preds, labels):
    return 1


class WrappedModel(pl.LightningModule):
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
        train_dataset = Dataset(x_file=self.config['train_input_dest_path'], y_file=self.config['train_output_dest_path'],
                                dataset_size=self.config['train_dataset_size'], batch_size=self.config['batch_size'], train=True)
        print("Train DataLoader Done")
        return train_dataset.DataLoader

    def val_dataloader(self):
        val_dataset = Dataset(x_file=self.config['val_input_dest_path'], y_file=self.config['val_output_dest_path'],
                              dataset_size=self.config['val_dataset_size'], batch_size=self.config['batch_size'], train=False)
        print("Validation DataLoader Done")
        return val_dataset.DataLoader

    def test_dataloader(self):
        test_dataset = Dataset(x_file=self.config['test_input_dest_path'], y_file=self.config['test_output_dest_path'],
                               dataset_size=self.config['test_dataset_size'], batch_size=self.config['batch_size'], train=False)
        print("Test DataLoader Done")
        return test_dataset.DataLoader

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config["learning_rate"])
#         self.lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": StepLR(optimizer, step_size=1, gamma=0.9),
        },
    }


#     def optimizer_step(self, *args, **kwargs):
#         super().optimizer_step(*args, **kwargs)
#         self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat=torch.reshape(y_hat, (-1, 30, 30))
        loss = self.criterion(y_hat, y)
        acc = custom_accuracy(y_hat, y)
        # self.log("train_loss", loss, on_step=False, on_epoch=True)
        # self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat=torch.reshape(y_hat, (-1, 30, 30))
        loss = self.criterion(y_hat, y)
        acc = custom_accuracy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat=torch.reshape(y_hat, (-1, 30, 30))
        loss = self.criterion(y_hat, y)
        acc = custom_accuracy(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        return loss



if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams.update({'font.size': 8})
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    random_seed(config["RANDOM_SEED"], True)
    wandb_logger = WandbLogger(
        project="Diffrentiable Spatial Planning Transformer", entity="agv_astar_dspt")
    model = WrappedModel(config)
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        gpus=1,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        callbacks=[lr_monitor]
    )
    trainer.fit(model)
    trainer.test(model)
    wandb.finish()
    # ignore this 
