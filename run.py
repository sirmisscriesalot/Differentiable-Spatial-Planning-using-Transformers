from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random
import math
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import autograd
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1lORbh70-sTXvY48ARifexxLsYepDcHSx',
                                    dest_path='/data/navdata_input_30.npy',
                                    unzip=False)
gdd.download_file_from_google_drive(file_id='10WwaIDmBo2cfdHJz4O0aABN0mIbF8ByH',
                                    dest_path='/data/navdata_output_30.npy',
                                    unzip=False)


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


random_seed(42, True)


class SyntheticNavigationDataset(Dataset):
    def __init__(self, x_file, y_file, n):
        self.x_list = []
        self.y_list = []
        with open(x_file, 'rb') as fx, open(y_file, 'rb') as fy:
            for i in range(n):
                self.x_list.append(np.load(fx))
                self.y_list.append(np.load(fy))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        x_tensor = torch.from_numpy(self.x_list[idx].astype(np.float32))
        y_tensor = torch.from_numpy(self.y_list[idx].astype(np.float32))
        sample = {'x': x_tensor, 'y': y_tensor}

        return sample


nav_dataset = SyntheticNavigationDataset(
    x_file='/data/navdata_input_30.npy', y_file='/data/navdata_output_30.npy', n=33000)
sample = nav_dataset[10]
print(sample['x'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=900, max_len=64):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, d_model, dtype=torch.float).unsqueeze(1)
        print(position.shape)
        div_term = torch.exp(torch.arange(
            0, max_len, 2).float() * (-math.log(900) / max_len))
        print(div_term.shape)
        
        pe[0::2,:] = torch.transpose(torch.sin(position * div_term),0,1)
        pe[1::2,:] = torch.transpose(torch.cos(position * div_term),0,1)

        
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
    
encod_block = PositionalEncoding(d_model=900, max_len=64)
pe = encod_block.pe.squeeze().T.cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
fig.colorbar(pos, ax=ax)
ax.set_xlabel("Position in sequence")
ax.set_ylabel("Hidden dimension")
ax.set_title("Positional encoding over hidden dimensions")
ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
plt.show()

class DSPT(nn.Module):

    def __init__(self):
        # add args dict
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
        )

        # # Encoder
        self.pe = PositionalEncoding(d_model=900, max_len=64)
        self.encoder = nn.ModuleList()
        self.num_trans_layers = 5
        for _ in range(self.num_trans_layers):
            self.encoder.append(nn.TransformerEncoderLayer(d_model=900, nhead=9, dim_feedforward=512, dropout=0,
                                layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None))

        # Decoder
        self.decoder = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.p1(x)
        x = self.pe(x)
        
        for i in range(self.num_trans_layers):
            x = self.encoder[i](x)
        x = self.decoder(x)
        return x


model = DSPT()
model = model.to(device)
train_dataloader = DataLoader(nav_dataset, batch_size=20, shuffle=True)


def train_loop(dataloader, model, epochs):
    model.train()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    # with autograd.detect_anomaly():
    for t in tqdm(range(epochs)):
        for batch, sample in enumerate(dataloader):
            pred = model(sample['x'].to(device))
            sample['y'] = sample['y'].to(device)
            loss = loss_fn(torch.reshape(pred, (-1, 30, 30)), sample['y'])
            if batch % 100 == 0:
                print(f"loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for sample in dataloader:
            pred = model(sample['x'].to(device))
            sample['y'] = sample['y'].to(device)
            test_loss += loss_fn(torch.reshape(pred,
                                 (-1, 30, 30)), sample['y']).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


EPOCHS = 40
train_loop(train_dataloader, model, epochs=EPOCHS)
print("Done!")
