import torch 
import math
import random
import argparse
from torch.types import Device
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd,Tensor
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# wandb.init(project="test1")

parser = argparse.ArgumentParser()
parser.add_argument("--sizet",help="size of the train dataset",type=int)
parser.add_argument("--sizev",help="size of the validation dataset",type=int)
parser.add_argument("--xfilet",help="name of the x train dataset")
parser.add_argument("--yfilet",help="name of the y train dataset")
parser.add_argument("--xfilev",help="name of the x validation dataset")
parser.add_argument("--yfilev",help="name of the y validation dataset")
args = parser.parse_args()

sizet = args.sizet
sizev = args.sizev
xfilet = args.xfilet + ".npy"
yfilet = args.yfilet + ".npy"
xfilev = args.xfilev + ".npy"
yfilev = args.yfilev + ".npy"

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

  def __init__(self,x_file,y_file,n):
    self.x_list = []
    self.y_list = []
    with open(x_file,'rb') as fx, open(y_file,'rb') as fy:
      for i in range(n):
        self.x_list.append(np.load(fx))
        self.y_list.append(np.load(fy))
  
  def __len__(self):
    return len(self.x_list)

  def __getitem__(self,idx):
    x_tensor = torch.from_numpy(self.x_list[idx].astype(np.float32))
    y_tensor = torch.from_numpy(self.y_list[idx].astype(np.float32))
    sample = {'x':x_tensor,'y':y_tensor}

    return sample   

train_dataset = SyntheticNavigationDataset(x_file = xfilet, y_file = yfilet,n = sizet)
validation_dataset = SyntheticNavigationDataset(x_file = xfilev, y_file = yfilev,n = sizev)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class PositionalEncoding(nn.Module):

  def __init__(self):
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

class TransformerModel(nn.Module):

  def __init__(self,d_model=64,nhead=8,d_hid=512,nlayers=5,dropout=0.1):
    super().__init__()
    self.model_type = 'Transformer'
    self.pos_encoder = PositionalEncoding()
    self.conv_encoder = CNNEncoding()
    encoder_layers = TransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True)
    self.transformer_encoder = TransformerEncoder(encoder_layers,nlayers)
    self.d_model = d_model
    self.decoder = nn.Linear(d_model,1)

  def init_weights(self) -> None:
    nn.init.kaiming_uniform_(self.encoder.weight.data,nonlinearity='relu')
    self.decoder.bias.data.zero_()
    nn.init.kaiming_uniform_(self.decoder.weight.data,nonlinearity='relu')

  def forward(self, src) -> Tensor:
    src = self.conv_encoder(src)
    src = self.pos_encoder(src)
    src = torch.transpose(src,1,2)
    output = self.transformer_encoder(src)
    output = self.decoder(output)

    return output

model = TransformerModel().to(device)
criterion = nn.MSELoss()
lr = 1.0
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,gamma=0.9)

def show_output():
  plt.subplot(122)
  sample = train_dataset[10]
  sample['x'] = torch.reshape(sample['x'],(1,2,sample['x'].shape[1],sample['x'].shape[2]))
  output = model(sample['x'].to(device))
  output = torch.reshape(output,(int(math.sqrt(output.shape[1])),int(math.sqrt(output.shape[1]))))
  output = output.to('cpu').detach().numpy()

  plt.imshow(output, cmap='hot', interpolation='nearest')
  plt.show()

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.subplot(121)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

train_dl = DataLoader(train_dataset,batch_size=1,shuffle=True)
valid_dl = DataLoader(validation_dataset,batch_size=2)

def loss_func(inx,outy):
  return criterion(torch.reshape(inx,(-1,int(math.sqrt(inx.shape[1])),int(math.sqrt(inx.shape[1])))),outy)

def train(model: nn.Module) -> None:

  model.train()

  for batch,sample in enumerate(train_dl):
    pred = model(sample['x'].to(device))
    out = sample['y'].to(device)
    loss = loss_func(pred,out)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()

    if batch%100 == 0:
      loss = loss.item()
      print(f"loss: {loss:>7f}")

epochs = 40

for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(model)
  scheduler.step()

  model.eval()
  with torch.no_grad():
    validation_loss = sum(loss_func(model(val_sample['x'].to(device)),val_sample['y'].to(device)) for val_sample in valid_dl)
  print(t , validation_loss/len(valid_dl))

print("Done!")


