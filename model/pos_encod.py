from numpy.lib.twodim_base import triu_indices
import torch
import torch.nn as nn
import math 
import matplotlib.pyplot as plt

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

PRINT_PIC=True
if PRINT_PIC:
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