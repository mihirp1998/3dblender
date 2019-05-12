# Made changes to init
# Ready for 3d

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from lib.ResidualModule import ResidualModule


class Writer(nn.Module):
    def __init__(self, indim, hiddim, outdim, ds_times, normalize, nlayers=4, writer_sigmoid=0):
        super(Writer, self).__init__()

        self.ds_times = ds_times
        self.writer_sigmoid = writer_sigmoid

        if normalize == 'gate':
          ifgate = True
        else:
          ifgate = False

        self.decoder = ResidualModule(modeltype='decoder', indim=indim, hiddim=hiddim, outdim=hiddim,
                                      nres=self.ds_times, nlayers=nlayers, ifgate=ifgate, normalize=normalize)

        self.out_conv = nn.Conv3d(hiddim, outdim, 3, 1, 1, 1)
        self.out_sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.decoder(x)
        out = self.out_conv(out)
        if self.writer_sigmoid:
          out = self.out_sigmoid(out)

        return out
