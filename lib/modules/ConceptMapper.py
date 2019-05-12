# Made changes to both functions
# Ready for 3d

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ConceptMapper(nn.Module):
  def __init__(self, CDHW, vocab_size):
    super(ConceptMapper, self).__init__()
    C, D, H, W = CDHW[0], CDHW[1], CDHW[2], CDHW[3]
    self.mean_dictionary = nn.Linear(vocab_size, C*D*H*W, bias=False)
    self.std_dictionary  = nn.Linear(vocab_size, C*D*H*W, bias=False)
    self.C, self.D, self.H, self.W = C, D, H, W

  def forward(self, x):
    word_mean = self.mean_dictionary(x)
    word_std  = self.std_dictionary(x)
    if self.D == 1 and self.H == 1 and self.W == 1:
      return [word_mean.view(-1, self.C, 1, 1, 1), \
              word_std.view(-1, self.C, 1, 1, 1)]
    else:
      return [word_mean.view(-1, self.C, self.D, self.H, self.W), \
              word_std.view(-1, self.C, self.D, self.H, self.W)]

