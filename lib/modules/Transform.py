# Added AffineGridGenerator()
# Made changes to forward
# Ready for 3d

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd.function import once_differentiable
import numpy as np

class AffineGridGenerator(Function):

    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        N, C, D, H, W = size
        ctx.size = size
        if theta.is_cuda:
            AffineGridGenerator._enforce_cudnn(theta)
            # assert False
        ctx.is_cuda = False
        base_grid = theta.new(N, D, H, W, 4)

        linear_points_d = torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])
        linear_points_h = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
        linear_points_w = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])

        outer = torch.ger(linear_points_d, torch.ones(H))
        outer = torch.ger(outer.view(-1), torch.ones(W))
        base_grid[:, :, :, :, 2] = outer.view(D,H,W).expand_as(base_grid[:, :, :, :, 2])

        outer = torch.ger(torch.ones(D), linear_points_h)
        outer = torch.ger(outer.view(-1), torch.ones(W))
        base_grid[:, :, :, :, 1] = outer.view(D,H,W).expand_as(base_grid[:, :, :, :, 1])

        outer = torch.ger(torch.ones(D), torch.ones(H))
        outer = torch.ger(outer.view(-1), linear_points_w)
        base_grid[:, :, :, :, 0] = outer.view(D,H,W).expand_as(base_grid[:, :, :, :, 0])

        base_grid[:, :, :, :, 3] = 1

        ctx.base_grid = base_grid
        grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
        grid = grid.view(N, D, H, W, 3)
        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        N, C, D, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, D, H, W, 3])
        assert ctx.is_cuda == grad_grid.is_cuda
        if grad_grid.is_cuda:
            AffineGridGenerator._enforce_cudnn(grad_grid)
            # assert False
        base_grid = ctx.base_grid
        grad_theta = torch.bmm(
            base_grid.view(N, D * H * W, 4).transpose(1, 2),
            grad_grid.view(N, D * H * W, 3))
        grad_theta = grad_theta.transpose(1, 2)
        return grad_theta, None


class Transform(nn.Module):
    def __init__(self, matrix='default'):
        super(Transform, self).__init__()

    def forward(self, x, hw, variance=False):
        if variance:
            x = torch.exp(x / 2.0)
        size = torch.Size([x.size(0), x.size(1), int(hw[0]), int(hw[1]), int(hw[2])])

        # grid generation
        theta = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=np.float32)

        theta = Variable(torch.from_numpy(theta), requires_grad=False).cuda()

        theta = theta.expand(x.size(0), theta.size(1), theta.size(2))
        gridout = AffineGridGenerator.apply(theta, size)
        # gridout = F.affine_grid(theta, size)

        # bilinear sampling
        out = F.grid_sample(x, gridout, mode='bilinear')

        if variance:
            out = torch.log(out) * 2.0
        return out


