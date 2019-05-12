from __future__ import print_function
import _init_paths
import datetime
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import pytz
import scipy.misc
import os.path as osp
import os
import numpy as np
import random
import pdb

from lib.data_loader.color_mnist_tree_multi import COLORMNISTTREE
from lib.data_loader.clevr.clevr_tree import CLEVRTREE
from lib.config import load_config, Struct
from models.PNPNet.pnp_net import PNPNet
from trainers.pnpnet_trainer import PNPNetTrainer
from lib.weight_init import weights_init


parser = argparse.ArgumentParser(description='PNPNet - main model experiment')
parser.add_argument('--config_path', type=str, default='./configs/pnp_net_configs.yaml', metavar='C',
                    help='path to the configuration file')


args = parser.parse_args()

config_dic = load_config(args.config_path)
configs = Struct(**config_dic)

# assert (torch.cuda.is_available())  # assume CUDA is always available

print('configurations:', configs)

# torch.cuda.set_device(configs.gpu_id)
# torch.manual_seed(configs.seed)
# torch.cuda.manual_seed(configs.seed)
np.random.seed(configs.seed)
random.seed(configs.seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# configs.exp_dir = 'results/' + configs.data_folder + '/' + configs.exp_dir_name
# exp_dir = configs.exp_dir

# try:
#     os.makedirs(configs.exp_dir)
# except:
#     pass
# try:
#     os.makedirs(osp.join(configs.exp_dir, 'samples'))
# except:
#     pass
# try:
#     os.makedirs(osp.join(configs.exp_dir, 'checkpoints'))
# except:
#     pass

# loaders
class_mapping = {"cyan":0, "brown":1, "gray":2, "purple":3, "green":4, "blue":5, "yellow":6, "red":7}
if 'CLEVR' in configs.data_folder:
    # we need the module's label->index dictionary from train loader
    train_loader = CLEVRTREE(class_mapping, configs.loss, phase='train', base_dir=osp.join(configs.base_dir, configs.data_folder),
                             class_count=configs.class_count, 
                             batch_size=configs.batch_size,
                             random_seed=configs.seed, shuffle=True)



def visualize_tree(im, trees, categories, ref):
    for i in range(len(trees)):
        print('************** tree **************')
        print(ref)
        print(trees[i])
        _visualize_tree(trees[i], 0)
        print('**********************************')


def _visualize_tree(tree, level):
    if tree == None:
        return
    for i in range(tree.num_children - 1, (tree.num_children - 1) // 2, -1):
        _visualize_tree(tree.children[i], level + 1)
    print(' ' * level + tree.word, tree.parent)
    if hasattr(tree, 'bbox'):
        print('Bouding box of {} is {}'.format(tree.word, tree.bbox))
    for i in range((tree.num_children - 1) // 2, -1, -1):
        _visualize_tree(tree.children[i], level + 1)
    return


for i in range(2):
    im, trees, categories, ref = train_loader.next_batch()
    visualize_tree(im, trees, categories, ref)




