import scipy.misc
import numpy as np

def visualize_tree(tree, text):
    text = _visualize_tree(tree, 0, text)
    return text

def _visualize_tree(tree, level, text):
    if tree == None:
        return text
    for i in range(tree.num_children - 1, (tree.num_children - 1) // 2, -1):
        text = _visualize_tree(tree.children[i], level + 1, text)
    text += ' ' * level + tree.word + '\n'
    if hasattr(tree, 'bbox'):
        text += 'Bouding box of {} is {}'.format(tree.word, tree.bbox) + '\n'
    for i in range((tree.num_children - 1) // 2, -1, -1):
        text = _visualize_tree(tree.children[i], level + 1, text)
    return text

def get_tree_text(tree):
    text = ''
    text = visualize_tree(tree, text)
    return text

def get_box(tree, level, box):
    if tree == None:
        return box
    for i in range(tree.num_children - 1, (tree.num_children - 1) // 2, -1):
        box = get_box(tree.children[i], level + 1, box)
    if hasattr(tree, 'offsets'):
        box.append(tree.offsets)
    for i in range(tree.num_children - 1, (tree.num_children - 1) // 2, -1):
        box = get_box(tree.children[i], level + 1, box)
    return box

def get_tree_bboxes(tree):
    box = []
    box = get_box(tree, 0, box)
    return box


def color_grid_vis(X, nh, nw, save_path):
    d, h, w = X[0].shape[:3]
    img = np.zeros((h * nh, w * nw, 3))
    for n, x in enumerate(X):
        j = int(n / nw)
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    scipy.misc.imsave(save_path, img)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.pixel_count = 0
        self.batch_count = 0

    def update(self, val, n=1, batch=1):
        self.val = val
        self.sum += val * n
        self.pixel_count += n
        self.batch_count += batch
        self.pixel_avg = self.sum / self.pixel_count
        self.batch_avg = self.sum / self.batch_count
