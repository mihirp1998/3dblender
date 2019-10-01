import pickle
import numpy as np

def get_tree_words(tree):
    words = [tree.word]
    for child in tree.children:
        words += get_tree_words(child)
    return words
NEW_SIZE_WORDS = ['small', 'large']
OLD_SIZE_WORDS = ['small', 'large']
SMALL_THRESHOLD = 16
def _get_size_word(width, height):
    maximum = max(width, height)
    if maximum < SMALL_THRESHOLD:
        return 'small'
    else:
        return 'large'
def visualize_tree(tree):
    _visualize_tree(tree, 0)
def _visualize_tree(tree, level):
    if tree == None:
        return
    for i in range(tree.num_children - 1, (tree.num_children - 1) // 2, -1):
        _visualize_tree(tree.children[i], level + 1)
    print(' ' * level + tree.word)
    if hasattr(tree, 'bbox'):
        print('Bouding box of {} is {}'.format(tree.word, tree.bbox))
    for i in range((tree.num_children - 1) // 2, -1, -1):
        _visualize_tree(tree.children[i], level + 1)
    return
def _adapt_tree(tree, parent_bbox):
    if tree.function == 'combine' and tree.word in OLD_SIZE_WORDS:
        width = parent_bbox[2]
        height = parent_bbox[3]
        tree.word = _get_size_word(width, height)
    if tree.function == 'combine':
        bbox_xywh = parent_bbox
    elif tree.function == 'describe':
        bbox_xywh = tree.bbox
    else:
        bbox_xywh = None
    if hasattr(tree, 'bbox'):
        bbox_yxhw = (tree.bbox[1], tree.bbox[0], tree.bbox[3], tree.bbox[2])
        tree.bbox = np.array(bbox_yxhw)
    for child in tree.children:
        _adapt_tree(child, parent_bbox=bbox_xywh)
    return tree
def adapt_tree(tree):
    tree = _adapt_tree(tree, parent_bbox=None)
    return tree

# path = 'data/CLEVR/CLEVR_64_OBJ_FULL_TRY/trees/train/CLEVR_new_000000.tree'
path = 'data/CLEVR/clevr-dataset-gen/output/CLEVR_64_OBJ_FULL_TRY/trees/train/CLEVR_new_000103.tree'
path = 'CLEVR_64_NEW/trees/train/CLEVR_new_000001.tree'
paths = [path]
for p in paths:
    with open(p, 'rb') as f:
        tree = pickle.load(f)
    tree = adapt_tree(tree)
    visualize_tree(tree)
