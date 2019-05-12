import pickle
import numpy as np

path = '../data/CLEVR/CLEVR_64_MULTI_LARGE/trees/train/CLEVR_new_001912.tree'
with open(path, 'rb') as f:
    tree = pickle.load(f)

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

    # if isinstance(tree.function_obj, Describe):
    #     print(tree.function_obj.attributes, tree.function_obj)
    #     if tree.function != 'combine':
    #         print('position {}'.format(tree.function_obj.position))

    if hasattr(tree, 'bbox'):
        print('Bouding box of {} is {}'.format(tree.word, tree.bbox))

    for i in range((tree.num_children - 1) // 2, -1, -1):
        _visualize_tree(tree.children[i], level + 1)

    return

def _adapt_tree(tree, parent_bbox):
    # adjust tree.word for object size according to the bounding box, since the original size is for 3-D world
    if tree.function == 'combine' and tree.word in OLD_SIZE_WORDS:
        width = parent_bbox[2]
        height = parent_bbox[3]
        tree.word = _get_size_word(width, height)

    # set the bbox for passing to children
    if tree.function == 'combine':
        bbox_xywh = parent_bbox
    elif tree.function == 'describe':
        bbox_xywh = tree.bbox
    else:
        bbox_xywh = None

    # then swap the bbox to (y,x,h,w)
    if hasattr(tree, 'bbox'):
        bbox_yxhw = (tree.bbox[1], tree.bbox[0], tree.bbox[3], tree.bbox[2])
        tree.bbox = np.array(bbox_yxhw)

    # pre-order traversal
    for child in tree.children:
        _adapt_tree(child, parent_bbox=bbox_xywh)
    return tree

def adapt_tree(tree):
    tree = _adapt_tree(tree, parent_bbox=None)
    return tree

tree = adapt_tree(tree)
visualize_tree(tree)
# tree_words = get_tree_words(tree)