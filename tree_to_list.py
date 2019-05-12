import pickle

def inorder_traversal(tree, node_list):
    if tree is None:
        return node_list
    if len(tree.children) > 0:
        node_list = inorder_traversal(tree.children[0], node_list)
    node_list.append(tree)
    if len(tree.children) > 1:
        node_list = inorder_traversal(tree.children[1], node_list)
    return node_list

def convert_to_list(tree):
    treex = []
    node_list = []
    node_list = inorder_traversal(tree, node_list)
    for i, node in enumerate(node_list):
        tree_dict = {}
        tree_dict['word'] = node.word
        tree_dict['function'] = node.function
        tree_dict['children'] = [node_list.index(node.children[i]) for i in range(len(node.children))]
        if not tree_dict['children']: tree_dict['children'] = -1
        tree_dict['bbox'] = -1 if not hasattr(node, 'bbox') else node.bbox
        print(tree_dict)
        treex.append(tree_dict)
    return treex


tree_path = 'data/CLEVR/CLEVR_32_MULTI_LARGE/trees/train/CLEVR_new_000014.tree'
tree = pickle.load(open(tree_path,'rb'))
treex = convert_to_list(tree)
