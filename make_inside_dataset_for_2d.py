
import _init_paths
import pickle
from lib.tree import Tree
import os.path as osp
import os
import ipdb
import shutil, errno
st = ipdb.set_trace

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

old_path_tree = 'CLEVR_64_36_WITH_INSIDE/trees'
new_path_tree = 'CLEVR_64_36_WITH_INSIDE_ONLY/trees'
old_path_scene = 'CLEVR_64_36_WITH_INSIDE/scenes'
new_path_scene = 'CLEVR_64_36_WITH_INSIDE_ONLY/scenes'
old_path_image = 'CLEVR_64_36_WITH_INSIDE/images'
new_path_image = 'CLEVR_64_36_WITH_INSIDE_ONLY/images'

split = ['train']

for s in split:
  treepath_old = osp.join(old_path_tree, s)
  treepath_new = osp.join(new_path_tree, s)
  scenepath_old = osp.join(old_path_scene, s)
  scenepath_new = osp.join(new_path_scene, s)
  imagepath_old = osp.join(old_path_image, s)
  imagepath_new = osp.join(new_path_image, s)
  if not os.path.exists(treepath_new):
      os.makedirs(treepath_new)
  if not os.path.exists(scenepath_new):
      os.makedirs(scenepath_new)
  if not os.path.exists(imagepath_new):
      os.makedirs(imagepath_new)

  file_tree = os.listdir(treepath_old)

  for fi_tree in file_tree:
    with open(osp.join(treepath_old, fi_tree), 'rb') as f:
      treei = pickle.load(f)
    if treei.word == 'inside':
      st()
      fi_scene = fi_tree.replace('tree','json')
      fi_image = fi_tree.replace('.tree','')
      copyanything(osp.join(treepath_old, fi_tree), osp.join(treepath_new, fi_tree))
      copyanything(osp.join(scenepath_old, fi_scene), osp.join(scenepath_new, fi_scene))
      copyanything(osp.join(imagepath_old, fi_image), osp.join(imagepath_new, fi_image))



