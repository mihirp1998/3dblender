
import _init_paths
import pickle
from lib.tree import Tree
import os.path as osp
import os
import numpy as np
import json

def _combine_bbox(bbox1, bbox2):
  left = min(bbox1[0], bbox2[0])
  top = min(bbox1[1], bbox2[1])
  right = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
  bottom = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
  return [left, top, right - left, bottom - top]

def update_inside_tree_2d_bbox(tree, scenei, view_key, offset):
  if tree.word == 'inside':
    obj = scenei['objects']
    inside_obj_index = [i for i,j in enumerate(obj) if j['shape'] != 'cup'][0]
    top_bbox = obj[inside_obj_index]['bbox_2d'][view_key]['pixel_coords_lefttop']
    bottom_bbox = obj[inside_obj_index]['bbox_2d'][view_key]['pixel_coords_rightbottom']
    inside_obj_bbox = [top_bbox[0], top_bbox[1], bottom_bbox[0] - top_bbox[0], bottom_bbox[1] - top_bbox[1]]
    cup_bbox = [top_bbox[0]-offset,
                top_bbox[1]-offset,
                bottom_bbox[0] - top_bbox[0] + (2*offset),
                bottom_bbox[1] - top_bbox[1] + (2*offset)
                ] 
    inside_obj_bbox = np.asarray(inside_obj_bbox)
    cup_bbox = np.asarray(cup_bbox)
    
    tree.bbox = np.array(_combine_bbox(inside_obj_bbox, cup_bbox))
    inside_obj_index = 0 if tree.children[0].word != 'cup' else 1
    tree.children[inside_obj_index].bbox = inside_obj_bbox
    tree.children[int(not inside_obj_index)].bbox = cup_bbox
  return tree



res = 64
# Two values below should be changed together
view_key = '240_40'
approx_offset = 8

path = 'CLEVR_64_36_WITH_INSIDE/trees_3d_bboxes'
outpath = 'CLEVR_64_36_WITH_INSIDE/trees'

split = ['train']
os.rename(outpath, path)


for s in split:
  treepath = osp.join(path, s)
  scenepath = treepath.replace('trees_3d_bboxes','scenes')
  files = os.listdir(treepath)
  files_scenes = os.listdir(scenepath)
  try:
    os.makedirs(osp.join(outpath, s))
  except:
    pass

  for fi in files:
    fis = fi.replace('tree','json')
    if fi.endswith('tree') and fis.endswith('json'):
      with open(osp.join(path, s, fi), 'rb') as f:
        treei = pickle.load(f)

      with open(osp.join(scenepath, fis), 'r') as f:
        scenei = json.load(f)

      treei = update_inside_tree_2d_bbox(treei, scenei, view_key, approx_offset)
      pickle.dump(treei, open(osp.join(outpath, s, fi), 'wb'))
