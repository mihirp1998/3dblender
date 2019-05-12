#!/usr/bin/env python

import _init_paths
import os
import os.path as osp
import pickle
import json

import numpy as np
import PIL.Image
import torch
import random
from nbtschematic import SchematicFile

import IPython;
ip = IPython.embed

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

# Ready for 3d
class CLEVRTREE():
    # set the thresholds for the size of objects here
    SMALL_THRESHOLD = 12
    # remove the medium size for simplicity and clarity

    NEW_SIZE_WORDS = ['small', 'large']
    OLD_SIZE_WORDS = ['small', 'large']

    def __init__(self, class_mapping, loss_type, batch_size=16,
                 base_dir='/cs/vml4/zhiweid/ECCV18/GenerativeNeuralModuleNetwork/data/CLEVR/CLEVR_128',
                 class_count=9,
                 random_seed=12138,
                 phase='train', shuffle=True, file_format='schematic'):
        self.phase = phase
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.fileformat = file_format
        self.class_count = class_count
        self.class_mapping = class_mapping
        self.loss_type = loss_type

        if phase not in ['train', 'test']:
            raise ValueError('invalid phase name {}, should be train or test'.format(phase))

        if file_format == 'schematic':
            self.image_dir = osp.join(base_dir, 'voxels', phase)
        elif file_format == 'npy':
            self.image_dir = osp.join(base_dir, 'features', phase)
        else:
            return ValueError('invalid input mode {}, should be voxel or feature'.format(file_format))
        self.tree_dir = osp.join(base_dir, 'trees', phase)
        self.scene_dir = osp.join(base_dir, 'scenes', phase)

        # get the file names for images, and corresponding trees, and the dictionary of the dataset
        self.image_files, self.tree_files, self.scene_files = self.prepare_file_list(phase)
        self.dictionary_path = osp.join(self.base_dir, 'dictionary_tree.pickle')
        self.dictionary = self.load_dictionary(self.tree_files, self.dictionary_path)

        # update the size words, since we need to adjust the size for all objects
        # according to the actual 2-D bounding box
        for word in self.OLD_SIZE_WORDS:
            self.dictionary.remove(word)
        for word in self.NEW_SIZE_WORDS:
            self.dictionary.append(word)

        # iterator part
        self.random_generator = random.Random()
        self.random_generator.seed(random_seed)

        self.files = dict()

        self.files[phase] = {
            'voxel': self.image_files,
            'tree': self.tree_files,
            'scene': self.scene_files,
        }        

        self.index_ptr = 0
        self.index_list = list(range(len(self.image_files)))

        self.shuffle = shuffle
        if shuffle:
            self.random_generator.shuffle(self.index_list)

        self.im_size = self.read_first()
        self.ttdim = len(self.dictionary) + 1

    def __len__(self):
        return len(self.files[self.phase]['voxel'])

    def read_first(self):
        images, _, _, _ = self.next_batch()
        self.index_ptr = 0

        return images.size()

    def prepare_file_list(self, phase):
        """
        Get the filename list for images and correspondign trees
        :return:
        """
        image_list = []
        tree_list = []
        scene_list = []
        for image_filename in sorted(os.listdir(self.image_dir)):
            if image_filename.endswith('.' + self.fileformat):
                image_path = os.path.join(self.image_dir, image_filename)
                image_list.append(image_path)
                filename, _ = os.path.splitext(image_filename)
                tree_filename = filename + '.tree'
                tree_path = os.path.join(self.tree_dir, tree_filename)
                tree_list.append(tree_path)
                scene_filename = filename + '.json'
                scene_path = os.path.join(self.scene_dir, scene_filename)
                scene_list.append(scene_path)

        c = list(zip(image_list, tree_list, scene_list))
        random.shuffle(c)
        image_list, tree_list, scene_list = zip(*c)
        if phase == 'train':
            # k = 30
            # return image_list[:k] , tree_list[:k], scene_list[:k]
            return image_list, tree_list, scene_list
        else:
            k = 200
            return image_list[:k] , tree_list[:k], scene_list[:k]

    def load_dictionary(self, tree_files, dictionary_path):
        if osp.isfile(dictionary_path):  # the dictionary has been created, then just load return
            with open(dictionary_path, 'rb') as f:
                dictionary = pickle.load(f)
        else:
            dictionary_set = set()
            for idx, tree_file_path in enumerate(tree_files):
                with open(tree_file_path, 'rb') as f:
                    tree = pickle.load(f)
                tree_words = self.get_tree_words(tree)
                dictionary_set.update(set(tree_words))

            dictionary = list(dictionary_set)

            # update the size words, since we need to adjust the size for all objects
            # according to the actual 2-D bounding box
            for word in self.OLD_SIZE_WORDS:
                dictionary.remove(word)
            for word in self.NEW_SIZE_WORDS:
                dictionary.append(word)
            with open(dictionary_path, 'wb') as f:
                pickle.dump(dictionary, f)

        return dictionary

    def get_tree_words(self, tree):
        words = [tree.word]
        for child in tree.children:
            words += self.get_tree_words(child)
        return words

    def next_batch(self):
        data_file = self.files[self.phase]
        class_count = self.class_count
        class_mapping = self.class_mapping

        images, trees, categories = [], [], []
        for i in range(0, min(self.batch_size, len(self) - self.index_ptr)):
            index = self.index_list[self.index_ptr]

            # load scene json
            scene_file = data_file['scene'][index]
            with open(scene_file) as f:
                scene_json = json.load(f)
            scene_obj_list = scene_json['objects']

            # check if extra object voxels need to be removed from blocks
            remove_extra_objects = False
            if len(scene_obj_list) == 1:
                remove_extra_objects = True
                orig_block_id = int(scene_obj_list[0]['obj_id'].split('blockid_')[-1])

            # load image
            if self.fileformat == 'schematic':
                voxel_file = data_file['voxel'][index]
                sf = SchematicFile.load(voxel_file)
                blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
                voxel_size = int(round(len(blocks)**(1./3)))
                blocks = blocks.reshape((voxel_size,voxel_size,voxel_size))
                blocks = np.moveaxis(blocks, [0,1,2], [1,0,2])
                blocks = blocks.copy()
                if remove_extra_objects:
                    blocks[blocks != orig_block_id] = 0

                if len(np.unique(blocks)) - 1 != len(scene_obj_list):
                    raise ValueError('voxel object count doesn\'t agree with object count')

                if self.loss_type == 'l1' or self.loss_type == 'l2' or self.loss_type == 'binary_entropy':
                    blocks[np.nonzero(blocks)] = 1
                    blocks = np.expand_dims(blocks, -1)
                    images.append(blocks)
                elif self.loss_type == 'cross_entropy':
                    voxel_inp = np.zeros((voxel_size, voxel_size, voxel_size, class_count))
                    for obj in scene_obj_list:
                        obj_id, color = obj['obj_id'], obj['color']
                        obj_id = int(obj_id.split('blockid_')[-1])
                        attribute_vec = np.zeros(class_count)
                        attribute_vec[class_mapping[color]] = 1
                        voxel_inp[np.where(blocks == obj_id)] = attribute_vec
                    attribute_vec = np.zeros(class_count)
                    attribute_vec[0] = 1
                    voxel_inp[np.where(blocks == 0)] = attribute_vec
                    images.append(voxel_inp)
                # elif self.loss_type == 'binary_entropy':
                #     voxel_inp = np.zeros((voxel_size, voxel_size, voxel_size, 2))
                #     voxel_inp[np.where(blocks == 0)] = np.array([1,0])
                #     voxel_inp[np.where(blocks == 1)] = np.array([0,1])
                #     images.append(voxel_inp)
                else:
                    raise ValueError('Wrong loss type specified')
            elif self.fileformat == 'npy':
                feature_file = data_file['voxel'][index]
                input_feature = np.load(feature_file)
                if self.loss_type == 'l1' or self.loss_type == 'l2':
                    images.append(input_feature)
                else:
                    raise ValueError('incorrect loss type for this input_mode')
            else:
                raise ValueError('wrong file format for voxels')

            # load tree
            with open(data_file['tree'][index], 'rb') as f:
                tree = pickle.load(f)
            # ip()
            tree = self.adapt_tree(tree)
            # ip()
            # print('-------------')
            trees.append(tree)
            categories.append(self.get_categorical_list(tree))

            self.index_ptr += 1

        images = np.array(images, dtype=np.float32).transpose(0, 4, 1, 2, 3)

        refetch = False
        if self.index_ptr >= len(self):
            self.index_ptr = 0
            refetch = True
            if self.shuffle:
                self.random_generator.shuffle(self.index_list)

        return torch.from_numpy(images), trees, categories, refetch

    def get_all(self):
        data_file = self.files[self.phase]
        class_count = self.class_count
        class_mapping = self.class_mapping

        images, trees, categories = [], [], []
        for i in range(len(self.index_list)):
            index = self.index_list[self.index_ptr]

            # load scene json
            scene_file = data_file['scene'][index]
            with open(scene_file) as f:
                scene_json = json.load(f)
            scene_obj_list = scene_json['objects']

            # load image
            if self.fileformat == 'schematic':
                voxel_file = data_file['voxel'][index]
                # img = PIL.Image.open(img_file)
                sf = SchematicFile.load(voxel_file)
                blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
                voxel_size = int(round(len(blocks)**(1./3)))
                blocks = blocks.reshape((voxel_size,voxel_size,voxel_size))
                if self.loss_type == 'l1' or self.loss_type == 'l2' or 'binary_entropy':
                    blocks = blocks.copy()
                    blocks[np.nonzero(blocks)] = 1
                    blocks = np.expand_dims(blocks, -1)
                    images.append(blocks)
                elif self.loss_type == 'cross_entropy':
                    voxel_inp = np.zeros((voxel_size, voxel_size, voxel_size, class_count))
                    for obj in scene_obj_list:
                        obj_id, color = obj['obj_id'], obj['color']
                        obj_id = int(obj_id.split('blockid_')[-1])
                        attribute_vec = np.zeros(class_count)
                        attribute_vec[class_mapping[color]] = 1
                        voxel_inp[np.where(blocks == obj_id)] = attribute_vec
                    attribute_vec = np.zeros(class_count)
                    attribute_vec[0] = 1
                    voxel_inp[np.where(blocks == 0)] = attribute_vec
                    images.append(voxel_inp)
                # elif self.loss_type == 'binary_entropy':
                #     voxel_inp = np.zeros((voxel_size, voxel_size, voxel_size, class_count))
                #     voxel_inp[np.where(blocks == 0)] = np.array([1,0])
                #     voxel_inp[np.where(blocks == 1)] = np.array([0,1])
                #     images.append(voxel_inp)
                else:
                    raise ValueError('Wrong loss type specified')
            elif self.fileformat == 'npy':
                feature_file = data_file['feature'][index]
                input_feature = np.load(feature_file)
                if self.loss_type == 'l1' or self.loss_type == 'l2':
                    images.append(input_feature)
                else:
                    raise ValueError('incorrect loss type for this input_mode')
            else:
                raise ValueError('wrong file format for voxels')

            # load tree
            with open(data_file['tree'][index], 'rb') as f:
                tree = pickle.load(f)
            tree = self.adapt_tree(tree)
            trees.append(tree)

            self.index_ptr += 1

        images = np.array(images, dtype=np.float32).transpose(0, 4, 1, 2, 3)

        return torch.from_numpy(images), trees

    def adapt_tree(self, tree):
        tree = self._adapt_tree(tree, parent_bbox=None)
        return tree

    def _adapt_tree(self, tree, parent_bbox):
        # If the input mode is feature based, the channels in the features are half the resolution of the scene voxel
        if self.fileformat == 'npy' and hasattr(tree,'bbox'):
            tree.bbox = tree.bbox // 2

        # adjust tree.word for object size according to the bounding box, since the original size is for 3-D world
        if tree.function == 'combine' and tree.word in self.OLD_SIZE_WORDS:
            depth = parent_bbox[3]
            width = parent_bbox[4]
            height = parent_bbox[5]
            tree.word = self._get_size_word(depth, width, height)

        # set the bbox for passing to children
        if tree.function == 'combine':
            bbox_xywh = parent_bbox
        elif tree.function == 'describe':
            bbox_xywh = tree.bbox
        else:
            bbox_xywh = None

        # then swap the bbox to (y,x,h,w)
        # if hasattr(tree, 'bbox'):
        #     bbox_yxhw = (tree.bbox[1], tree.bbox[0], tree.bbox[3], tree.bbox[2])
        #     tree.bbox = np.array(bbox_yxhw)

        # pre-order traversal
        for child in tree.children:
            self._adapt_tree(child, parent_bbox=bbox_xywh)
        return tree

    def get_categorical_list(self, tree):
        categorical_list, attr_list = self._get_categorical_list(tree)
        return categorical_list

    def _get_categorical_list(self, tree):
        # must be post-ordering traversal, parent need info from children
        category_list = list()
        attr_list = list()
        for child in tree.children:
            children_category_list, children_attr_list = self._get_categorical_list(child)
            category_list += children_category_list
            attr_list += children_attr_list

        if tree.function == 'describe':
            bbox = tree.bbox
            # adapted_bbox = (bbox[1], bbox[0], bbox[3], bbox[2])
            adapted_bbox = (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5])
            attr_list.append(tree.word)
            attr_vec = self._get_attr_vec(attr_list)
            obj_category = (adapted_bbox, attr_vec)
            category_list.append(obj_category)

        if tree.function == 'combine':
            attr_list.append(tree.word)  # just pass its word to parent

        return category_list, attr_list

    def _get_attr_vec(self, attr_list):
        vec = np.zeros(len(self.dictionary), dtype=np.float64)
        for attr in attr_list:
            attr_idx = self.dictionary.index(attr)
            vec[attr_idx] = 1.0
        return vec

    @classmethod
    def _get_size_word(cls, depth, width, height):
        maximum = max(depth, width, height)
        if maximum < cls.SMALL_THRESHOLD:
            return 'small'
        else:
            return 'large'


if __name__ == '__main__':
    loader = CLEVRTREE(phase='test',
                       base_dir='/zhiweid/work/gnmn/GenerativeNeuralModuleNetwork/data/CLEVR/CLEVR_64_MULTI_LARGE')
    for i in range(3):
        im, trees, categories, ref = loader.next_batch()
        import IPython;

        IPython.embed()
        print('In CLEVR Tree main func')
        break
        print(im[0].shape)
        print(categories[0])
    print(loader.dictionary)

'''
# For testing the format of PIL.Image and cv2
img2 = PIL.Image.open('/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/CLEVR/clevr-dataset-gen/output/images/train/CLEVR_new_000003.png')
img2 = np.array(img2)
img2 = img2[:,:,:-1]

out = PIL.Image.fromarray(img2)
out.save('test.png')
# img = cv2.imread('/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/CLEVR/clevr-dataset-gen/output/images/train/CLEVR_new_000003.png')
# cv2.imwrite('test.png',img)
print(img2)
#
img = PIL.Image.open('test.png')
img = np.array(img)

print(img2 - img)

'''
