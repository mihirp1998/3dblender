# // Modified render 5 object
{
	"scene": [{"shape": "sphere", "size": "large", "color": "yellow", "texture": "metal", "location": [38, 32, 34,  8,  6,  8]},
		{"shape": "sphere", "size": "large", "color": "blue", "texture": "metal", "location": [22, 32, 28,  8,  6,  8]},
		{"shape": "sphere", "size": "large", "color": "green", "texture": "metal", "location": [32, 32, 26,  8,  6,  8]},
		{"shape": "sphere", "size": "large", "color": "yellow", "texture": "metal", "location": [12, 32, 20,  8,  6,  8]},
		{"shape": "cylinder", "size": "large", "color": "red", "texture": "metal", "location": [20, 32, 18,  8,  6,  8]}]
}

# // Modified render 4 object
{
	"scene": [{"shape": "sphere", "size": "large", "color": "blue", "texture": "metal", "location": [30, 32, 26,  8,  6,  8]},
		{"shape": "sphere", "size": "large", "color": "brown", "texture": "metal", "location": [38, 32, 19,  8,  6,  8]},
		{"shape": "sphere", "size": "large", "color": "yellow", "texture": "metal", "location": [19, 32, 20,  8,  6,  8]},
		{"shape": "cylinder", "size": "large", "color": "red", "texture": "metal", "location": [26, 32, 14,  8,  6,  8]}]
}





x = []
voxel_inp = np.zeros((voxel_size, voxel_size, voxel_size, class_count))
for obj in scene_obj_list:
    obj_id, color = obj['obj_id'], obj['color']
    obj_id = int(obj_id.split('blockid_')[-1])
    attribute_vec = np.zeros(class_count)
    attribute_vec[class_mapping[color]] = 1
    voxel_inp[np.where(blocks == obj_id)] = attribute_vec
    x.append((obj_id, color, class_mapping[color]))


t = 356
a=np.where(blocks == 3)[0][t]
b=np.where(blocks == 3)[1][t]
c=np.where(blocks == 3)[2][t]

blocks[a,b,c]
voxel_inp[a,b,c].argmax()
x




import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._thnn import type2backend
from .thnn.auto import function_by_name
import torch.backends.cudnn as cudnn

for param_group in optimizer.param_groups:
print(param_group[‘lr’])


scp -r configs/ lib/ mains/ models/ trainers/ sajaved@matrix.ml.cmu.edu:/home/sajaved/projects/text2scene/3dProbNeuralProgNet/

print('-'*30)
print(CDHW)
print('-'*30)

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)

loss = nn.BCEWithLogitsLoss()
loss.size_average = False
input = torch.randn(32, 32, 32, requires_grad=True)
target = torch.empty(32, 32, 32, dtype=torch.float).random_(2)
output = loss(input, target)


srun -p KATE_RESERVED -w compute-0-26 --time=30:00:00 --gres gpu:1 -c1 --mem=8g --pty $SHELL
srun -w compute-0-28 --time=24:00:00 --gres gpu:1 -c1 --mem=8g --pty $SHELL
srun -x compute-0-[36,32] --time=24:00:00 --gres gpu:1 -c1 --mem=8g --pty $SHELL

IMG="/projects/data/singularity/ubuntu-16.04-lts-tensorflow-gpu-1.7.0-python-keras-chainer-opencv-ffmpeg.img"
module load singularity
singularity shell -B /projects:/projects --nv $IMG

blender --background --python render_images.py -- --save_blendfiles 1 --num_images 2000 --max_objects 3 --min_pixels_per_object 20 --min_obj_2d_size 4

scp -r configs/ lib/ mains/ models/ trainers/ sajaved@matrix.ml.cmu.edu:/home/sajaved/projects/text2scene/3dProbNeuralProgNet/

scp -r sajaved@matrix.ml.cmu.edu:/home/sajaved/projects/text2scene/3dProbNeuralProgNet/results/CLEVR_32_MULTI_LARGE_2/PNP-Net-dual_recon-reduced-16k-mae ./results/CLEVR_32_MULTI_LARGE_2/

scp -r sajaved@matrix.ml.cmu.edu:/home/sajaved/projects/text2scene/3dProbNeuralProgNet/results/CLEVR_32_MULTI_LARGE_2/ ./results/

export CUDA_VISIBLE_DEVICES=0,3
CUDA_VISIBLE_DEVICES=0



# Blender
scp -r data/CLEVR/clevr-dataset-gen/image_generation/ sajaved@matrix.ml.cmu.edu:/home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/
cd /home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/image_generation

srun -w compute-0-28 --time=24:00:00 --gres gpu:1 -c1 --mem=8g --pty $SHELL
IMG="/home/sajaved/tensorflow-ubuntu-16.04.3-nvidia-375.26_wzy.img"
module load singularity
singularity shell --writable -B /projects:/projects --nv $IMG

/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 5000 --max_objects 3 --use_gpu 1 --start_idx 0
/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 5000 --max_objects 3 --use_gpu 1 --save_depth_maps 1  --start_idx 0


import os
import pickle
import json
path = '/Users/ashar/Downloads/test/'
folders = os.listdir(path)
for i in folders:
    if i[0] != '.':
        full_pickle_path = os.path.join(path, i, 'blender_info.p')
        full_json_path = os.path.join(path, i, 'blender_info.json')
        with open(full_pickle_path, 'rb') as f:
            given_objects = pickle.load(f)
        for obj in given_objects:
            obj['location'] = obj['location'].tolist()
        with open(full_json_path, 'w') as f:
            json.dump(given_objects, f)







/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 200 --max_objects 2 --save_depth_maps 0  --start_idx 0 --all_views 1 --include_inside_config 0


cd 
IMG="tensorflow-ubuntu-16.04.3-nvidia-375.26_wzy.img"
module load singularity
singularity shell --writable -B /projects:/projects --nv $IMG

cd projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/image_generation/

/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 200 --max_objects 2 --save_depth_maps 0  --start_idx 0 --all_views 1 --use_gpu 1 --back_front_only_flag 1

/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 200 --max_objects 3 --save_depth_maps 0  --start_idx 0 --all_views 1 --use_gpu 1 --back_front_only_flag 1



/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 600 --max_objects 1 --save_depth_maps 0  --start_idx 0 --all_views 1 --use_gpu 1 --include_inside_config 1 --percent_inside_samples 0.5

/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 800 --max_objects 3 --use_gpu 1 --start_idx 0

/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 800 --max_objects 3 --use_gpu 1 --start_idx 0 --allow_floating_objects 1

/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 800 --max_objects 4 --use_gpu 1 --start_idx 0 --radius 15



/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 800 --min_objects 2 --max_objects 3 --use_gpu 1 --start_idx 0 --save_depth_maps 1 --allow_floating_objects 1 --dataset_name CLEVR_64_36_WITH_FLOATING_WITH_DEPTH_NEW

/home/sajaved/blender/blender --background --python render_images.py -- --save_blendfiles 1 --num_images 800 --min_objects 2 --max_objects 3 --use_gpu 1 --start_idx 0 --save_depth_maps 1 --allow_floating_objects 0 --dataset_name CLEVR_64_36_WITH_DEPTH_NEW

blender --background --python render_images.py -- --render_from_given_objects 1 --dataset_name CLEVR_RENDERED_FROM_GIVEN --all_views 0
