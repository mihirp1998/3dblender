import pickle
import numpy as np
import binvox_rw_new
from nbtschematic import SchematicFile

def save_voxel(voxel_, filename, THRESHOLD=0.5):
    S1 = voxel_.shape[2]
    S2 = voxel_.shape[1]
    S3 = voxel_.shape[0]
    # st()
    binvox_obj = binvox_rw_new.Voxels(
        np.transpose(voxel_, [0, 2, 1]) >= THRESHOLD,
        dims = [S1, S2, S3],
        translate = [0.0, 0.0, 0.0],
        scale = 1.0,
        axis_order = 'xyz'
    )   

    with open(filename, "wb") as f:
        binvox_obj.write(f)

voxel_size = 64
obj_id_list = [1,1,1,5,3]
tree_path = 'data/CLEVR/clevr-dataset-gen/output/CLEVR_64_36_MAYHEM_AGAIN/trees/train/CLEVR_new_00000{}.tree'
voxel_path = 'data/CLEVR/clevr-dataset-gen/output/CLEVR_64_36_MAYHEM_AGAIN/voxels/train/CLEVR_new_00000{}.schematic'
for i in range(5):
    treex = pickle.load(open(tree_path.format(i),'rb'))
    sf = SchematicFile.load(voxel_path.format(i))
    blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
    blocks = blocks.reshape((voxel_size,voxel_size,voxel_size))
    object_idx = np.where(blocks == obj_id_list[i])
    x_top = object_idx[0].min()
    y_top = object_idx[1].min()
    z_top = object_idx[2].min()

    x_bottom = object_idx[0].max()
    y_bottom = object_idx[1].max()
    z_bottom = object_idx[2].max()

    bbox = (x_top, y_top, z_top, x_bottom - x_top,
            y_bottom - y_top, z_bottom - z_top)

    print(treex.bbox)
    print(bbox)
    print('\n')
    save_voxel(blocks, voxel_path.replace('.schematic', '_from_schem.binvox').format(i))



