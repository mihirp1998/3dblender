from __future__ import print_function
# import ipdb
import math, sys, random, argparse, json, os, tempfile, pickle
import mathutils
import shutil
# import pdb
import subprocess
import os
import numpy as np
import time
import pathos.pools as pp
# st = ipdb.set_trace
import bpy
import ipdb 
st = ipdb.set_trace
xmax, ymax, zmax = 5.0, 5.0, 5.0
# from preprocess_settings import *
# blender --background --python preprocess_shapenet.py
# obj_filename = '/home/shamitl/blender_stuff/8d458ab12073c371caa2c06fded3ca21/models/model_normalized.obj'
# target_filename = '/home/shamitl/projects/3dblender/data/shapes/model_normalized.obj'

# root = "/projects/katefgroup/datasets/shamit_shapenet/ShapeNetCore.v2"
# init()
# init_all()
# cars: 02958343
root = "/home/mprabhud/dataset/keypoint_net/models/ShapeNetCore.v2"
def job(instances):
    def combine_objects():
        """combine all objects in the scene
        """
        scene = bpy.context.scene
        obs = []

        for ob in scene.objects:
        # whatever objects you want to join...
            if ob.type == 'MESH':
                obs.append(ob)

        ctx = bpy.context.copy()
        # one of the objects to join
        ctx['active_object'] = obs[0]
        ctx['selected_objects'] = obs
        # we need the scene bases as well for joining
        ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
        bpy.ops.object.join(ctx)

    def move_obj_above_ground(mesh_obj):
        minz = 999999.0
        for vertex in mesh_obj.data.vertices:
            # object vertices are in object space, translate to world space
            v_world = mesh_obj.matrix_world * mathutils.Vector((vertex.co[0],vertex.co[1],vertex.co[2]))

            if v_world[2] < minz:
                minz = v_world[2]
        
        mesh_obj.location.z = mesh_obj.location.z - minz + 0.3
        return 0 - minz + 0.3

    def clear_mesh():
        """ clear all meshes in the secene
        """
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.select = True
        bpy.ops.object.delete()

    def init():
        render_args = bpy.context.scene.render
        render_args.engine = "CYCLES"
        bpy.data.worlds['World'].cycles.sample_as_light = True
        bpy.context.scene.cycles.blur_glossy = 2.0

    def init_all():
        """init everything we need for rendering
        an image
        """
        scene_setting_init(g_gpu_render_enable)
        node_setting_init()
        cam_obj = bpy.data.objects['Camera']
        cam_obj.rotation_mode = g_rotation_mode

        bpy.data.objects['Lamp'].data.energy = 50
        bpy.ops.object.lamp_add(type='SUN')

    def node_setting_init():
        """node settings for render rgb images
        mainly for compositing the background images
        """


        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        for node in tree.nodes:
            tree.nodes.remove(node)
        
        image_node = tree.nodes.new('CompositorNodeImage')
        scale_node = tree.nodes.new('CompositorNodeScale')
        alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
        render_layer_node = tree.nodes.new('CompositorNodeRLayers')
        file_output_node = tree.nodes.new('CompositorNodeOutputFile')

        scale_node.space = g_scale_space
        file_output_node.base_path = g_syn_rgb_folder

        links.new(image_node.outputs[0], scale_node.inputs[0])
        links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
        links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
        links.new(alpha_over_node.outputs[0], file_output_node.inputs[0])

    def scene_setting_init(use_gpu):
        """initialize blender setting configurations
        """
        
        sce = bpy.context.scene.name
        bpy.data.scenes[sce].render.engine = g_engine_type
        bpy.data.scenes[sce].cycles.film_transparent = g_use_film_transparent
        #output
        bpy.data.scenes[sce].render.image_settings.color_mode = g_rgb_color_mode
        bpy.data.scenes[sce].render.image_settings.color_depth = g_rgb_color_depth
        bpy.data.scenes[sce].render.image_settings.file_format = g_rgb_file_format

        #dimensions
        bpy.data.scenes[sce].render.resolution_x = g_resolution_x
        bpy.data.scenes[sce].render.resolution_y = g_resolution_y
        bpy.data.scenes[sce].render.resolution_percentage = g_resolution_percentage

        if use_gpu:
            bpy.data.scenes[sce].render.engine = 'CYCLES' #only cycles engine can use gpu
            bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
            bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
            bpy.types.CyclesRenderSettings.device = 'GPU'
            bpy.data.scenes[sce].cycles.device = 'GPU'
    instance_id_list = []
    root = "/home/mprabhud/dataset/keypoint_net/models/ShapeNetCore.v2"
    # st()
    import os
    # from __future__ import print_function
    # import ipdb
    import math, sys, random, argparse, json, os, tempfile, pickle
    import mathutils
    import shutil
    # import pdb
    import subprocess
    import os
    import numpy as np
    import time
    import pathos.pools as pp
    # st = ipdb.set_trace
    import bpy
    xmax, ymax, zmax = 5.0, 5.0, 5.0    
    instances = instances[1]
    print(instances[0])
    classes = '02958343'
    class_path = os.path.join(root, classes)
    model_dir = os.path.join(class_path, instances, "models")
    obj_filename = os.path.join(class_path, instances, "models", "model_normalized.obj")
    instance_id = classes + "_" + instances
    instance_id_list.append(instance_id)
    # target_filename = os.path.join(class_path, instances, "models", instance_id + ".obj")
    target_filename = os.path.join('/home/mprabhud/dataset/preprocessed_shapenet_2', instance_id + ".obj")

    # Delete everything
    for obj in bpy.context.scene.objects:
        obj.select = True
    bpy.ops.object.delete()
    # clear_mesh()

    imported_object = bpy.ops.import_scene.obj(filepath=obj_filename)
    combine_objects()
    # st()

    obj_object = bpy.context.selected_objects[0]
    # me = obj_object.data
    # if me.uv_textures.active is not None:
    #     for tf in me.uv_textures.active.data:
    #         if tf.image:
    #             st()
    #             img = tf.image.name
    #             print(img)

    obj_object.name = instance_id #"model_normalized"
    obj_object.data.name = instance_id

    x = obj_object.dimensions[0]
    y = obj_object.dimensions[1]
    z = obj_object.dimensions[2]
    mini = min(xmax/x, ymax/y, zmax/z)
    bpy.ops.transform.resize(value=(mini, mini, mini))
    disp = move_obj_above_ground(obj_object)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    # bpy.ops.export_scene.obj(filepath=target_filename)
    return {instance_id:{"scale": mini, "disp":disp}}


classes = '02958343'
import os
import json 
class_path = os.path.join(root, classes)
instances = os.listdir(class_path)
# instances = instances[:2]
p = pp.ProcessPool(8)
jobs = sorted(list(enumerate(instances)))
# st()
print(jobs)
results = p.map(job, jobs, chunksize = 1)
result_dict = {}
for result in results:
    
    instance_id = list(result.keys())[0]
    result_dict[instance_id] = result[instance_id]

json_path = "/home/mprabhud/dataset/shapenet_scale_disp_files/" + classes + ".json"
with open(json_path, 'w') as outfile:
    json.dump(result_dict, outfile)

# for classes in os.listdir(root):
#     # st()
#     if not classes == '02958343':
#         continue
#     class_path = os.path.join(root, classes)
#     for instances in os.listdir(class_path):
#         model_dir = os.path.join(class_path, instances, "models")
#         obj_filename = os.path.join(class_path, instances, "models", "model_normalized.obj")
#         instance_id = classes + "_" + instances
#         instance_id_list.append(instance_id)
#         # target_filename = os.path.join(class_path, instances, "models", instance_id + ".obj")
#         target_filename = os.path.join('/home/mprabhud/dataset/preprocessed_shapenet_gpu', instance_id + ".obj")

#         # Delete everything
#         for obj in bpy.context.scene.objects:
#             obj.select = True
#         bpy.ops.object.delete()
#         # clear_mesh()

#         imported_object = bpy.ops.import_scene.obj(filepath=obj_filename)
#         combine_objects()
#         # st()

#         obj_object = bpy.context.selected_objects[0]
#         # me = obj_object.data
#         # if me.uv_textures.active is not None:
#         #     for tf in me.uv_textures.active.data:
#         #         if tf.image:
#         #             st()
#         #             img = tf.image.name
#         #             print(img)

#         obj_object.name = instance_id #"model_normalized"
#         obj_object.data.name = instance_id

#         x = obj_object.dimensions[0]
#         y = obj_object.dimensions[1]
#         z = obj_object.dimensions[2]
#         mini = min(xmax/x, ymax/y, zmax/z)
#         bpy.ops.transform.resize(value=(mini, mini, mini))
#         move_obj_above_ground(obj_object)
#         bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
#         bpy.ops.export_scene.obj(filepath=target_filename)
#         # st()
#         # aa=1

#         # new_img_dir = os.path.join(model_dir, instance_id)
#         # old_img_dir  = os.path.join(class_path, instances, "images")
#         # if os.path.exists(new_img_dir):
#         #     shutil.rmtree(new_img_dir)
#         # # os.mkdir(new_img_dir)
#         # if os.path.exists(old_img_dir):
#         #     shutil.copytree(old_img_dir, new_img_dir)

#         # # st()
#         # mtl_file_path = os.path.join(model_dir, instance_id + ".mtl")
#         # new_mtl_file_path = os.path.join(model_dir, instance_id + "_.mtl")
#         # mtl_file = open(mtl_file_path, 'r')
#         # new_mtl_file = open(new_mtl_file_path, 'w')
#         # for line in mtl_file.readlines():
#         #     # st()
#         #     if 'map_Kd' in line:
#         #         st()
#         #         texture_name = line.split('/')[-1]
#         #         new_mtl_file.write('map_Kd ' + instance_id + "/images/" + texture_name + "\n")
#         #     else:
#         #         new_mtl_file.write(line)

#         # mtl_file.close()
#         # new_mtl_file.close()
#         # # st()
#         # # os.remove(mtl_file_path)
#         # os.rename(new_mtl_file_path, mtl_file_path)

# # dict_file = open("/home/mprabhud/shamit/shapenet_preprocess/dict_file.txt", "w")

# # dict_file.write('[')
# # for cnt, instanceid in enumerate(instance_id_list):
# #     if cnt == len(instance_id_list) - 1:
# #         dict_file.write("'{}'".format(instanceid))
# #     else:
# #         dict_file.write("'{}',".format(instanceid))
# # dict_file.write(']')
# # dict_file.close()

# # json_file = open("/home/mprabhud/shamit/shapenet_preprocess/json_file.txt", "w")
# # for cnt, instanceid in enumerate(instance_id_list):
# #     json_file.write('"{}":"{}"'.format(instance_id, instance_id))
# # json_file.close()



        

