from __future__ import print_function
import ipdb
import math, sys, random, argparse, json, os, tempfile, pickle
import mathutils
import shutil
import pdb
import subprocess
import os
import numpy as np
import time

st = ipdb.set_trace
import bpy
xmax, ymax, zmax = 5.0, 5.0, 5.0

# blender --background --python preprocess_shapenet.py
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

# obj_filename = '/home/shamitl/blender_stuff/8d458ab12073c371caa2c06fded3ca21/models/model_normalized.obj'
# target_filename = '/home/shamitl/projects/3dblender/data/shapes/model_normalized.obj'

root = "/projects/katefgroup/datasets/shamit_shapenet/ShapeNetCore.v2"
init()

for classes in os.listdir(root):
    class_path = os.path.join(root, classes)
    for instances in os.listdir(class_path):
        model_dir = os.path.join(class_path, instances, "models")
        obj_filename = os.path.join(class_path, instances, "models", "model_normalized.obj")
        instance_id = classes + "_" + instances
        target_filename = os.path.join(class_path, instances, "models", instance_id + ".obj")

        # # Delete everything
        # for obj in bpy.context.scene.objects:
        #     obj.select = True
        # bpy.ops.object.delete()
        clear_mesh()

        imported_object = bpy.ops.import_scene.obj(filepath=obj_filename)
        combine_objects()
        # st()

        obj_object = bpy.context.selected_objects[0]

        obj_object.name = instance_id #"model_normalized"
        obj_object.data.name = instance_id

        x = obj_object.dimensions[0]
        y = obj_object.dimensions[1]
        z = obj_object.dimensions[2]
        mini = min(xmax/x, ymax/y, zmax/z)
        bpy.ops.transform.resize(value=(mini, mini, mini))
        move_obj_above_ground(obj_object)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        bpy.ops.export_scene.obj(filepath=target_filename)
        # st()
        # aa=1

        # new_img_dir = os.path.join(model_dir, instance_id)
        # old_img_dir  = os.path.join(class_path, instances, "images")
        # if os.path.exists(new_img_dir):
        #     shutil.rmtree(new_img_dir)
        # # os.mkdir(new_img_dir)
        # if os.path.exists(old_img_dir):
        #     shutil.copytree(old_img_dir, new_img_dir)

        # # st()
        # mtl_file_path = os.path.join(model_dir, instance_id + ".mtl")
        # new_mtl_file_path = os.path.join(model_dir, instance_id + "_.mtl")
        # mtl_file = open(mtl_file_path, 'r')
        # new_mtl_file = open(new_mtl_file_path, 'w')
        # for line in mtl_file.readlines():
        #     # st()
        #     if 'map_Kd' in line:
        #         st()
        #         texture_name = line.split('/')[-1]
        #         new_mtl_file.write('map_Kd ' + instance_id + "/images/" + texture_name + "\n")
        #     else:
        #         new_mtl_file.write(line)

        # mtl_file.close()
        # new_mtl_file.close()
        # # st()
        # # os.remove(mtl_file_path)
        # os.rename(new_mtl_file_path, mtl_file_path)

        

