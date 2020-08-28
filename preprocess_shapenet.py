from __future__ import print_function
import ipdb
import math, sys, random, argparse, json, os, tempfile, pickle
import mathutils

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


obj_filename = '/home/shamitl/blender_stuff/8d458ab12073c371caa2c06fded3ca21/models/model_normalized.obj'
target_filename = '/home/shamitl/projects/3dblender/data/shapes/model_normalized.obj'

# Delete everything
for obj in bpy.context.scene.objects:
    obj.select = True
bpy.ops.object.delete()

imported_object = bpy.ops.import_scene.obj(filepath=obj_filename)
combine_objects()

obj_object = bpy.context.selected_objects[0]
obj_object.name = "model_normalized"

x = obj_object.dimensions[0]
y = obj_object.dimensions[1]
z = obj_object.dimensions[2]
mini = min(xmax/x, ymax/y, zmax/z)
bpy.ops.transform.resize(value=(mini, mini, mini))
move_obj_above_ground(obj_object)
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
bpy.ops.export_scene.obj(filepath=target_filename)

