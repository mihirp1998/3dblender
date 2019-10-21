import bpy
import ipdb

# bpy.ops.wm.open_mainfile(filepath="data/base_scene_full.blend")
D.objects['Cube'].select_set(True)
bpy.ops.object.delete()
bpy.ops.import_scene.obj(filepath='/home/mihir/Documents/projects/3dblender/clevr-dataset-gen/image_generation/data/base_scene_full.obj')
bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()
# st()
bpy.context.view_layer.objects.active.name = 'base_material'
# print("Selected obj:", bpy.context.selected_objects[0])




bpy.ops.import_scene.obj(filepath='/home/mihir/Documents/projects/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.obj')
bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()

bpy.context.view_layer.objects.active.name = 'Tomato_material'
print("Selected obj:", bpy.context.selected_objects[0])

name = "Tomato_material"
new_name = name + "_0"
bpy.data.objects[name].name = new_name
render_args = bpy.context.scene.render
render_args.engine = "CYCLES"
filename = "/home/mihir/Documents/out.png"
render_args.filepath = filename
import os
os.remove(filename)
# st()
bpy.ops.render.render(write_still=True)
print("hello")