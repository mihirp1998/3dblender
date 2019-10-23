# import bpy
# import ipdb

# # bpy.ops.wm.open_mainfile(filepath="data/base_scene_full.blend")
# #bpy.data.objects['Cube'].select_set(True)
# #bpy.ops.object.delete()
# bpy.context.scene.render.engine = 'CYCLES'
# bpy.ops.import_scene.obj(filepath='/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/base_scene_full.obj')
# bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
# #bpy.ops.object.join()
# # st()
# bpy.context.view_layer.objects.active.name = 'base_material'
# # print("Selected obj:", bpy.context.selected_objects[0])




# bpy.ops.import_scene.obj(filepath='/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.obj')
# bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
# #bpy.ops.object.join()

# bpy.context.view_layer.objects.active.name = 'Tomato_material'
# print("Selected obj:", bpy.context.selected_objects[0])

# name = "Tomato_material"
# new_name = name + "_0"
# bpy.data.objects[name].name = new_name
# render_args = bpy.context.scene.render
# render_args.engine = "CYCLES"
# filename = "/home/zhouxian/Documents/out.png"
# render_args.filepath = filename
# import os
# #os.remove(filename)
# # st()
# #bpy.ops.render.render(write_still=True)
# print("hello")







'''
import bpy
import ipdb

#bpy.ops.wm.open_mainfile(filepath="home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/base_scene_full.blend")
bpy.ops.wm.open_mainfile(filepath="/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/base_scene_full.blend")

#bpy.data.objects['Cube'].select_set(True)
#bpy.ops.object.delete()
#bpy.context.scene.render.engine = 'CYCLES'
#bpy.ops.import_scene.obj(filepath='/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes//base_scene_full.obj')
#bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()
# st()
#bpy.context.view_layer.objects.active.name = 'base_material'
# print("Selected obj:", bpy.context.selected_objects[0])




bpy.ops.import_scene.obj(filepath='/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.obj')
bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()

bpy.context.view_layer.objects.active.name = 'Tomato_material'
print("Selected obj:", bpy.context.selected_objects[0])

name = "Tomato_material"
new_name = name + "_0"
bpy.data.objects[name].name = new_name
render_args = bpy.context.scene.render
render_args.engine = "CYCLES"
filename = "/home/zhouxian/Documents/out.png"
render_args.filepath = filename
import os
#os.remove(filename)
# st()
bpy.ops.render.render(write_still=True)
print("hello")
'''



import bpy
import ipdb

#bpy.ops.wm.open_mainfile(filepath="home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/base_scene_full.blend")
bpy.ops.wm.open_mainfile(filepath="/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/base_scene_full.blend")

#bpy.data.objects['Cube'].select_set(True)
#bpy.ops.object.delete()
#bpy.context.scene.render.engine = 'CYCLES'
#bpy.ops.import_scene.obj(filepath='/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes//base_scene_full.obj')
#bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()
# st()
#bpy.context.view_layer.objects.active.name = 'base_material'
# print("Selected obj:", bpy.context.selected_objects[0])




bpy.ops.import_scene.obj(filepath='/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.obj')
# bpy.context.scene.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()

bpy.context.scene.objects.active.name = 'Tomato_material'
# print("Selected obj:", bpy.context.selected_objects[0])

name = "Tomato_material"
new_name = name + "_0"
bpy.data.objects[name].name = new_name
render_args = bpy.context.scene.render
render_args.engine = "CYCLES"
filename = "/home/zhouxian/Documents/out.png"
render_args.filepath = filename
import os
#os.remove(filename)
# st()
bpy.ops.render.render(write_still=True)
print("hello")


'''
import bpy
bpy.ops.wm.open_mainfile(filepath="/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/base_scene_full.blend")
bpy.ops.wm.append(filename="/home/zhouxian/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/SmoothCube_v2.blend/Object/SmoothCube_v2")
render_args = bpy.context.scene.render
render_args.engine = "CYCLES"
filename = "/home/zhouxian/Documents/out.png"
render_args.filepath = filename
import os
#os.remove(filename)
# st()
bpy.ops.render.render(write_still=True)
print("hello")
'''