import bpy
ob = bpy.context.active_object
print(ob)
blendfile = '//home/sirdome/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.blend/Object/Tomato_material'
#section = 'Object/'
#object = 'Tomato_material.obj'

matfile = '//home/sirdome/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.blend/Material/tomato_stick'
#filepath = blendfile + section + object
#directory = blendfile + section
#filename = object

for material in bpy.data.materials:
    material.user_clear()
    bpy.data.materials.remove(material)

bpy.ops.wm.append(filename=matfile)
bpy.ops.wm.append(filename=blendfile)
bpy.data.objects['Tomato_material'].name = 'Tomato_material'
tom = bpy.data.objects['Tomato_material']

print("Active object:", tom)
print("Printing  material information")
mat = bpy.data.materials.get('tomato_stick')
print("material is:", mat)
if tom.data.materials:
    tom.data.materials[0] = mat
else:
    tom.data.materials.append(mat)
#tom.active_material.diffuse_color = (1, 0, 0)
for mat in bpy.data.materials:
    print(mat)
#print(bpy.data.materials)
path = '/home/sirdome/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.blend/Material/'
print(bpy.data.objects['Tomato_material'].active_material.name)
#bpy.ops.wm.append(filename=material_name, directory=path)





#Code 2


import bpy
ob = bpy.context.active_object
print(ob)
blendfile = '//home/sirdome/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.blend/Object/Tomato_material'
#section = 'Object/'
#object = 'Tomato_material.obj'

matfile = '//home/sirdome/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.blend/Material/tomato_stick'
#filepath = blendfile + section + object
#directory = blendfile + section
#filename = object



bpy.ops.wm.append(filename=blendfile)


for material in bpy.data.materials:
    material.user_clear()
    bpy.data.materials.remove(material)
    
bpy.data.objects['Tomato_material'].name = 'Tomato_material'
bpy.ops.wm.append(filename=matfile)
tom = bpy.data.objects['Tomato_material']
print("Activate object: ", tom)
mat = bpy.data.materials.get('tomato_stick')
tom.active_material = mat
print("Active object:", tom)
print("Printing  material information")

print(bpy.data.objects['Tomato_material'].active_material.name)

#bpy.ops.wm.append(filename=material_name, directory=path)


#Code 3. Loading objects from .obj files

import bpy
objpath = '//home/sirdome/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.obj'
bpy.ops.import_scene.obj(filepath=objpath)
bpy.context.scene.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()
bpy.context.scene.objects.active.name = 'Tomato_material'
print("Selected obj:", bpy.context.selected_objects[0])

name = "Tomato_material"
new_name = name + "_0"
bpy.data.objects[name].name = new_name




objpath = '//home/sirdome/shamit/3dblender/clevr-dataset-gen/image_generation/data/shapes/Tomato_material.obj'
bpy.ops.import_scene.obj(filepath=objpath)
bpy.context.scene.objects.active = bpy.context.selected_objects[0]
#bpy.ops.object.join()
bpy.context.scene.objects.active.name = 'Tomato_material1'


print("active object name")
print(bpy.context.scene.objects.active.name)
print("printint objects")
for obj in bpy.data.objects:
    print(obj)
