import bpy
import os
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
import numpy
import json, pickle

def sketch_mesh_to_gpencil(sketch_ob, mat=None, size=0.002):
    # Transform the sketch mesh into a curve object, extrude with the right thickness and set material
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = sketch_ob
    sketch_ob.select_set(True)
    bpy.ops.object.convert(target='GPENCIL')
    #bpy.context.object.data.pixel_factor = 0.1
    #bpy.ops.object.gpencil_modifier_add(type='GP_OPACITY')
    #bpy.context.object.grease_pencil_modifiers["Opacity"].factor = 0.5

    #ob = bpy.context.view_layer.objects.active

    #bpy.ops.object.select_all(action='DESELECT')
    #ob.select_set(True)
    #bpy.context.object.data.bevel_depth = size

    # Default to a black flat material
    #if mat is None:
    #    mat = new_emissive_mat("stroke_mat", (0, 0, 0, 1))

    #ob.active_material = mat

    bpy.ops.object.select_all(action='DESELECT')

def sketch_mesh_to_tubes(sketch_ob, mat=None, size=0.002):
    # Transform the sketch mesh into a curve object, extrude with the right thickness and set material
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = sketch_ob
    sketch_ob.select_set(True)
    bpy.ops.object.convert(target='CURVE')

    ob = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.object.data.bevel_depth = size

    # Default to a black flat material
    #if mat is None:
    #    mat = new_emissive_mat("stroke_mat", (0, 0, 0, 1))

    ob.active_material = mat

    bpy.ops.object.select_all(action='DESELECT')


MATERIAL_FILE_PATH = os.path.join("/Users/fhahnlei/Pictures/symmetric_sketch/materials.blend")

def load_mat(mat_name):
    path = MATERIAL_FILE_PATH + "\\Material\\"
    bpy.ops.wm.append(filename=mat_name, directory=path)
    mat = bpy.data.materials.get(mat_name)
    return mat

def load_sketch_mesh(sketch_name, sketch_obj_file, thickness=0.001, pressure=None, color=None):

    # Load sketch file
    bpy.ops.import_scene.obj(filepath=sketch_obj_file, split_mode='OFF', axis_forward="Y", axis_up="Z")
    ob = bpy.context.selected_objects[0]
    ob.name = f"{sketch_name}"
    # Apply your transform to correct axis, eg:
    #ob.rotation_euler[0] = numpy.deg2rad(90) # Correct axis orientations
    #bpy.ops.object.transform_apply() # Apply all transforms

    mat = load_mat("stroke-black")
    mat = mat.copy()
    tree = mat.node_tree
    print(tree.nodes)
    if not pressure is None:
        tree.nodes["Mix Shader"].inputs[0].default_value = pressure
    if not color is None:
        tree.nodes["Principled BSDF"].inputs[0].default_value = (color[0], color[1], color[2], 1)
    #bpy.data.materials["stroke-black"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 1)
    ob.active_material.blend_method = "BLEND"

    sketch_mesh_to_tubes(ob, mat, size=thickness)
    #sketch_mesh_to_gpencil(ob, mat, size=thickness)
    #sketch_mesh_to_tubes(ob, None, size=thickness)

    ob.active_material = mat

    return ob

def load_mesh(mesh_name, obj_file):
    bpy.ops.import_scene.obj(filepath=obj_file, split_mode='OFF', axis_forward="Y", axis_up="Z")
    ob = bpy.context.selected_objects[0]
    #ob.active_material.node_tree.nodes["Principled BSDF"].inputs[18].default_value = 0.5
    ob.active_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.381326, 0.617207, 0.768151, 1)
    ob.active_material.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 1.0
    ob.active_material.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 0.8
    ob.active_material.node_tree.nodes["Principled BSDF"].inputs[6].default_value = 0.35
    bpy.ops.object.shade_smooth()
    #ob.active_material.blend_method = "BLEND"

    ob.name = f"{mesh_name}"

def get_last_mesh_id(data_folder):
    max_num = 0
    for f in os.listdir(data_folder):
        if not "shape_" in f or not ".obj" in f:
            continue
        max_num = max(max_num, int(f.split("shape_")[1].split(".obj")[0]))
    return max_num

def setup_scene(envmap_path=None):

    bpy.context.scene.render.film_transparent = True

    # Environment lighting
    if envmap_path is not None:
        bpy.context.scene.world.use_nodes = True
        node_tex = bpy.context.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        node_tex.image = bpy.data.images.load(envmap_path)
        node_tree = bpy.context.scene.world.node_tree
        # Tweak saturation
        node_hsv = bpy.context.scene.world.node_tree.nodes.new("ShaderNodeHueSaturation")
        node_hsv.inputs[1].default_value = 0.2 # Set saturation
        node_tree.links.new(node_tex.outputs['Color'], node_hsv.inputs['Color'])
        node_tree.links.new(node_hsv.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])
        node_tree.links.new(node_hsv.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])
        node_tree.nodes['Background'].inputs[1].default_value = 0.7

    bpy.ops.object.select_all(action = 'SELECT')
    bpy.data.objects.get("Camera").select_set(False)
    #bpy.data.objects.get("Camera_2").select_set(False)
    bpy.ops.object.delete()
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.device = 'GPU'

if __name__ == "__main__":
    import sys
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    print(argv)
    folder = argv[0]
    theta = argv[1]
    phi = argv[2]
    radius = "1.4"

    folder = os.path.join("/Users/fhahnlei/Documents/cad2sketch", folder)

    setup_scene(os.path.join("/Applications/Blender.app/Contents/Resources/3.1/datafiles/studiolights/world/interior.exr"))

    # load camera
    view_folder = os.path.join(folder, theta+"_"+phi+"_"+radius)
    #view_folder = os.path.join(folder)
    cam_file = os.path.join(view_folder, "blender_cam.npy")
    cam_data = numpy.load(cam_file, allow_pickle=True)
#
    cam_pos = cam_data[0]
    cam = bpy.data.objects['Camera']

    # with quaternion
    #cam_rot_quat = cam_data[1]
    #cam.location = cam_pos
    #cam.rotation_mode = "QUATERNION"
    #cam.rotation_quaternion[0] = cam_rot_quat[3]
    #cam.rotation_quaternion[1] = cam_rot_quat[0]
    #cam.rotation_quaternion[2] = cam_rot_quat[1]
    #cam.rotation_quaternion[3] = cam_rot_quat[2]

    # with rot mat
    cam.rotation_mode = "XYZ"
    rot_mat = cam_data[1]
    trans = rot_mat[3,:3]
    rot_mat = rot_mat[:3,:3]
    cam.location = -rot_mat @ trans
    cam.rotation_euler = Matrix(rot_mat).to_euler('XYZ')

    cam.data.clip_start = 0.0001
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    #scene.render.engine = 'CYCLES'
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=cam.location, scale=(1, 1, 1))
    bpy.data.objects["Point"].data.energy = 2

    # go through cad sequence
    print(folder)
    max_mesh_id = get_last_mesh_id(folder)
    # rendering
    render_folder = os.path.join(view_folder, "blender")
    if not os.path.exists(render_folder):
        os.mkdir(render_folder)
    for i in range(max_mesh_id+1):
        ## load mesh
        mesh_file_name = os.path.join(folder, "shape_"+str(i)+".obj")
        if not os.path.exists(mesh_file_name):
            continue
        load_mesh(str(i), mesh_file_name)
        #continue

        # load feature lines
        obj_name = "new_feature_lines_"+str(i)
        obj_name = "feature_lines_"+str(i)
        feature_lines_file_name = os.path.join(folder, obj_name+".obj")
        if not os.path.exists(feature_lines_file_name):
            continue
        # compute bbox dimensions for feature line thickness
        dim = bpy.data.objects[str(i)].dimensions
        diag = numpy.linalg.norm(numpy.array(dim))
        try:
            load_sketch_mesh("feature_lines_"+str(i), feature_lines_file_name, color=[0, 0, 0], thickness=0.002*diag)
        except:
            continue
        #bpy.data.objects[str(i)].hide_render = True
        #bpy.data.objects[str(i)].hide_viewport = True

        #if i < 15:
        #    continue
        # load feature_faces
        feature_faces_collection = bpy.data.collections.new("faces_"+str(i))
        bpy.context.scene.collection.children.link(feature_faces_collection)
        for x in os.listdir(folder):
            if not "feature_faces_"+str(i)+"_" in x or not ".obj" in x:
                continue
            load_mesh(x, os.path.join(folder, x))
            #bpy.data.objects[x].hide_render = True
            #bpy.data.objects[x].hide_viewport = True
            feature_faces_collection.objects.link(bpy.data.objects[x])
            bpy.data.collections["Collection"].objects.unlink(bpy.data.objects[x])

        # render mesh
        #tmp_file_name = os.path.join(render_folder, "final_"+str(i)+".png")
        #bpy.data.scenes['Scene'].render.filepath = tmp_file_name
        #bpy.ops.render.render(write_still = True)
        # after render
        #feature_faces_collection.hide_render = True
        #feature_faces_collection.hide_viewport = True
        #bpy.data.objects[obj_name].hide_render = True
        #bpy.data.objects[obj_name].hide_viewport = True

    try:
        load_sketch_mesh("sketch_lines", os.path.join(view_folder, "final_edges.obj"), 
            color=[0, 0, 0], thickness=0.002*diag)
    except:
        print("Final_edges.obj not found")

    #for i in range(160):
    #    feature_lines_file_name = os.path.join(view_folder, "unique_edges"+str(i)+".obj")
    #    print(feature_lines_file_name)
    #    if not os.path.exists(feature_lines_file_name):
    #        continue
    #    # compute bbox dimensions for feature line thickness
    #    try:
    #        load_sketch_mesh("unique_edges_"+str(i), feature_lines_file_name, color=[0, 0, 0], thickness=0.002*diag)
    #    except:
    #        continue

        ## render feature lines
        #tmp_file_name = os.path.join(render_folder, obj_name+".png")
        #bpy.data.scenes['Scene'].render.filepath = tmp_file_name
        #bpy.ops.render.render(write_still = True)
        #bpy.data.objects[obj_name].hide_render = True
        #bpy.data.objects[obj_name].hide_viewport = True
        #bpy.data.objects[str(i)].hide_render = True
        #bpy.data.objects[str(i)].hide_viewport = True

    #feature_lines_file_name = os.path.join(view_folder, "unique_edges.obj")
    #load_sketch_mesh("unique_edges", feature_lines_file_name, color=[0, 0, 0], thickness=0.002*diag)
    #feature_lines_file_name = os.path.join(view_folder, "final_edges.obj")
    #load_sketch_mesh("final_edges", feature_lines_file_name, color=[0, 0, 0], thickness=0.002*diag)
