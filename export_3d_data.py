import os
from trimesh.exchange.obj import export_obj
from trimesh import Trimesh
import json
import numpy as np
from utils import load_last_mesh, get_cam_pos_obj_center
from pyrr import Matrix44

# save blender cam
def save_blender_cam(folder, theta, phi, radius=1.4):
    # prepare blender camera data
    mesh = load_last_mesh(folder)
    cam_pos, obj_center = get_cam_pos_obj_center(mesh.vertices, radius=radius, theta=theta, phi=phi)
    up_vec = np.array([0, 0, 1])
    for x in os.listdir(folder):
        if "camparam.json" in x:
            with open(os.path.join(folder, x), "r") as fp:
                cam_params = json.load(fp)["restricted"]
                #cam_params = json.load(fp)["general"]
                cam_pos = np.array(cam_params["C"]) - obj_center
                up_vec = np.array(cam_params["up"])
    #print("cam_pos", cam_pos)
    #print("obj_center", obj_center)
    eye = cam_pos+obj_center
    view_folder = os.path.join(folder, str(theta)+"_"+str(phi)+"_"+str(radius))
    if not os.path.exists(view_folder):
        os.mkdir(view_folder)
    cam_file = os.path.join(view_folder, "blender_cam")
    np.save(cam_file, np.array([cam_pos, np.array(Matrix44.look_at(eye, obj_center, up_vec))], dtype=object))

def export_strokes(strokes, obj_file_name):
    obj_file_txt = ""
    p_counter = 0
    for s_id, s in enumerate(strokes):
        if len(s) == 0:
            continue
        p_counter += 1
        #obj_file_txt += "o ["+str(s_id)+"]\n"
        for p_id, p in enumerate(s):
            obj_file_txt += "v "+str(p[0])+" "+str(p[1])+" "+str(p[2])+"\n"
        for p_id, p in enumerate(s[:-1]):
            obj_file_txt += "l "+str(p_counter)+" "+str(p_counter+1)+"\n"
            p_counter += 1
    #print(obj_file_txt)
    with open(obj_file_name, "w") as fp:
        fp.write(obj_file_txt)

def export_construction_lines(folder):
    with open(os.path.join(folder, "unique_edges.json"), "r") as fp:
        unique_edges = json.load(fp)
    unique_strokes = [s["geometry"] for s in unique_edges]
    export_strokes(unique_strokes, os.path.join(folder, "unique_edges.obj"))
    for s_id, s in enumerate(unique_edges):
        export_strokes([s["geometry"]], os.path.join(folder, "unique_edges"+str(s_id)+".obj"))

    with open(os.path.join(folder, "final_edges.json"), "r") as fp:
        unique_edges = json.load(fp)
    unique_strokes = [s["geometry"] for s in unique_edges.values()]
    export_strokes(unique_strokes, os.path.join(folder, "final_edges.obj"))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="", type=str, help="data_folder path")
parser.add_argument("--theta", default=60, type=float, help="Spherical coordinate theta of camera position around the object")
parser.add_argument("--phi", default=125, type=float, help="Spherical coordinate phi of camera position around the object")
parser.add_argument("--radius", default=1.4, type=float, help="Radius of camera position around the object")
args = parser.parse_args()
folder = args.folder
theta = args.theta
phi = args.phi
radius = args.radius

# load sketch lines
view_folder = os.path.join(folder, str(theta)+"_"+str(phi)+"_"+str(radius))
with open(os.path.join(view_folder, "final_edges.json"), "r") as fp:
    edges = json.load(fp)
lines_3d = [edge["geometry"] for edge in edges.values()]
export_strokes(lines_3d, os.path.join(view_folder, "final_edges.obj"))


save_blender_cam(folder, theta, phi, radius)
for i in range(30):

    # load feature faces
    feature_lines_file_name = os.path.join(folder, "feature_faces_"+str(i)+".json")
    if not os.path.exists(feature_lines_file_name):
        continue
    with open(feature_lines_file_name, "r") as fp:
        feature_lines = json.load(fp)
    if len(feature_lines) == 0:
        continue
    feature_line_ids = feature_lines.keys()
    for f_id in feature_line_ids:
        mesh = feature_lines[f_id]
        vertices = []
        face_ids = []
        v_counter = 0
        for f_i, f in enumerate(mesh):
            for v in f:
                vertices.append(v)
            face_ids.append([v_counter, v_counter+1, v_counter+2])
            v_counter += 3
        tri_mesh = Trimesh(vertices=vertices, faces=face_ids)
        mesh_str = export_obj(tri_mesh)
        with open(os.path.join(folder, "feature_faces_"+str(i)+"_"+f_id+".obj"), "w") as fp:
            fp.write(mesh_str)

    # load feature lines
    feature_lines_file_name = os.path.join(folder, "feature_lines_"+str(i)+".json")
    if not os.path.exists(feature_lines_file_name):
        continue
    with open(feature_lines_file_name, "r") as fp:
        feature_lines = json.load(fp)
    if len(feature_lines) == 0:
        continue
    feature_lines = feature_lines.values()
    export_strokes(feature_lines, os.path.join(folder, "feature_lines_"+str(i)+".obj"))

#    # do the blender render
#    cmd = "/Applications/Blender.app/Contents/MacOS/Blender -b -P blender_result.py -- "+folder+" "+str(theta)+" "+str(phi)
#    os.system(cmd)