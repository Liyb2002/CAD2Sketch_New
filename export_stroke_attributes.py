import argparse, os, json
import numpy as np
from utils import load_last_mesh, get_cam_pos_obj_center, project_points
from pylowstroke.sketch_core import Sketch, Stroke
from pylowstroke.sketch_io import SketchSerializer as sk_io
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="", type=str, help="data_folder path")
parser.add_argument("--theta", default=60, type=float, help="Spherical coordinate theta of camera position around the object")
parser.add_argument("--phi", default=125, type=float, help="Spherical coordinate phi of camera position around the object")
parser.add_argument("--radius", default=1.4, type=float, help="Radius of camera position around the object")
parser.add_argument("--feature", default="feature_id", type=str, help="feature to colorize. Has to be one of the following: line_type, feature_id")
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

mesh = load_last_mesh(folder)
cam_pos, obj_center = get_cam_pos_obj_center(mesh.vertices, radius=radius, theta=theta, phi=phi)
up_vec = np.array([0, 0, 1])

cmap = sns.color_palette("Dark2", 8).as_hex()
# get all features and assign each one a color
all_features = set()
for edge in edges.values():
    all_features.add(edge[args.feature])
all_features = np.sort(list(all_features))
features_cmap = {}
for feat_id,feat in enumerate(all_features):
    features_cmap[feat] = str(cmap[feat_id])

projected_edges = project_points(lines_3d, cam_pos, obj_center, up_vec)
strokes = []
for edge_id, edge in enumerate(projected_edges):
    s = Stroke([])
    s.add_avail_data("pressure")
    s.from_array(edge)
    s.set_width(3.0)
    for p_id in range(len(s.points_list)):
        s.points_list[p_id].add_data("pressure", 1.0)
    s.add_avail_data("pressure")
    s.svg_color = features_cmap[edges[str(edge_id)][args.feature]]
    strokes.append(s)
sketch = Sketch(strokes)

file_name = os.path.join(view_folder, "strokes_attribute.svg")
print("Exported to ", file_name)
sk_io.save(sketch, file_name)