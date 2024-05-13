import json
from utils import project_points
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import pyrr
from svgpathtools import wsvg, Path, Line
from pylowstroke.sketch_io import SketchSerializer as sk_io
from pylowstroke.sketch_core import Sketch, Stroke
from utils import spherical_to_cartesian_coords

def lookAt(center, target, up):
    f = (target - center); f = f/np.linalg.norm(f)
    s = np.cross(f, up); s = s/np.linalg.norm(s)
    u = np.cross(s, f); u = u/np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = s
    m[1, :-1] = u
    m[2, :-1] = -f
    m[-1, -1] = 1.0

    return m

# the radius is a scale factor of the bbox diagonale
def render_shapes(feature_lines_file, radius, theta=0, phi=45):

    with open(feature_lines_file, "r") as f:
        feature_lines = json.load(f)
    #print(feature_lines)
    #ps.init()
    #for curve_id, curve_geom in enumerate(feature_lines):
    #    edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
    #    ps.register_curve_network(str(curve_id), nodes=np.array(curve_geom),
    #                              edges=edges_array, color=(0, 0, 0))
    #ps.show()

    points = np.array([p for l in feature_lines.values() for p in l])
    max = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
    min = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
    bbox_diag = np.linalg.norm(max - min)
    #print("bbox:", min, max)
    obj_center = (max+min)/2
    cam_pos = np.array([(min[0]+max[0])/2, 10*min[1], 10*max[2]])
    cam_pos = np.array([max[0], 10*min[1], 10*max[2]])
    cam_pos = spherical_to_cartesian_coords(radius*bbox_diag, np.deg2rad(theta), np.deg2rad(phi))
    #print(cam_pos)
    look_vec = obj_center - cam_pos
    look_vec /= np.linalg.norm(look_vec)
    #print(look_vec)
    tilt = -look_vec[2]*90
    heading = look_vec[0]*90
    #heading = 0
    #print(tilt)

    view_mat = pyrr.matrix44.create_look_at(cam_pos,
                                            #obj_center,
                                            np.array([0, 0, 0]),
                                            np.array([0, 0, 1]))
    #proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=min[0], right=max[0], top=max[2], bottom=min[2],
    #                                                                          near=0.00001, far=10.0)
    left = -0.04
    right = 0.04
    bottom = 0.017
    top = -0.03
    near = 0.001
    far = 1.0
    proj_mat = pyrr.matrix44.create_perspective_projection_matrix(fovy=45., aspect=4/3, near=near, far=far)
    view_edges = []
    total_view_points = []
    for f_line in feature_lines.values():
        view_points = []
        for p in f_line:
            p -= obj_center
            hom_p = np.ones(4)
            hom_p[:3] = p
            proj_p = np.matmul(view_mat.T, hom_p)
            #proj_p[:3] /= proj_p[-1]
            #print(proj_p)
            #proj_p[:3] /= proj_p[-1]
            #view_points.append(proj_p[:3])
            view_points.append(proj_p)
            #total_view_points.append(proj_p[:3])
            total_view_points.append(proj_p)
        view_edges.append(np.array(view_points))
    print("view space")
    for f_line in view_edges:
        plt.plot(f_line[:, 0], f_line[:, 1], c="black")
    ylim = plt.ylim()
    #plt.gca().set_ylim(ylim[1], ylim[0])
    plt.show()
    #view_points = []
    #for p in points:
    #    hom_p = np.ones(4)
    #    hom_p[:3] = p
    #    proj_p = np.matmul(view_mat, hom_p)
    #    #print(proj_p)
    #    proj_p[:3] /= proj_p[-1]
    #    view_points.append(proj_p[:3])
    total_view_points = np.array(total_view_points)
    max = np.array([np.max(total_view_points[:, 0]), np.max(total_view_points[:, 1]), np.max(total_view_points[:, 2])])
    min = np.array([np.min(total_view_points[:, 0]), np.min(total_view_points[:, 1]), np.min(total_view_points[:, 2])])
    print(max)
    print(min)
    #print("bbox:", min, max)
    proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=0, right=1, bottom=0, top=1,
                                                                              near=near, far=far)
    proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=min[0], right=max[0], bottom=min[1], top=max[1],
                                                                              near=near, far=far)

    #exit()
    total_projected_points = []
    projected_edges = []
    for f_line in view_edges:
        #print(f_line)
        projected_points = []
        for p in f_line:
            #hom_p = np.ones(4)
            #hom_p[:3] = p
            #proj_p = np.matmul(proj_mat, np.matmul(view_mat.T, hom_p))
            #proj_p = np.matmul(proj_mat, hom_p)
            #print(p)
            proj_p = np.matmul(proj_mat, p)
            #print(proj_p)
            proj_p[:3] /= proj_p[-1]
            total_projected_points.append(proj_p[:2])
            projected_points.append(proj_p[:2])
        projected_edges.append(np.array(projected_points))
    total_projected_points = np.array(total_projected_points)
    #print(total_projected_points)
    #ps.init()
    #ps.register_point_cloud("pc", projected_points)
    ##ps.register_point_cloud("inital pc", points)
    #ps.show()
    #plt.scatter(total_projected_points[:, 0], total_projected_points[:, 1], c="r")
    #plt.scatter(total_view_points[:, 0], total_view_points[:, 1], c="r")
    for f_line in projected_edges:
        plt.plot(f_line[:, 0], f_line[:, 1], c="black")
    ylim = plt.ylim()
    #plt.gca().set_ylim(ylim[1], ylim[0])
    plt.show()
    #exit()

def indexed_lines_to_svg(feature_lines, indices, cam_pos, obj_center, up_vec, svg_file_name="out.svg", title=""):
    projected_edges = project_points([c["geometry"] for c in feature_lines], cam_pos, obj_center, up_vec)
    # TODO: plot for debug
    #for s in projected_edges:
    #    plt.plot(np.array(s)[:, 0], np.array(s)[:, 1])
    #plt.show()
    strokes = []
    for edge_id, edge in enumerate(projected_edges):
        s = Stroke([])
        s.add_avail_data("pressure")
        #print(edge_id)
        #print(edge)
        s.from_array(edge)
        #for p in s.points_list:
        #    print(p.coords)
        s.set_width(1.0)
        if edge_id in indices:
        #if (not edge_id in indices) and (feature_lines[edge_id]["type"] == "sketch" or feature_lines[edge_id]["type"] == "feature_line"
        #                                 or feature_lines[edge_id]["type"] == "extrude_line"):
            s.set_width(5.0)
        for p_id in range(len(s.points_list)):
            s.points_list[p_id].add_data("pressure", 1.0)
        #print(edge)
        #elif feature_lines[edge_id]["type"] == "extrude_line" or feature_lines[edge_id]["type"] == "section_line":
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 0.3)
        #else:
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 0.5)
        s.add_avail_data("pressure")
        strokes.append(s)
    sketch = Sketch(strokes)
    #print("first stroke")
    #for p in sketch.strokes[0].points_list:
    #    print(p.coords)
    #exit()
    sk_io.save(sketch, svg_file_name, title=title)

def typed_feature_lines_to_svg(feature_lines, cam_pos, obj_center, up_vec, svg_file_name="out.svg", title=""):
    projected_edges = project_points([c["geometry"] for c in feature_lines], cam_pos, obj_center, up_vec)
    # TODO: plot for debug
    #for s in projected_edges:
    #    plt.plot(np.array(s)[:, 0], np.array(s)[:, 1])
    #plt.show()
    strokes = []
    for edge_id, edge in enumerate(projected_edges):
        s = Stroke([])
        s.add_avail_data("pressure")
        #print(edge_id)
        #print(edge)
        s.from_array(edge)
        #for p in s.points_list:
        #    print(p.coords)
        s.set_width(1.0)
        if feature_lines[edge_id]["type"] == "outline":
            s.set_width(3.0)
            for p_id in range(len(s.points_list)):
                s.points_list[p_id].add_data("pressure", 1.0)
        else:
            for p_id in range(len(s.points_list)):
                s.points_list[p_id].add_data("pressure", 1.0)
        #print(edge)
        #elif feature_lines[edge_id]["type"] == "extrude_line" or feature_lines[edge_id]["type"] == "section_line":
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 0.3)
        #else:
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 0.5)
        s.add_avail_data("pressure")
        strokes.append(s)
    sketch = Sketch(strokes)
    #print("first stroke")
    #for p in sketch.strokes[0].points_list:
    #    print(p.coords)
    #exit()
    sk_io.save(sketch, svg_file_name, title=title)

def typed_feature_lines_to_svg_successive(feature_lines, cam_pos, obj_center, up_vec, svg_file_name="out.svg", title=""):
    projected_edges = project_points([c["geometry"] for c in feature_lines], cam_pos, obj_center, up_vec)
    # TODO: plot for debug
    #for s in projected_edges:
    #    plt.plot(np.array(s)[:, 0], np.array(s)[:, 1])
    #plt.show()
    strokes = []
    for edge_id, edge in enumerate(projected_edges):
        s = Stroke([])
        s.add_avail_data("pressure")
        #print(edge_id)
        #print(edge)
        s.from_array(edge)
        #for p in s.points_list:
        #    print(p.coords)
        #s.set_width(1.0)
        s.set_width(5.0)
        for p_id in range(len(s.points_list)):
            s.points_list[p_id].add_data("pressure", 1.0)
        #if feature_lines[edge_id]["type"] == "sketch" or feature_lines[edge_id]["type"] == "feature_line":
        #    s.set_width(3.0)
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 1.0)
        #else:
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 0.6)
        #print(edge)
        #elif feature_lines[edge_id]["type"] == "extrude_line" or feature_lines[edge_id]["type"] == "section_line":
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 0.3)
        #else:
        #    for p_id in range(len(s.points_list)):
        #        s.points_list[p_id].add_data("pressure", 0.5)
        s.add_avail_data("pressure")
        strokes.append(s)
        sketch = Sketch(strokes)
        tmp_svg_file_name = svg_file_name.split(".svg")[0]+"_"+str(np.char.zfill(str(edge_id), 3))+".svg"
        if "id" in feature_lines[edge_id].keys():
            tmp_svg_file_name = svg_file_name.split(".svg")[0]+"_"+str(np.char.zfill(str(feature_lines[edge_id]["id"]), 3))+".svg"
        sk_io.save(sketch, tmp_svg_file_name, title=title)
        strokes[-1].set_width(0.5)
        for p_id in range(len(strokes[-1].points_list)):
            strokes[-1].points_list[p_id].add_data("pressure", 0.5)

def features_lines_to_svg(feature_lines, svg_file_name="out.svg", radius=1, theta=65, phi=35):
    projected_edges = project_points(feature_lines, radius=radius, theta=theta, phi=phi)
    strokes = []
    for edge in projected_edges:
        s = Stroke([])
        s.from_array(edge)
        strokes.append(s)
    sketch = Sketch(strokes)
    sk_io.save(sketch, svg_file_name)

def feature_lines_file_to_svg(feature_lines_file, radius=1, theta=65, phi=35):
    with open(feature_lines_file, "r") as f:
        feature_lines = json.load(f)
    features_lines_to_svg(feature_lines.values(), radius, theta, phi)

#if __name__ == "__main__":
#    # theta = height
#    # phi = ground plane
#    feature_lines_file = "data/6/feature_lines_3.json"
#    #render_shapes("data/6/feature_lines_3.json", radius=1, theta=60, phi=45)
#
#    with open(feature_lines_file, "r") as f:
#        feature_lines = json.load(f)
#    projected_edges = project_points(feature_lines, radius=1, theta=65, phi=35)
#    paths = []
#    strokes = []
#    for edge in projected_edges:
#        #edge += 300
#        s = Stroke([])
#        s.from_array(edge)
#        strokes.append(s)
#        #p = Path()
#        #for i in range(len(edge) - 1):
#        #    p.append(Line(edge[i].astype(complex), edge[i+1].astype(complex)))
#        #paths.append(p)
#    sketch = Sketch(strokes)
#    sk_io.save(sketch, "out.svg")
#
#    #wsvg(paths, filename="out.svg")
#    for f_line in projected_edges:
#        plt.plot(f_line[:, 0], f_line[:, 1], c="black")
#    #ylim = plt.ylim()
#    #plt.gca().set_ylim(ylim[1], ylim[0])
#    #plt.gca().set_axes("equal")
#    plt.gca().invert_yaxis()
#    #plt.gca().set_aspect('equal', adjustable='box')
#    plt.gca().axis("equal")
#    plt.show()
#
#    exit()
#    for theta in range(0, 180, 15):
#        for phi in range(30, 70, 10):
#            print(theta, phi)
#            render_shapes("data/5/feature_lines_3.json", radius=100, theta=theta, phi=phi)
#    #render_shapes("data/1/feature_lines_9.json", radius=1, phi=45)
#    #render_shapes("data/1/feature_lines_9.json", radius=1, phi=30)
#
#    # render feature line sequence for a given model and viewpoint
#    model_name = "1"


if __name__ == "__main__":
    sketch = sk_io.load("../sketches_json_first_viewpoint/student9/bumps/view1_concept.json")
    for s_id in range(len(sketch.strokes)):
        sketch.strokes[s_id].set_width(5.0)
        #for p_id in range(len(sketch.strokes[s_id].points_list)):
        #    sketch.strokes[s_id].points_list[p_id].pressure
    sketch.strokes = sketch.strokes[:30]
    sk_io.save(sketch, "with_intersections.svg")
