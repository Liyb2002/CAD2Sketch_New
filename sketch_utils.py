import numpy as np
from scipy.spatial.distance import directed_hausdorff
import polyscope as ps
import networkx as nx
from shapely.geometry import MultiPoint
from skspatial.objects import Plane, Line
from trimesh.ray import ray_triangle
from scipy.spatial.distance import cdist
from extrude_utils import get_mesh_section_lines, get_mesh_section_lines_v2
from utils import plot_curves

def compute_sketch_loops(parsed_features, entity):
    ent = parsed_features["entities"][entity]
    x_axis = np.array([ent["transform"]["x_axis"]["x"],
                       ent["transform"]["x_axis"]["y"],
                       ent["transform"]["x_axis"]["z"]])
    y_axis = np.array([ent["transform"]["y_axis"]["x"],
                       ent["transform"]["y_axis"]["y"],
                       ent["transform"]["y_axis"]["z"]])
    for face_id, face in enumerate(ent["profiles"]["faces"]):
        loops_graph = nx.Graph()
        for edge_id in ent["profiles"]["faces"][face_id]["edges"]:
            for vert_id in ent["profiles"]["edges"][edge_id]["vertices"]:
                loops_graph.add_edge(edge_id, vert_id)
        loops_edge_ids = []
        already_looped_edges = set()
        for loop in nx.connected_components(loops_graph):
            loop_edge_ids = []
            for loop_edge_id in loop:
                if loop_edge_id in ent["profiles"]["faces"][face_id]["edges"]:
                    loop_edge_ids.append(loop_edge_id)
                    already_looped_edges.add(loop_edge_id)
            loops_edge_ids.append(loop_edge_ids)
        # solitary edges. p.ex. circles
        for edge_id in ent["profiles"]["faces"][face_id]["edges"]:
            if not edge_id in already_looped_edges:
                loops_edge_ids.append([edge_id])

        # sort according to loop surface area
        loops_areas = []
        for loop_edge_ids in loops_edge_ids:
            loop_vertices = []
            for edge_id in loop_edge_ids:
                if ent["profiles"]["edges"][edge_id]["param"]["type"] == "Circle" and len(ent["profiles"]["edges"][edge_id]["vertices"]) == 0:
                    radius = np.array(ent["profiles"]["edges"][edge_id]["param"]["radius"])
                    for alpha in np.linspace(0, 360, 8):
                        p = np.array(ent["profiles"]["edges"][edge_id]["param"]["coordSystem"]["origin"]) + \
                            radius*np.cos(alpha)*x_axis + radius*np.sin(alpha)*y_axis
                        loop_vertices.append(np.array([np.dot(x_axis, p), np.dot(y_axis, p)]))
                for vert_id in ent["profiles"]["edges"][edge_id]["vertices"]:
                    vert = np.array(ent["profiles"]["vertices"][vert_id]["param"]["Vector"])
                    loop_vertices.append(np.array([np.dot(x_axis, vert), np.dot(y_axis, vert)]))
            loops_areas.append(MultiPoint(loop_vertices).convex_hull.area)
        sorted_loops_edge_ids = []
        for i in reversed(np.argsort(loops_areas)):
            sorted_loops_edge_ids.append(loops_edge_ids[i])
        parsed_features["entities"][entity]["profiles"]["faces"][face_id]["loops_edge_ids"] = sorted_loops_edge_ids

def get_faces_edges_vertices(geo_ids, parsed_features):
    faces = []
    edges = []
    vertices = []
    for geo_id in geo_ids:
        found_face = False
        for sketch in parsed_features["sequence"]:
            if sketch["type"] != "Sketch":
                continue
            sketch_profiles = parsed_features["entities"][sketch["entity"]]["profiles"]
            for face in sketch_profiles["faces"]:
                if face["id"] == geo_id:
                    faces.append(face)
                    found_face = True
                    face_edges = {}
                    face_vertices = {}
                    for edge_id in face["edges"]:
                        face_edges[edge_id] = sketch_profiles["edges"][edge_id]
                        for vert_id in sketch_profiles["edges"][edge_id]["vertices"]:
                            face_vertices[vert_id] = np.array(sketch_profiles["vertices"][vert_id]["param"]["Vector"])
                    edges.append(face_edges)
                    vertices.append(face_vertices)
                    break
            if found_face:
                break
        if found_face:
            continue
    return faces, edges, vertices


def grid_intersections(grids, sketch_id, feat_id, data_folder, feature_faces_meshes_pool):
    curr_grid = grids[sketch_id]
    grid_planes = []
    # add ground plane of curr_grid
    curr_grid.plane_normal /= np.linalg.norm(curr_grid.plane_normal)
    closest_normal = np.eye(3, k=np.argmax([np.abs(np.dot(np.eye(3, k=i)[0], curr_grid.plane_normal)) for i in range(3)]))[0]
    new_plane = Plane(point=list(curr_grid.segments.values())[0].endpoints[0], normal=curr_grid.plane_normal)
    #new_plane = Plane(point=list(curr_grid.segments.values())[0].endpoints[0], normal=closest_normal)
    grid_planes.append(new_plane)
    for seg in curr_grid.segments.values():
        new_plane_normal = np.cross(curr_grid.plane_normal, seg.line.direction)
        new_plane_normal /= np.linalg.norm(new_plane_normal)
        new_plane_normal = closest_normal
        new_plane = Plane(point=seg.endpoints[0], normal=new_plane_normal)
        grid_planes.append(new_plane)
    new_lines = []
    plane_normals = [np.cross(closest_normal, np.array([1, 0, 0])),
                     np.cross(closest_normal, np.array([0, 1, 0])),
                     np.cross(closest_normal, np.array([0, 0, 1]))]
    new_lines = [l for l in get_mesh_section_lines_v2(feat_id, data_folder, plane_normals,
                                                    [seg.endpoints for seg in curr_grid.segments.values()],
                                                    feature_faces_meshes_pool)
                  if not np.isclose(np.linalg.norm(l[0] - l[1]), 0.0, atol=1e-4)]
    for plane in grid_planes:
        for tmp_grid_id in grids.keys():
            if tmp_grid_id == sketch_id:
                continue
            new_lines += intersection_quads_plane(grids[tmp_grid_id].quads, plane)
    #ps.init()
    #plot_curves(new_lines)
    #ps.show()

    # connect with "through the air" lines
    for sketch_pt in curr_grid.sketch_points.values():
        for axis_id in range(3):
            axis_vec = np.zeros(3)
            axis_vec[axis_id] = 1

            intersection_pts = np.array([sketch_pt.point])
            for tmp_grid_id in grids.keys():
                if tmp_grid_id == sketch_id:
                    continue
                for quad in grids[tmp_grid_id].quads:
                    raycaster = ray_triangle.RayMeshIntersector(quad.mesh)
                    hits, _, _ = raycaster.intersects_location(ray_origins=[sketch_pt.point],
                                                               ray_directions=[axis_vec], multiple_hits=True)
                    if len(hits) > 0:
                        intersection_pts = np.concatenate([intersection_pts, hits], axis=0)

                    hits, _, _ = raycaster.intersects_location(ray_origins=[sketch_pt.point],
                                                               ray_directions=[-axis_vec], multiple_hits=True)
                    if len(hits) > 0:
                        intersection_pts = np.concatenate([intersection_pts, hits], axis=0)
            if len(intersection_pts) == 2:
                new_lines.append([sketch_pt.point, intersection_pts[1]])
            elif len(intersection_pts) > 2:
                min_pt = intersection_pts[np.argmin(np.dot(axis_vec, intersection_pts.T))]
                max_pt = intersection_pts[np.argmax(np.dot(axis_vec, intersection_pts.T))]
                new_lines.append([min_pt, max_pt])

    return new_lines


# return intersected lines
def intersection_quads_plane(quads, plane):
    lines = []
    for quad in quads:
        new_line = quad.intersection_plane(plane)
        #for new_line in quad.intersection_plane(plane):
        if len(new_line) > 0:
            lines.append(new_line)
    return lines

def passes_through_both_points(seg, p0, p1):
    v = seg[1] - seg[0]
    length = np.linalg.norm(v)
    if np.isclose(length, 0.0):
        return False
    v /= length
    l = Line(seg[0], v)
    if np.isclose(l.distance_point(p0), 0.0, atol=1e-4) and np.isclose(l.distance_point(p1), 0.0, atol=1e-4):
        if np.linalg.norm(p0-seg[0]) > length or np.linalg.norm(p0-seg[1]) > length or \
                np.linalg.norm(p1-seg[0]) > length or np.linalg.norm(p1-seg[1]) > length:
            return False
        return True
    return False

def draw_midpoint_curves(const, sketch, grid_lines, all_edges, VERBOSE=False):

    midpoint = np.array(const["midpoint_3d"])
    p0 = np.array(const["first_3d"])
    if "second_3d" in const.keys():
        p1 = np.array(const["second_3d"])
    if len(p0) == 2:
        p1 = p0[1]
        p0 = p0[0]
    elif len(p1) == 2:
        p0 = p1[0]
        p1 = p1[1]
    seg_0 = np.array([p0, p1])
    seg_vec = seg_0[1] - seg_0[0]
    seg_vec /= np.linalg.norm(seg_vec)
    # find rectangles which we can use to draw midpoint diagonal lines
    sketch_curves = [s for s in sketch["curves"] if len(s) == 2]
    sketch_curves += [s for s in grid_lines if len(s) == 2]
    sketch_curves = np.array(sketch_curves)
    sketch_curves_vecs = sketch_curves[:, 0] - sketch_curves[:, 1]
    sketch_curves_vecs /= np.linalg.norm(sketch_curves[:, 0] - sketch_curves[:, 1], axis=-1).reshape(-1, 1)

    sketch_pts = []
    sketch_pts_line_ids = []
    for l_id, l in enumerate(sketch_curves):
        if len(l) > 2:
            continue
        sketch_pts.append(l[0])
        sketch_pts.append(l[1])
        sketch_pts_line_ids.append(l_id)
        sketch_pts_line_ids.append(l_id)
    sketch_pts = np.array(sketch_pts)
    sketch_pts_line_ids = np.array(sketch_pts_line_ids)

    dists = cdist(sketch_pts, sketch_pts)
    seg_dists = cdist([p0, p1], sketch_pts)

    p0_perp_line_ids = []
    for p0_close_pt_id in np.argwhere(seg_dists[0] < 1e-4).flatten():
        l_id = sketch_pts_line_ids[p0_close_pt_id]
        l = sketch_curves[l_id]
        if not np.isclose(np.abs(np.dot(sketch_curves_vecs[l_id], seg_vec)), 0.0):
            continue
        p0_perp_line_ids.append(l_id)

    p1_perp_line_ids = []
    for p1_close_pt_id in np.argwhere(seg_dists[1] < 1e-4).flatten():
        l_id = sketch_pts_line_ids[p1_close_pt_id]
        l = sketch_curves[l_id]
        if not np.isclose(np.abs(np.dot(sketch_curves_vecs[l_id], seg_vec)), 0.0):
            continue
        p1_perp_line_ids.append(l_id)

    rectangles = []
    for perp_0_id in p0_perp_line_ids:
        perp_0 = sketch_curves[perp_0_id]
        other_p0 = perp_0[0]
        #exit()
        other_p0_id = np.argwhere(sketch_pts_line_ids == perp_0_id).flatten()[0]
        if np.isclose(np.linalg.norm(p0 - other_p0), 0.0, atol=1e-4):
            other_p0 = perp_0[1]
            other_p0_id = np.argwhere(sketch_pts_line_ids == perp_0_id).flatten()[1]
        found_rectangle = False
        for tmp_rec in rectangles:
            if np.isclose(np.linalg.norm(tmp_rec[2]-other_p0), 0.0, atol=1e-4):
                found_rectangle = True
                break
        if found_rectangle:
            found_rectangle = False
            continue
        for perp_1_id in p1_perp_line_ids:
            perp_1 = sketch_curves[perp_1_id]
            other_p1 = perp_1[0]
            other_p1_id = np.argwhere(sketch_pts_line_ids == perp_1_id).flatten()[0]
            if np.isclose(np.linalg.norm(p1 - other_p1), 0.0, atol=1e-4):
                other_p1 = perp_1[1]
                other_p1_id = np.argwhere(sketch_pts_line_ids == perp_1_id).flatten()[1]
            for tmp_rec in rectangles:
                if np.isclose(np.linalg.norm(tmp_rec[3]-other_p1), 0.0, atol=1e-4):
                    found_rectangle = True
                    break
            if found_rectangle:
                found_rectangle = False
                continue

            for c in sketch_curves:
                if passes_through_both_points(c, other_p0, other_p1):
                    rectangles.append([p0, p1, other_p0, other_p1])
                    break
            #for fourth_p0_id in np.unique(sketch_pts_line_ids[dists[other_p0_id] < 1e-4]):
            #    found_snd_pt = False
            #    for fourth_p1_id in np.unique(sketch_pts_line_ids[dists[other_p1_id] < 1e-4]):
            #        if fourth_p1_id != fourth_p0_id:
            #            continue
            #        fourth_line = sketch_curves[fourth_p0_id]
            #        if not np.isclose(np.abs(np.dot(sketch_curves_vecs[fourth_p0_id], seg_vec)), 1.0):
            #            continue
            #        rectangles.append([p0, p1, other_p0, other_p1])
            #        found_snd_pt = True
            #        break
            #    if found_snd_pt:
            #        break
    rectangles = np.array(rectangles)

    diag_lines = []
    for rec_id, rec in enumerate(rectangles):
        diag_0 = np.array([rec[0], rec[3]])
        diag_1 = np.array([rec[1], rec[2]])
        mid_line = np.array([(rec[3]+rec[2])/2, midpoint])
        diag_lines.append(diag_0)
        diag_lines.append(diag_1)
        diag_lines.append(mid_line)
        #ps.register_curve_network("diag_0_"+str(rec_id), diag_0.reshape(-1, 3),
        #                          np.array([[0, 1]]))
        #ps.register_curve_network("diag_1_"+str(rec_id), diag_1.reshape(-1, 3),
        #                          np.array([[0, 1]]))
        #ps.register_curve_network("mid_line_"+str(rec_id), mid_line.reshape(-1, 3),
        #                          np.array([[0, 1]]))

    # find minimum insert id
    minimum_insert_id = -1
    if "affected_element" in const.keys():
        affected_element = const["affected_element"]
        if affected_element["entityType"] in ["lineSegment", "arc"]:
            affected_geometry = np.array([affected_element["startPoint"],
                                          affected_element["endPoint"]])
        affected_stroke_ids = []
        for s_id, s in enumerate(all_edges):
            geom = np.array(s["geometry"])
            if affected_element["entityType"] in ["lineSegment", "arc"]:# and len(s["geometry"]) == 2:
                h_d = directed_hausdorff(np.array(affected_geometry), np.array(s["geometry"]))[0]
                if np.isclose(h_d, 0.0, atol=1e-4):
                    affected_stroke_ids.append(s_id)
        if len(affected_stroke_ids) > 0:
            minimum_insert_id = np.min(affected_stroke_ids)

    if VERBOSE:
        ps.init()
        ps.remove_all_structures()
        for s_id, s in enumerate(all_edges):
            if len(s["geometry"]) > 2:
                continue
            ps.register_curve_network(str(s_id), np.array(s["geometry"]).reshape(-1, 3),
                                      np.array([[0, 1]]))
        ps.register_curve_network("edges", sketch_curves.reshape(-1, 3),
                                  np.array([[2*i, 2*i+1] for i in range(len(sketch_curves))]))
        ps.register_curve_network("seg", seg_0.reshape(-1, 3),
                                  np.array([[0, 1]]))
        for rec_id, rec in enumerate(rectangles):
            ps.register_point_cloud(str(rec_id), rec)
        for diag_line_id, diag_line in enumerate(diag_lines):
            ps.register_curve_network("diag_"+str(diag_line_id), diag_line.reshape(-1, 3),
                                      np.array([[0, 1]]))
        ps.show()

    #exit()
    #perpendicular_lines = []
    #parallel_lines = []

    return diag_lines, minimum_insert_id
