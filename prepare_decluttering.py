import numpy as np
from time import time
from aabbtree import AABBTree, AABB
from scipy.spatial.distance import cdist, squareform, pdist
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
#from fitCurves import fitCurve
#from bezier import q as eval_bezier
#from bezier import qprime as eval_tangent_bezier
from skspatial.objects import Line
from scipy.spatial.distance import directed_hausdorff
from copy import deepcopy
import json
import os
import polyscope as ps
import utils
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from skspatial.objects import Plane
import declutter_gurobi
from render_shapes import features_lines_to_svg, typed_feature_lines_to_svg, typed_feature_lines_to_svg_successive
import utils_occ

line_priorities = {
    "fillet_line": -1,
    "extrude_line": 0,
    "sketch": 1,
    "feature_line": 1,
    "silhouette_line": 1,
    "grid_lines": 2,
    "section_lines": 2,
    "circle_square_line": 2,
}

line_priorities_inv = {
    -1: "fillet_line",
    0: "extrude_line",
    1: "feature_line",
    #1: "silhouette_line",
    2: "grid_lines",
}

def filter_identical_bvh(all_edges, only_feature_lines=False):
    if only_feature_lines:
        line_priorities_inv[0] = "feature_line"
        line_priorities_inv[1] = "extrude_line"
        line_priorities["feature_line"] = 0
        line_priorities["extrude_line"] = 1
    identical_edges = []
    all_pts = [p for edge in all_edges for p in edge["geometry"]]
    all_pts_line_ids = np.array([edge_id for edge_id, edge in enumerate(all_edges) for p in edge["geometry"]])
    line_pts_ids = [[] for edge in all_edges]
    global_cnt = 0
    for edge_id, edge in enumerate(all_edges):
        for p_id, p in enumerate(edge["geometry"]):
            line_pts_ids[edge_id].append(global_cnt)
            global_cnt += 1

    dists = squareform(pdist(all_pts))
    #print(dists)
    #print(all_pts_line_ids)
    for edge_id, edge in enumerate(all_edges):
        all_edges[edge_id]["original_labels"] = [{"type": edge["type"], "feature_id": edge["feature_id"]}]
        tmp_identical_edges = []
        min_edge_priority = line_priorities[edge["type"]]
        if edge_id in identical_edges:
            continue
        geom = np.array(edge["geometry"])
        # check for intersections with previous lines
        close_line_ids = []
        for p in np.isclose(dists[line_pts_ids[edge_id]], 0.0, atol=1e-4):
            close_line_ids += all_pts_line_ids[p].tolist()
        #print(close_line_ids)
        close_line_ids = np.array(close_line_ids)
        original_feature_line_included = False
        if edge["type"] == "feature_line" or edge["type"] == "silhouette_line":
            original_feature_line_included = True
        #exit()
        #for next_edge_id, next_edge in enumerate(all_edges):
        for next_edge_id in np.unique(close_line_ids):
            next_edge = all_edges[next_edge_id]
            if next_edge_id <= edge_id:
                continue
            if next_edge_id in identical_edges:
                continue
            next_geom = np.array(next_edge["geometry"])
            #if edge_id == 235 and next_edge_id == 241:
            #    print(edge["fitted_curve"])
            #    print(hd)
            if len(geom) > 2 and len(next_geom) > 2:
                s1 = utils_occ.sample_fitted_curve(edge["fitted_curve"])
                s2 = utils_occ.sample_fitted_curve(next_edge["fitted_curve"])
                hd = directed_hausdorff(s1, s2)[0]
                eps = 0.01*min(utils.line_3d_length(s1), utils.line_3d_length(s2))
                #print(hd, eps, not np.isclose(hd, 0.0, atol=eps))
                #print(utils.line_3d_length(geom))
                #print(utils.line_3d_length(next_geom))
                #print(np.isclose(abs(utils.line_3d_length(geom) - utils.line_3d_length(next_geom)), 0.0, atol=1e-4))
                #ps.init()
                #utils.plot_curves([s1, s2])
                #ps.show()
                if not np.isclose(hd, 0.0, atol=eps):
                    continue
            elif np.sum(close_line_ids == next_edge_id) < len(edge["geometry"]):
                continue
            #if edge_id == 235 and next_edge_id == 241:
            #    print(abs(utils.line_3d_length(geom) - utils.line_3d_length(next_geom)), 0.0)
            #    print("finally")
            if not (len(geom) > 2 and len(next_geom) > 2) and not np.isclose(abs(utils.line_3d_length(geom) - utils.line_3d_length(next_geom)), 0.0, atol=1e-4):
                continue
            #    exit()
            if (len(geom) == 2 and len(next_geom) == 2) and not np.isclose(utils.chamfer_distance(geom, next_geom), 0.0, atol=1e-4):
                continue
            #if len(geom) > 2 and len(next_geom) > 2:
            #    print("merged")
            identical_edges.append(next_edge_id)
            if not(all_edges[edge_id]["type"] == "silhouette_line" and next_edge["type"] == "silhouette_line"):
                min_edge_priority = np.minimum(min_edge_priority, line_priorities[next_edge["type"]])
                all_edges[edge_id]["type"] = line_priorities_inv[min_edge_priority]
            all_edges[edge_id]["original_labels"].append({"type": next_edge["type"], "feature_id": next_edge["feature_id"]})
            tmp_identical_edges.append(next_edge_id)
            #if edge["type"] == "silhouette_line" or next_edge["type"] == "silhouette_line":
            #    print(edge_id, next_edge_id)
            #    s1 = utils_occ.sample_fitted_curve(edge["fitted_curve"])
            #    s2 = utils_occ.sample_fitted_curve(next_edge["fitted_curve"])
            #    ps.init()
            #    utils.plot_curves([s1, s2])
            #    ps.show()
            if next_edge["type"] == "feature_line" or next_edge["type"] == "silhouette_line":
                original_feature_line_included = True
        all_edges[edge_id]["original_feature_line"] = original_feature_line_included
    #print(np.unique(identical_edges))
    for i in reversed(np.unique(identical_edges)):
        #if not ((all_edges[i]["type"] == "feature_line") or (all_edges[i]["type"] == "sketch") or \
        #        (all_edges[i]["type"] == "extrude_line")):
        del all_edges[i]
    return all_edges

def filter_identical(all_edges, only_feature_lines=False):
    if only_feature_lines:
        line_priorities_inv[0] = "feature_line"
        line_priorities_inv[1] = "extrude_line"
        line_priorities["feature_line"] = 0
        line_priorities["extrude_line"] = 1
    identical_edges = []
    for edge_id, edge in enumerate(all_edges):
        tmp_identical_edges = []
        min_edge_priority = line_priorities[edge["type"]]
        if edge_id in identical_edges:
            continue
        geom = np.array(edge["geometry"])
        # check for intersections with previous lines
        for next_edge_id, next_edge in enumerate(all_edges):
            if next_edge_id <= edge_id:
                continue
            if next_edge_id in identical_edges:
                continue
            next_geom = np.array(next_edge["geometry"])
            if not np.isclose(abs(utils.line_3d_length(geom) - utils.line_3d_length(next_geom)), 0.0):
                continue
            if np.isclose(utils.chamfer_distance(geom, next_geom), 0.0, atol=1e-5):
                identical_edges.append(next_edge_id)
                min_edge_priority = np.minimum(min_edge_priority, line_priorities[next_edge["type"]])
                all_edges[edge_id]["type"] = line_priorities_inv[min_edge_priority]
                tmp_identical_edges.append(next_edge_id)
        #print(edge_id, tmp_identical_edges)
    for i in reversed(np.unique(identical_edges)):
        #if not ((all_edges[i]["type"] == "feature_line") or (all_edges[i]["type"] == "sketch") or \
        #        (all_edges[i]["type"] == "extrude_line")):
        del all_edges[i]
    return all_edges

def plane_clustering(all_edges):
    # only consider straight edges
    edges = [np.array(edge["geometry"]) for edge in all_edges]
    per_edge_plane_ids = {}
    planes = []
    for edge_id, edge in enumerate(edges):
        if len(edge) > 2:
            continue
        line_dir = edge[1] - edge[0]
        line_dir /= np.linalg.norm(line_dir).astype("d")
        for i in range(3):
            axis_vec = np.zeros(3)
            axis_vec[i] = 1.0
            if not np.isclose(np.abs(np.dot(axis_vec, line_dir)), 1.0):
                planes.append(Plane(point=edge[0], normal=axis_vec))
                if edge_id not in per_edge_plane_ids.keys():
                    per_edge_plane_ids[edge_id] = [len(planes)-1]
                else:
                    per_edge_plane_ids[edge_id].append(len(planes)-1)
    adj_mat = np.zeros([len(planes), len(planes)], dtype=bool)
    for plane_id, plane in enumerate(planes):
        for other_plane_id, other_plane in enumerate(planes):
            if np.isclose(np.abs(np.dot(plane.normal, other_plane.normal)), 1.0) and \
                    np.isclose(plane.distance_point_signed(other_plane.point), 0.0):
                adj_mat[plane_id, other_plane_id] = True
    cluster_graph = nx.from_numpy_array(adj_mat)
    cluster_ids = [list(c) for c in nx.connected_components(cluster_graph)]
    new_plane_ids = np.zeros(len(planes), dtype=int)
    for cluster_id, cluster in enumerate(cluster_ids):
        for plane_id in cluster:
            new_plane_ids[plane_id] = cluster_id
    for edge_id in per_edge_plane_ids.keys():
        per_edge_plane_ids[edge_id] = new_plane_ids[per_edge_plane_ids[edge_id]].tolist()
    return per_edge_plane_ids

def plane_clustering_v2(all_edges):
    # only consider straight edges
    edges = [np.array(edge["geometry"]) for edge in all_edges]
    per_edge_plane_ids = {}
    planes = []
    plane_points_d = [[], [], []]
    plane_points_d_plane_ids = [[], [], []]
    per_plane_normal_axis_id = []
    per_plane_per_axis_point_id = []
    for edge_id, edge in enumerate(edges):
        if len(edge) > 2:
            continue
        line_dir = edge[1] - edge[0]
        line_dir /= np.linalg.norm(line_dir).astype("d")
        for i in range(3):
            axis_vec = np.zeros(3)
            axis_vec[i] = 1.0
            if not np.isclose(np.abs(np.dot(axis_vec, line_dir)), 1.0):
                plane_points_d_plane_ids[i].append(len(planes))
                per_plane_normal_axis_id.append(i)
                per_plane_per_axis_point_id.append(len(plane_points_d[i]))
                plane_points_d[i].append(edge[0][i])
                planes.append(Plane(point=edge[0], normal=axis_vec))
                if edge_id not in per_edge_plane_ids.keys():
                    per_edge_plane_ids[edge_id] = [len(planes)-1]
                else:
                    per_edge_plane_ids[edge_id].append(len(planes)-1)

    x_dists = squareform(pdist(np.array(plane_points_d[0]).reshape(-1, 1)))
    y_dists = squareform(pdist(np.array(plane_points_d[1]).reshape(-1, 1)))
    z_dists = squareform(pdist(np.array(plane_points_d[2]).reshape(-1, 1)))
    dists = [x_dists, y_dists, z_dists]

    adj_mat = np.zeros([len(planes), len(planes)], dtype=bool)
    for i in range(3):
        plane_points_d_plane_ids[i] = np.array(plane_points_d_plane_ids[i])
    for plane_id, normal_id in enumerate(per_plane_normal_axis_id):
        close_dists = dists[normal_id][per_plane_per_axis_point_id[plane_id]] < 1.0e-5
        neighbours = plane_points_d_plane_ids[normal_id][close_dists]
        adj_mat[plane_id, neighbours] = True

    #for plane_id, plane in enumerate(planes):
    #    for other_plane_id, other_plane in enumerate(planes):
    #        if np.isclose(np.abs(np.dot(plane.normal, other_plane.normal)), 1.0) and \
    #            np.isclose(plane.distance_point_signed(other_plane.point), 0.0):
    #            adj_mat[plane_id, other_plane_id] = True

    cluster_graph = nx.from_numpy_array(adj_mat)
    cluster_ids = [list(c) for c in nx.connected_components(cluster_graph)]
    new_plane_ids = np.zeros(len(planes), dtype=int)
    for cluster_id, cluster in enumerate(cluster_ids):
        for plane_id in cluster:
            new_plane_ids[plane_id] = cluster_id
    for edge_id in per_edge_plane_ids.keys():
        per_edge_plane_ids[edge_id] = new_plane_ids[per_edge_plane_ids[edge_id]].tolist()
    return per_edge_plane_ids

def extract_constraint_intersections(all_edges, parsed_features):
    constraint_intersections = []
    constraint_line_ids = {}
    const_counter = 0
    eps = 1e-4
    for ent in parsed_features["entities"].values():
        if ent["type"] == "Sketch":
            for const in ent["constraints"]:
                affected_stroke_ids = []
                if "affected_element" in const.keys():
                    affected_element = const["affected_element"]
                    #print(const["affected_element"])
                    if affected_element["entityType"] == "lineSegment":
                        affected_geometry = np.array([affected_element["startPoint"],
                                                      affected_element["endPoint"]])
                    elif affected_element["entityType"] == "interpolatedSplineSegment":
                        affected_geometry = np.array([affected_element["interpolationPoints"][0],
                                                      affected_element["interpolationPoints"][-1]])
                    elif affected_element["entityType"] == "arc":
                        affected_geometry = np.array([affected_element["startPoint"],
                                                      affected_element["endPoint"]])
                    #elif affected_element["entityType"] == "unknownGeometry":
                else:
                    continue
                if const["constraintType"] in ["HORIZONTAL", "VERTICAL"] and "second_3d" in const.keys() and "second_element_pts" in const.keys():
                    print(const)
                    if not const["first_local"] or not const["second_local"]:
                        continue
                    constraint_intersections.append([const["first_3d"], const["second_3d"]])
                    second_element_pts = np.array(const["second_element_pts"])
                    line_dir = np.array(const["first_3d"]) - np.array(const["second_3d"])
                    line_dir /= np.linalg.norm(line_dir)
                    #ps.init()
                    #ps.remove_all_structures()
                    #ps.register_point_cloud("second_elem", second_element_pts)
                    #for s_id, s in enumerate(all_edges):
                    #    geom = np. array(s["geometry"])
                    #    ps.register_curve_network(str(s_id), geom,
                    #                              np.array([[i, i+1] for i in range(len(geom)-1)]))
                    #ps.show()

                    #print("np.array(affected_geometry)")
                    #print(np.array(affected_geometry))
                    for s_id, s in enumerate(all_edges):
                        if affected_element["entityType"] == "lineSegment":# and len(s["geometry"]) == 2:
                            #print("s_id", s_id)
                            h_d = directed_hausdorff(np.array(affected_geometry), np.array(s["geometry"]))[0]
                            #print(h_d)
                            if np.isclose(h_d, 0.0, atol=1e-4):
                                affected_stroke_ids.append(s_id)
                        #h_d = directed_hausdorff(np.array(second_element_pts), np.array(s["geometry"]))[0]
                        #h_d = directed_hausdorff( np.array(s["geometry"]), np.array(second_element_pts))[0]
                        #if np.isclose(h_d, 0.0):
                        #    #constraint_line_ids[const_counter] = s_id
                        #    stroke_ids = []
                    constraint_line_ids[const_counter] = {"connecting_stroke_ids": [],
                                                          "intersection_id": len(constraint_intersections)-1,
                                                          "line_3d": Line(const["first_3d"], line_dir),
                                                          "type": "projection",
                                                          "affected_stroke_ids": affected_stroke_ids}
                    const_counter += 1
                elif const["constraintType"] in ["MIDPOINT"]:
                    midpoint = np.array(const["midpoint_3d"])
                    p0 = np.array(const["first_3d"])
                    affected_stroke_ids = []

                    if "second_3d" in const.keys():
                        p1 = np.array(const["second_3d"])
                    if len(p0) == 2:
                        p1 = p0[1]
                        p0 = p0[0]
                    elif len(p1) == 2:
                        p0 = p1[0]
                        p1 = p1[1]
                    const_dir = p1 - p0
                    const_dir /= np.linalg.norm(const_dir)

                    plane_normal = np.array([ent["transform"]["z_axis"]["x"],
                                             ent["transform"]["z_axis"]["y"],
                                             ent["transform"]["z_axis"]["z"]])
                    plane_origin = midpoint
                    sketch_plane = Plane(plane_origin, plane_normal)
                    constraint_intersections.append([const["midpoint_3d"]])

                    midpoint_lines = []
                    midpoint_perp_lines = []
                    p0_perp_lines = []
                    p0_diag_lines = []
                    p1_perp_lines = []
                    p1_diag_lines = []
                    last_lines = []

                    print(affected_element)
                    if affected_element["entityType"] == "point":
                        affected_geometry = np.array(affected_element["point"])
                    for s_id, s in enumerate(all_edges):
                        geom = np.array(s["geometry"])
                        #if affected_element["entityType"] == "lineSegment" and len(s["geometry"]) == 2:
                        if affected_element["entityType"] in ["lineSegment", "arc"]:# and len(s["geometry"]) == 2:
                            #print("s_id", s_id)
                            h_d = directed_hausdorff(np.array(affected_geometry), np.array(s["geometry"]))[0]
                            #print(h_d)
                            if np.isclose(h_d, 0.0, atol=1e-4):
                                affected_stroke_ids.append(s_id)
                    if len(affected_stroke_ids) == 0:
                        continue
                    print("affected_stroke_ids", affected_stroke_ids)
                    min_affected_stroke_id = np.min(affected_stroke_ids)

                    for s_id, s in enumerate(all_edges):
                        if s_id > min_affected_stroke_id:
                            continue
                        geom = np.array(s["geometry"])
                        dir = geom[1] - geom[0]
                        dir /= np.linalg.norm(dir)
                        geom_line = Line(geom[0], dir)
                        geom_length = np.linalg.norm(geom[0]-geom[1])
                        if len(geom) > 2:
                            continue
                        if not np.isclose(sketch_plane.distance_point(geom[0]), 0.0, atol=1e-4):
                            continue
                        if not np.isclose(sketch_plane.distance_point(geom[1]), 0.0, atol=1e-4):
                            continue

                        if np.isclose(np.abs(np.dot(dir, const_dir)), 0.0, atol=1e-4):
                            if np.isclose(geom_line.distance_point(midpoint), 0.0, atol=1e-4):
                                # seg check
                                if np.linalg.norm(geom[0]-midpoint) > geom_length+eps or np.linalg.norm(geom[1]-midpoint) > geom_length+eps:
                                    continue
                                midpoint_perp_lines.append(s_id)
                                continue
                            if np.isclose(geom_line.distance_point(p0), 0.0, atol=1e-4):
                                # seg check
                                if np.linalg.norm(geom[0]-p0) > geom_length+eps or np.linalg.norm(geom[1]-p0) > geom_length+eps+eps:
                                    continue
                                p0_perp_lines.append(s_id)
                                continue
                            if np.isclose(geom_line.distance_point(p1), 0.0, atol=1e-4):
                                # seg check
                                if np.linalg.norm(geom[0]-p1) > geom_length+eps or np.linalg.norm(geom[1]-p1) > geom_length+eps:
                                    continue
                                p1_perp_lines.append(s_id)
                                continue

                        if not np.isclose(np.abs(np.dot(dir, const_dir)), 0.0, atol=1e-4) and not np.isclose(np.abs(np.dot(dir, const_dir)), 1.0, atol=1e-4):
                            if np.isclose(geom_line.distance_point(p0), 0.0, atol=1e-4):
                                # seg check
                                if np.linalg.norm(geom[0]-p0) > geom_length+eps or np.linalg.norm(geom[1]-p0) > geom_length+eps:
                                    continue
                                p0_diag_lines.append(s_id)
                                continue
                            if np.isclose(geom_line.distance_point(p1), 0.0, atol=1e-4):
                                # seg check
                                if np.linalg.norm(geom[0]-p1) > geom_length+eps or np.linalg.norm(geom[1]-p1) > geom_length+eps:
                                    continue
                                p1_diag_lines.append(s_id)
                                continue

                        if np.isclose(np.abs(np.dot(dir, const_dir)), 1.0, atol=1e-4):
                            if not np.isclose(geom_line.distance_point(midpoint), 0.0, atol=1e-4):
                                if not geom_length+eps >= np.linalg.norm(p0-p1):
                                    continue
                                last_lines.append(s_id)
                            if np.isclose(geom_line.distance_point(midpoint), 0.0, atol=1e-4):
                                # seg check
                                if np.linalg.norm(geom[0]-midpoint) > geom_length+eps or np.linalg.norm(geom[1]-midpoint) > geom_length+eps:
                                    continue
                                midpoint_lines.append(s_id)
                                continue

                    #print(midpoint_lines)
                    #print(midpoint_perp_lines)
                    #print(p0_perp_lines)
                    #print(p0_diag_lines)
                    #print(p1_perp_lines)
                    #print(p1_diag_lines)
                    #print(last_lines)
                    #ps.init()
                    #for l_id in midpoint_lines:
                    #    geom = np.array(all_edges[l_id]["geometry"])
                    #    ps.register_curve_network("midpoint_"+str(l_id), geom, np.array([[0,1]]))
                    #for l_id in midpoint_perp_lines:
                    #    geom = np.array(all_edges[l_id]["geometry"])
                    #    ps.register_curve_network("midpoint_perp_"+str(l_id), geom, np.array([[0,1]]))
                    #for l_id in p0_perp_lines:
                    #    geom = np.array(all_edges[l_id]["geometry"])
                    #    ps.register_curve_network("p0_perp_"+str(l_id), geom, np.array([[0,1]]))
                    #for l_id in p0_diag_lines:
                    #    geom = np.array(all_edges[l_id]["geometry"])
                    #    ps.register_curve_network("p0_diag_"+str(l_id), geom, np.array([[0,1]]))
                    #for l_id in p1_perp_lines:
                    #    geom = np.array(all_edges[l_id]["geometry"])
                    #    ps.register_curve_network("p1_perp_"+str(l_id), geom, np.array([[0,1]]))
                    #for l_id in p1_diag_lines:
                    #    geom = np.array(all_edges[l_id]["geometry"])
                    #    ps.register_curve_network("p1_diag_"+str(l_id), geom, np.array([[0,1]]))
                    #for l_id in last_lines:
                    #    geom = np.array(all_edges[l_id]["geometry"])
                    #    ps.register_curve_network("last_"+str(l_id), geom, np.array([[0,1]]))
                    #ps.show()
                    #exit()
                    print("midpoint affected_stroke_ids")
                    print(affected_stroke_ids)
                    constraint_line_ids[const_counter] = {"midpoint_line_ids": midpoint_lines,
                                                          "midpoint_perp_line_ids": midpoint_perp_lines,
                                                          "p0_perp_line_ids": p0_perp_lines,
                                                          "p0_diag_line_ids": p0_diag_lines,
                                                          "p1_perp_line_ids": p1_perp_lines,
                                                          "p1_diag_line_ids": p1_diag_lines,
                                                          "last_line_ids": last_lines,
                                                          "type": "midpoint",
                                                          "affected_stroke_ids": affected_stroke_ids}
                    const_counter += 1

        elif ent["type"] == "fillet":
            if not "fillet_projection_constraints" in ent.keys():
                continue
            for const in ent["fillet_projection_constraints"]:
                affected_stroke_ids = []
                constraint_intersections.append([const["first_3d"], const["second_3d"]])
                line_dir = np.array(const["first_3d"]) - np.array(const["second_3d"])
                line_dir /= np.linalg.norm(line_dir)

                found_affected_id = False
                for s_id, s in enumerate(all_edges):
                    if s["type"] != "fillet_line":
                        continue
                    # if any of the support lines matches this constraint, add the constraint with s_id
                    #print(s)
                    if not "support_lines" in s.keys():
                        continue
                    for supp in s["support_lines"]:
                        h_d = directed_hausdorff(np.array(supp), np.array([const["first_3d"], const["second_3d"]]))[0]
                        if np.isclose(h_d, 0.0, atol=1e-4):
                            affected_stroke_ids.append(s_id)
                            found_affected_id = True
                            break
                    if found_affected_id:
                        break

                constraint_line_ids[const_counter] = {"connecting_stroke_ids": [],
                                                      "intersection_id": len(constraint_intersections)-1,
                                                      "line_3d": Line(const["first_3d"], line_dir),
                                                      "type": "projection",
                                                      "affected_stroke_ids": affected_stroke_ids}
                const_counter += 1

    return constraint_intersections, constraint_line_ids

def prepare_decluttering_v2(all_edges, cam_pos, obj_center, up_vec,
                            parsed_features=None, VERBOSE=False):
    strokes = []
    if VERBOSE:
        ps.init()
    constraint_intersections = []
    if parsed_features is not None:
        constraint_intersections, constraint_line_ids = extract_constraint_intersections(all_edges, parsed_features)

    #print(constraint_intersections)
    #print(constraint_line_ids)
    #for const in constraint_line_ids.values():
    #    print(const)
    #exit()
    #ps.init()
    #ps.remove_all_structures()
    #for const_id, const in enumerate(constraint_line_ids.values()):
    #    edge = np.array(constraint_intersections[const["intersection_id"]])
    #    ps.register_curve_network(str(const_id), edge, np.array([[0, 1]]))
    #ps.show()
    intersections = []
    #old_per_edge_plane_ids = plane_clustering(all_edges)
    per_edge_plane_ids = plane_clustering_v2(all_edges)
    # sanity check
    #for edge_id in old_per_edge_plane_ids.keys():
    #    if not edge_id in list(per_edge_plane_ids.keys()):
    #        print("ERROR 1")
    #        continue
    #    if (not len(np.intersect1d(old_per_edge_plane_ids[edge_id], per_edge_plane_ids[edge_id])) == len(old_per_edge_plane_ids[edge_id])) or \
    #            (len(old_per_edge_plane_ids[edge_id]) != len(per_edge_plane_ids[edge_id])):
    #        print("ERROR 2")

    # fit curves for tangent computations
    #fitted_curves_2d = {}
    #for edge_id, edge_dict in enumerate(all_edges):
    #    if len(edge_dict["geometry"]) < 3:
    #        continue
    #    edge = np.array(edge_dict["geometry"])
    #    #print(edge)
    #    projected_edge = utils.project_points([edge], cam_pos, obj_center)[0]
    #    #print(projected_edge)
    #    fitted_curve = fitCurve(projected_edge, 1)
    #    #print(fitted_curve)
    #    fitted_curves_2d[edge_id] = {"linestring": LineString(projected_edge),
    #                                 "bspline": fitted_curve}
        #exit()
    #print(per_edge_plane_ids)
    #print(all_edges)

    # construct line aabb tree to efficiently test for intersections
    tree = AABBTree()
    eps = 2.0e-5
    aabbs = []
    for edge_id, edge in enumerate(all_edges):
        points = np.array(edge["geometry"])
        max = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
        min = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
        box = AABB([(min[0]-eps, max[0]+eps),
                    (min[1]-eps, max[1]+eps),
                    (min[2]-eps, max[2]+eps)])
        tree.add(box, edge_id)
        aabbs.append(box)

    for edge_id, edge in enumerate(all_edges):
        #projection_constraint_counter = np.zeros([len(constraint_intersections), 2], dtype=bool)
        geom = np.array(edge["geometry"])
        projection_constraint_ids = []
        if len(geom) == 2:
            line_dir = np.array(geom[0]) - np.array(geom[1])
            line_dir /= np.linalg.norm(line_dir)
            edge_line = Line(np.array(geom[0]), line_dir)
            for const_id, const in enumerate(constraint_line_ids.values()):
                if const["type"] != "projection":
                    continue
                #if edge_id < const["stroke_id"]:
                intersected, tmp_intersections, tmp_tangents = \
                    utils.polyline_polyline_intersection(np.array(constraint_intersections[const["intersection_id"]]), geom)
                if intersected and len(tmp_intersections) > 1 and \
                        np.all([utils.pt_withing_seg_v2(p, geom) for p in constraint_intersections[const["intersection_id"]]]):
                    const["connecting_stroke_ids"].append(edge_id)
                    projection_constraint_ids.append(const_id)
                #if const["intersection_id"] == 6 and edge_id == 48:
                #    print(constraint_intersections[const["intersection_id"]])
                #    print(geom)
                #    print(intersected, tmp_intersections)
                #    exit()
                #if intersected and len(tmp_intersections) > 1:
                #    const["connecting_stroke_ids"].append(edge_id)
                #    projection_constraint_ids.append(const_id)
                #    #projection_constraint_ids.append(const["intersection_id"])

        #if VERBOSE:
        #    utils.plot_curves([geom], str(edge_id), color=np.random.uniform(0, 1, 3))
        edge_intersections = []
        edge_intersections_ids = []
        tangent_intersections_ids = []
        edge_tangents = []
        identical_intersection_ids = []
        overlapping_stroke_ids = []
        tangent_intersection_clusters = []
        # check for intersections with previous lines
        intersecting_boxes = np.array(tree.overlap_values(aabbs[edge_id]))
        intersecting_boxes = np.array([prev_edge_id for prev_edge_id in range(0, edge_id)
                                       if aabbs[prev_edge_id].overlaps(aabbs[edge_id])])
        #print("intersecting_boxes")
        #print(intersecting_boxes)
        prev_edge_ids = intersecting_boxes[intersecting_boxes < edge_id]
        #if edge_id == 8:
        #    print("prev_edge_ids")
        #    print(prev_edge_ids)
        #    print(intersecting_boxes)
        #    exit()
        len_c1 = utils.line_3d_length(geom)
        #for prev_edge_id, prev_edge in enumerate(all_edges):
        #print("prev_edge_ids", prev_edge_ids)
        for prev_edge_id in prev_edge_ids:
            if prev_edge_id >= edge_id:
                continue
            prev_edge = all_edges[prev_edge_id]
            prev_geom = np.array(prev_edge["geometry"])
            #if np.isclose(utils.chamfer_distance(geom, prev_geom), 0.0):
            #    continue
            #print(geom)
            #print(prev_geom)
            #print("NEW")
            #if edge_id == 76 and prev_edge_id == 72:
            #    print("here")
            #if not(edge_id == 65 and prev_edge_id == 25):
            #    continue
            #if edge_id == 65 and prev_edge_id == 25:
            #    print("prev_edge_id", prev_edge_id)

            # ORIGINAL
            #intersected, tmp_intersections, tmp_tangents = utils.polyline_polyline_intersection(prev_geom, geom)
            # NEW
            len_c2 = utils.line_3d_length(prev_geom)
            min_len = np.min([len_c1, len_c2])
            dist_eps = 0.01*min_len
            #print(len_c1, len_c2, "dist_eps", dist_eps)
            if edge["type"] == "silhouette_line":
                intersected, tmp_intersections, tmp_tangents = utils_occ.intersection_curve_curve(prev_edge["fitted_curve"], edge["fitted_curve"],
                                                                                                  dist_eps=dist_eps,
                                                                                                  VERBOSE=False)
                #print(prev_edge_id, intersected)
                #if edge_id == 88 and prev_edge_id == 84:
                #    exit()
            elif len(prev_geom) == 2 and len(geom) == 2:
            # intersection between two lines
                intersected, tmp_intersections, tmp_tangents = utils.polyline_polyline_intersection(prev_geom, geom)
            elif len(prev_geom) > 2 and len(geom) > 2:
                # we don't need intersections between curves (I think)
                continue
            else:
                #if edge_id == 106 and prev_edge_id == 86:
                #    print("HERE")
                #print(edge_id, prev_edge_id)
                #print(prev_edge["geometry"])
                #print(edge["geometry"])
                intersected, tmp_intersections, tmp_tangents = utils_occ.intersection_curve_curve(prev_edge["fitted_curve"], edge["fitted_curve"],
                                                                                                  dist_eps=dist_eps)
                #if edge_id == 106 and prev_edge_id == 86:
                #    print(intersected)
                #    print(tmp_tangents)
                #    exit()

            if not intersected:
                continue
            if len(tmp_intersections) == 2:
                # overlapping strokes are not really useful for construction purposes
                # but don't use overlapping ids between curves and straight lines
                if not ((len(geom) > 2 and len(prev_geom) == 2) or (len(prev_geom) > 2 and len(geom) == 2)):
                    overlapping_stroke_ids.append(prev_edge_id)
                #continue
            intersections += deepcopy(tmp_intersections)
            edge_intersections += deepcopy(tmp_intersections)

            if np.any(tmp_tangents):
                edge_tangents.append(prev_edge_id)
            edge_intersections_ids += list(np.repeat(prev_edge_id, len(tmp_intersections)))
            tangent_intersections_ids += tmp_tangents
            #print("intersect", intersected)
        anchor_intersections = []
        if len(edge_intersections) > 0:
            edge_intersections_ids = np.array(edge_intersections_ids)
            # cluster intersections
            dists = squareform(pdist(edge_intersections))
            adj_mat = np.isclose(dists, 0.0, atol=1e-4)
            inter_graph = nx.from_numpy_array(adj_mat)
            # take most distant intersections as outer intersections
            #most_distant_inter_ids = np.argmax(dists, axis=-1)
            tmp_identical_intersection_ids = [list(c) for c in nx.connected_components(inter_graph)
                                              if not(len(list(c)) == 1 and edge_intersections_ids[list(c)[0]] in overlapping_stroke_ids)]
            # filter out overlapping strokes
            tmp_identical_intersection_ids = [[e for e in c
                                               if not edge_intersections_ids[e] in overlapping_stroke_ids]
                                              for c in tmp_identical_intersection_ids]
            if len(np.array(tmp_identical_intersection_ids, dtype=object).flatten()) > 0:
                # remove empty inter_lists
                tmp_identical_intersection_ids = [c for c in tmp_identical_intersection_ids if len(c) > 0]
                tmp_anchor_intersections = np.array([edge_intersections[c[0]] for c in tmp_identical_intersection_ids])
                most_distant_inter_ids = (-1, -1)
                if len(tmp_anchor_intersections) > 0:
                    anchor_dists = squareform(pdist(tmp_anchor_intersections))
                    most_distant_inter_ids = np.unravel_index(np.argmax(anchor_dists, axis=None), anchor_dists.shape)
                    most_distant_inter_ids = (tmp_identical_intersection_ids[most_distant_inter_ids[0]][0],
                                              tmp_identical_intersection_ids[most_distant_inter_ids[1]][0])

                clustered_intersections = list(nx.connected_components(inter_graph))
                #for c in clustered_intersections:
                for c in tmp_identical_intersection_ids:
                    if most_distant_inter_ids[0] in c or most_distant_inter_ids[1] in list(c):
                            #or np.any(np.array(edge_tangents)[c]):
                        anchor_intersections.append(edge_intersections_ids[list(c)].tolist())
                if len(geom) > 2: # circle or arc
                    for c in tmp_identical_intersection_ids:
                        tmp_cluster = []
                        for tmp_id in c:
                            if tangent_intersections_ids[tmp_id]:
                                tmp_cluster.append(edge_intersections_ids[tmp_id])
                        if len(tmp_cluster) > 0:
                            tangent_intersection_clusters.append(np.unique(tmp_cluster).tolist())

                unique_set = set()
                for c in anchor_intersections:
                    unique_set.add(tuple(np.unique(c)))
                anchor_intersections = [list(c) for c in unique_set]

                identical_intersection_ids = [edge_intersections_ids[c].tolist() for c in tmp_identical_intersection_ids]

        plane_ids = [-1]
        if edge_id in per_edge_plane_ids.keys():
            plane_ids = per_edge_plane_ids[edge_id]
        strokes.append(utils.Stroke(id=edge_id, intersections=identical_intersection_ids,
                                    previous_strokes=np.unique(edge_intersections_ids).tolist(),
                                    planes=plane_ids, type=edge["type"], anchor_intersections=anchor_intersections,
                                    tangent_intersections=tangent_intersection_clusters,
                                    overlapping_stroke_ids=overlapping_stroke_ids,
                                    projection_constraint_ids=projection_constraint_ids,
                                    original_feature_line=edge["original_feature_line"],
                                    feature_id=edge["feature_id"]))
        if VERBOSE:
            if len(intersections) > 0:
                ps.register_point_cloud("intersections", np.array(intersections))
            utils.plot_curves([geom], str(edge_id), color=np.random.uniform(0, 1, 3))
    if VERBOSE:
        ps.show()
    #ps.init()
    #if len(intersections) > 0:
    #        ps.register_point_cloud("intersections", np.array(intersections))
    #utils.plot_curves([edge["geometry"] for edge in all_edges], "edge")
    #ps.show()
    for const in constraint_line_ids.values():
        const["line_3d"] = []
        #print(const)
    #exit()
    return strokes, intersections, constraint_line_ids

def extract_strokes_dict(strokes):
    strokes_dict = []
    for s in strokes:
        strokes_dict.append({"id": int(s.id),
                             "intersections": list(s.intersections),
                             "type": s.type,
                             "previous_strokes": list(s.previous_strokes),
                             "planes": list(s.planes),
                             "anchor_intersections": [np.array(c, dtype=int).tolist() for c in s.anchor_intersections],
                             "tangent_intersections": [np.array(c, dtype=int).tolist() for c in s.tangent_intersections],
                             "overlapping_stroke_ids": np.array(s.overlapping_stroke_ids, dtype=int).tolist(),
                             "projection_constraint_ids": np.array(s.projection_constraint_ids, dtype=int).tolist(),
                             "original_feature_line": s.original_feature_line,
                             "feature_id": s.feature_id})
    return strokes_dict

if __name__ == "__main__":
    abc_id = 22
    theta = 60
    phi = 50
    data_folder = os.path.join("data", str(abc_id))
    all_edges_file_name = os.path.join(data_folder, "all_edges.json")
    with open(all_edges_file_name, "r") as f:
        all_edges = json.load(f)
    #all_edges = all_edges[:20]
    final_curves = [edge["geometry"] for edge_id, edge in enumerate(all_edges)]
    #ps.init()
    #utils.plot_curves(final_curves, "final_curve", enabled=False)
    #ps.show()
    #for edge_id, edge in enumerate(all_edges):
    #    if edge["type"] == "extrude_line":
    #        print(edge_id, edge["type"])
    #print(all_edges[479])
    #print("dist", np.linalg.norm(np.array(all_edges[479]["geometry"][0])-np.array(all_edges[479]["geometry"][1])))
    #exit()
    all_edges = filter_identical(all_edges)
    #for edge_id, edge in enumerate(all_edges):
    #    print(edge_id, edge["type"])
    svg_file_name = os.path.join("data", str(abc_id), "input.svg")
    typed_feature_lines_to_svg_successive(deepcopy(all_edges), svg_file_name=svg_file_name,
                                          theta=theta, phi=phi)
    for edge_id, edge in enumerate(all_edges):
        print(edge_id, edge["type"])

    #strokes = prepare_decluttering(all_edges, VERBOSE=True)
    strokes = prepare_decluttering(all_edges, VERBOSE=False)
    # save prepared strokes
    with open(os.path.join(data_folder, "strokes_dict.json"), "w") as fp:
        strokes_dict = []
        for s in strokes:
            strokes_dict.append({"id": int(s.id),
                                 "intersections": list(s.intersections),
                                 "type": s.type,
                                 "previous_strokes": list(s.previous_strokes),
                                 "planes": list(s.planes),
                                 "anchor_intersections": [np.array(c, dtype=int).tolist() for c in s.anchor_intersections],
                                 "overlapping_stroke_ids": list(s.overlapping_stroke_ids)})
        json.dump(strokes_dict, fp)
    with open(os.path.join(data_folder, "filtered_all_edges.json"), "w") as fp:
        json.dump(all_edges, fp)
    exit()
    for s in strokes:
        print(s)
    selected_stroke_ids = declutter_gurobi.declutter(strokes)
    print("selected_stroke_ids", selected_stroke_ids)
    final_curves = [edge["geometry"] for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
    construction_lines = [edge["geometry"] for edge_id, edge in enumerate(all_edges)
                          if edge_id in selected_stroke_ids and edge["type"] != "sketch" and edge["type"] != "feature_line"]
    feature_lines = [edge["geometry"] for edge_id, edge in enumerate(all_edges)
                     if edge_id in selected_stroke_ids and ((edge["type"] == "sketch") or (edge["type"] == "feature_line"))]
    final_edges = [edge for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
    #for edge_id, edge in enumerate(final_edges):
    #    print(edge_id, edge["type"])
    for new_s_id, s_id in enumerate(selected_stroke_ids):
        print(s_id, new_s_id)
    #ps.init()
    #ps.remove_all_structures()
    #utils.plot_curves(final_curves, "final_curve", enabled=False)
    #utils.plot_curves(feature_lines, "feature_line", color=(0, 0, 0))
    #utils.plot_curves(construction_lines, "construction_line", color=(1, 0, 0))
    #ps.show()
    #strokes_file = os.path.join(data_folder, "strokes.json")
    #with open(strokes_file, "w") as f:
    #    json.dump(strokes, f)

    #prepare_decluttering(all_edges)
    final_edges = [edge for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
    #for edge_id, edge in enumerate(final_edges):
    #    print(edge_id, edge["type"])
    svg_file_name = os.path.join("data", str(abc_id), "decluttered.svg")
    typed_feature_lines_to_svg(deepcopy(final_edges), svg_file_name=svg_file_name,
                               theta=theta, phi=phi, title="Final drawing")
    typed_feature_lines_to_svg_successive(deepcopy(final_edges), svg_file_name=svg_file_name,
                                          theta=theta, phi=phi)
    pdf_file_name = os.path.join("data", str(abc_id), "final_output.pdf")
    os.system("rsvg-convert -f pdf "+svg_file_name+" > "+pdf_file_name)
