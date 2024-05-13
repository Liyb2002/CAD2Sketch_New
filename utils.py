import math
import utils_occ
from trimesh.ray import ray_triangle
from time import time
from scipy.optimize import least_squares, minimize
from trimesh.proximity import closest_point
from skimage.measure import EllipseModel
#from geomdl.fitting import approximate_curve
import matplotlib.pyplot as plt
import pyrr
from shapely.geometry import LineString, MultiLineString, Point
import seaborn as sns
import trimesh
from trimesh.path.polygons import medial_axis
from trimesh.exchange.obj import export_obj
import json
import os
from svgpathtools import svg2paths, wsvg
from copy import deepcopy
#import open3d as o3d
from collections import OrderedDict
import polyscope as ps
import skspatial.objects
from shapely.geometry import MultiPoint
import numpy as np
import scipy.spatial.distance
from numpy.linalg import norm
#import pymesh
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform
from skspatial.objects import Line, Points, Plane
#import sympy
from trimesh.intersections import mesh_plane
import trimesh
from sklearn.neighbors import NearestNeighbors


class Stroke:
    def __init__(self, id=0, intersections=None, planes=-1, type="feature_line", previous_strokes=None,
                 anchor_intersections=None, tangent_intersections=None, overlapping_stroke_ids=None,
                 projection_constraint_ids=None, original_feature_line=False, occlusions=None,
                 feature_id=-1):
        self.id = id
        self.intersections = intersections
        self.type = type
        self.previous_strokes = previous_strokes
        self.planes = planes
        self.anchor_intersections = anchor_intersections
        self.tangent_intersections = tangent_intersections
        self.overlapping_stroke_ids = overlapping_stroke_ids
        self.projection_constraint_ids = projection_constraint_ids
        self.original_feature_line = original_feature_line
        self.occlusions = occlusions
        self.feature_id = feature_id

    def __str__(self):
        return "stroke_id: "+str(self.id) +"\n intersection_ids: " +str(self.intersections) + \
               "\n stroke_type: " + self.type + "\n previous_stroke_ids: " + str(self.previous_strokes) + \
               "\n plane_ids: "+str(self.planes)+"\n anchor_intersection_ids: " +str(self.anchor_intersections) + \
               "\n tangent_intersection_ids: " +str(self.tangent_intersections) + \
               "\n overlapping_stroke_ids: " + str(self.overlapping_stroke_ids)


from numpy import array, linalg, matrix
from numpy.linalg import pinv
from scipy.special import comb as n_over_k
Mtk = lambda n, t, k: t**k * (1-t)**(n-k) * n_over_k(n,k)
BézierCoeff = lambda ts: [[Mtk(3,t,k) for k in range(4)] for t in ts]

fcn = np.log
tPlot = np.linspace(0. ,1. , 81)
xPlot = np.linspace(0.1,2.5, 81)
tData = tPlot[0:81:10]
xData = xPlot[0:81:10]
data = np.column_stack((xData, fcn(xData))) # shapes (9,2)

def chordLengthParameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i-1] + linalg.norm(points[i] - points[i-1]))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]
    return u

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

def fit_bezier(points):
    tData = chordLengthParameterize(points)
    Pseudoinverse = pinv(BézierCoeff(tData)) # (9,4) -> (4,9)
    control_points = Pseudoinverse.dot(points)     # (4,9)*(9,2) -> (4,2)
    return control_points

def eval_bezier(ctrlPoly, t):
    return (1.0-t)**3 * ctrlPoly[0] + 3*(1.0-t)**2 * t * ctrlPoly[1] + 3*(1.0-t)* t**2 * ctrlPoly[2] + t**3 * ctrlPoly[3]


# v1 and v2 should be normalized
# returns closest points on line 1 and on line 2
def line_line_collision(p1, v1, p2, v2):
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)

    rhs = p2 - p1
    lhs = np.array([v1, -v2, v3]).T

    t_solutions = np.linalg.lstsq(lhs, rhs, rcond=None)
    t1 = t_solutions[0][0]
    t2 = t_solutions[0][1]

    closest_line_1 = p1 + t1*v1
    closest_line_2 = p2 + t2*v2
    return np.array([closest_line_1, closest_line_2])

def pt_withing_seg_v2(pt, seg):
    # if the point is one of the endpoints
    seg_vec = seg[1] - seg[0]
    seg_len = np.linalg.norm(seg_vec)
    #print("seg_len", seg_len, np.isclose(seg_len, 0.0))
    if np.isclose(seg_len, 0.0):
        return False
    #print(np.linalg.norm(pt-seg[0]), np.linalg.norm(pt-seg[1]))
    seg_vec /= seg_len
    proj_pt = np.dot(seg_vec, pt)
    proj_a = np.dot(seg_vec, seg[0])
    proj_b = np.dot(seg_vec, seg[1])
    if np.isclose(proj_pt, proj_a, atol=1e-4) or np.isclose(proj_pt, proj_b, atol=1e-4) or (proj_pt >= proj_a and proj_pt <= proj_b):
        return True
    return False

def pt_withing_seg(pt, seg):
    # if the point is one of the endpoints
    seg_vec = seg[1] - seg[0]
    seg_len = np.linalg.norm(seg_vec)
    #print("seg_len", seg_len, np.isclose(seg_len, 0.0))
    if np.isclose(seg_len, 0.0):
        return False
    #print(np.linalg.norm(pt-seg[0]), np.linalg.norm(pt-seg[1]))
    if np.isclose(np.linalg.norm(pt-seg[0]), 0.0, atol=1e-4) or \
            np.isclose(np.linalg.norm(pt-seg[1]), 0.0, atol=1e-4):
        return True
    seg_vec /= seg_len
    a_p = pt - seg[0]
    a_p_l = np.linalg.norm(a_p)
    b_p = pt - seg[-1]
    b_p_l = np.linalg.norm(b_p)
    if not (np.isclose(a_p_l, 0.0, atol=5e-4) or np.isclose(b_p_l, 0.0, atol=5e-4)):
        # if not (np.isclose(a_p_l, 0.0) or np.isclose(b_p_l, 0.0)):
        # intersection is not coincidental with one of the extremities
        a_p /= a_p_l
        b_p /= b_p_l
        #ps.register_point_cloud("p", np.array([pt]))
        if seg_vec.dot(a_p) < 0 or seg_vec.dot(b_p) > 0:
            return False
    return True


def segment_segment_intersection_optim(seg_1, seg_2):
    v1 = seg_1[-1] - seg_1[0]
    v2 = seg_2[-1] - seg_2[0]

    def segments_eval(t):
        dist_vec = (seg_1[0] + t[0]*v1) - (seg_2[0] + t[1]*v2)
        return dist_vec[0]*dist_vec[0]+dist_vec[1]*dist_vec[1]+dist_vec[2]*dist_vec[2]

    t_0 = [0.0, 0.0]
    #res_1 = least_squares(segments_eval, t_0, bounds=(0.0, 1.0))
    res_1 = minimize(segments_eval, t_0, method='SLSQP', bounds=((0.0, 1.0), (0.0, 1.0)))
    return res_1

def segment_segment_intersection(seg_1, seg_2):
    #ps.register_curve_network("seg1", seg_1, np.array([[0, 1]]))
    #ps.register_curve_network("seg2", seg_2, np.array([[0, 1]]))
    v1 = seg_1[-1] - seg_1[0]
    v1 /= np.linalg.norm(v1)
    v2 = seg_2[-1] - seg_2[0]
    v2 /= np.linalg.norm(v2)
    # if both segments are parallel
    #print(np.isclose(np.abs(np.dot(v1, v2)), 1.0, atol=1e-9))
    #print(np.abs(np.dot(v1, v2)))
    #print(math.isclose(np.abs(np.dot(v1, v2)), 1.0, rel_tol=1e-8))
    #print(seg_1, seg_2)
    #print(v1, v2)
    l1 = Line(point=seg_1[0], direction=v1)
    l2 = Line(point=seg_2[0], direction=v2)
    #print("is_parallel", l1.direction.is_parallel(l2.direction, rel_tol=1e-5))
    if l1.direction.is_parallel(l2.direction, rel_tol=1e-5):
    #if math.isclose(np.abs(np.dot(v1, v2)), 1.0, rel_tol=1e-5):
    #if np.isclose(np.abs(np.dot(v1, v2)), 1.0):
        #l1 = Line(point=seg_1[0], direction=v1)
        #l2 = Line(point=seg_2[0], direction=v2)
        #print(np.abs(np.dot(v1, v2)))
        #print(l1.direction.cosine_similarity(l2.direction))
        #print("is_parallel", l1.direction.is_parallel(l2.direction))
        #print(l1.point, l2.point, np.linalg.norm(np.array(l1.point) - np.array(l2.point)))
        dist = l1.distance_line(l2)
        #if not l1.direction.is_parallel(l2.direction) and l1.is_coplanar(l2):
        #    print("inter", l1.intersect_line(l2))
        #print("dist", dist)
        if not np.isclose(dist, 0.0, atol=1e-5):
            return False, None, False
        # check if it is a false parallel
        if not l1.direction.is_parallel(l2.direction):
            line_inters = line_line_collision(seg_1[0], v1, seg_2[0], v2)
            #print("line_inters", line_inters)
            if not (pt_withing_seg_v2(line_inters[0], seg_1) and pt_withing_seg_v2(line_inters[1], seg_2)):
                return False, None, False
            if not (pt_withing_seg_v2(line_inters[1], seg_1) and pt_withing_seg_v2(line_inters[0], seg_2)):
                return False, None, False

        # check if partially overlapping segments
        a_1 = np.min(np.dot(v1, seg_1.T))
        b_1 = np.max(np.dot(v1, seg_1.T))
        a_2 = np.min(np.dot(v1, seg_2.T))
        b_2 = np.max(np.dot(v1, seg_2.T))
        #a_2 = np.min(np.dot(v2, seg_2.T))
        #b_2 = np.max(np.dot(v2, seg_2.T))
        #print("here")
        #print(v1, v2)
        # TODO: comparisons with np.is_close
        #print("here")
        #print(seg_1, seg_2)
        #print(a_1, b_1, a_2, b_2)
        if (a_1 <= a_2 and b_1 >= a_2 and b_1 <= b_2):
            #print("0")
            if np.isclose(a_2, b_1, atol=1e-5):
                return True, seg_2[0], True
            return True, np.array([seg_2[0], seg_1[1]]), True
        if (a_2 <= a_1 and b_2 >= a_1 and b_2 <= b_1):
            #print("1")
                if np.isclose(a_1, b_2, atol=1e-5):
                    return True, seg_1[0], True
                return True, np.array([seg_1[0], seg_2[1]]), True
        if (a_1 <= a_2 and b_2 <= b_1):
            #print("2")
            return True, np.array([seg_2[0], seg_2[1]]), True
        if (a_2 <= a_1 and b_1 <= b_2):
            #print("3")
            return True, np.array([seg_1[0], seg_1[1]]), True
        return False, None, False
    line_inters = line_line_collision(seg_1[0], v1, seg_2[0], v2)
    #print("line_inters", line_inters)
    dist = np.linalg.norm(line_inters[0]-line_inters[1])
    #print("dist non parallel", dist)
    if np.isclose(dist, 0.0, atol=1e-5):
        #print("dist", dist)
        # check if intersections are withing segments
        #print(pt_withing_seg_v2(line_inters[0], seg_1), pt_withing_seg_v2(line_inters[1], seg_2), pt_withing_seg_v2(line_inters[1], seg_1), pt_withing_seg_v2(line_inters[0], seg_2))
        if not (pt_withing_seg_v2(line_inters[0], seg_1) and pt_withing_seg_v2(line_inters[1], seg_2)):
            #print("return false")
            return False, None, False
        if not (pt_withing_seg_v2(line_inters[1], seg_1) and pt_withing_seg_v2(line_inters[0], seg_2)):
            #print("return false")
            return False, None, False
        #a_p = line_inters[0] - seg_1[0]
        #a_p_l = np.linalg.norm(a_p)
        #b_p = line_inters[0] - seg_1[-1]
        #b_p_l = np.linalg.norm(b_p)
        #if not(np.isclose(a_p_l, 0.0, atol=1e-5) or np.isclose(b_p_l, 0.0, atol=1e-5)):
        ##if not (np.isclose(a_p_l, 0.0) or np.isclose(b_p_l, 0.0)):
        #    # intersection is not coincidental with one of the extremities
        #    a_p /= a_p_l
        #    b_p /= b_p_l
        #    if v1.dot(a_p) < 0 or v1.dot(b_p) > 0:
        #        return False, None, False
        ##print("in first seg")
        #a_p = line_inters[1] - seg_2[0]
        #a_p_l = np.linalg.norm(a_p)
        #b_p = line_inters[1] - seg_2[-1]
        #b_p_l = np.linalg.norm(b_p)
        ##print(a_p_l, b_p_l)
        ##if not(np.isclose(a_p_l, 0.0, atol=1e-5) or np.isclose(b_p_l, 0.0, atol=1e-5)):
        #if not(math.isclose(a_p_l, 0.0, abs_tol=1e-5) or math.isclose(b_p_l, 0.0, abs_tol=1e-5)):
        #    a_p /= a_p_l
        #    b_p /= b_p_l
        #    #print(v2.dot(a_p), v2.dot(b_p))
        #    if v2.dot(a_p) < 0 or v2.dot(b_p) > 0:
        #        return False, None, False
        #print("in snd seg")
        # all good
        return True, line_inters[0], False

    return False, None, False

def get_planar_convex_hull(points):
    # construct plane
    p = Plane.best_fit(points)
    vec_0 = np.array([1, 0, 0])
    if np.isclose(np.abs(np.dot(vec_0, p.normal)), 1.0):
        vec_0 = np.array([0, 1, 0])
    vec_1 = np.cross(vec_0, p.normal)
    vec_1 /= np.linalg.norm(vec_1)
    # get planar coordinates
    plane_coords_0 = np.dot(vec_0, points.T)
    plane_coords_1 = np.dot(vec_1, points.T)
    pts_2d = np.array([[plane_coords_0[i], plane_coords_1[i]] for i in range(len(points))])
    # use shapely
    cvx_2d = np.array(MultiPoint(pts_2d).convex_hull.exterior.coords)
    # reconstruct 3d cvx hull
    cvx_3d = cvx_2d[:, 0].reshape(-1, 1)*vec_0 + cvx_2d[:, 1].reshape(-1, 1)*vec_1
    return cvx_3d

def chamfer_distance(x, y, metric='l2', direction='bi', return_pointwise_distances=False):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
        if return_pointwise_distances:
            return chamfer_dist, min_y_to_x
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
        if return_pointwise_distances:
            return chamfer_dist, min_x_to_y
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        if return_pointwise_distances:
            return chamfer_dist, min_x_to_y, min_y_to_x
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def polyline_polyline_intersection(poly_1, poly_2, VERBOSE=False):
    intersections = []
    tangents = []
    if VERBOSE:
        ps.init()
        ps.register_curve_network("poly1", poly_1,
                                  np.array([[i, i+1] for i in range(len(poly_1)-1)]))
        ps.register_curve_network("poly2", poly_2,
                                  np.array([[i, i+1] for i in range(len(poly_2)-1)]))
    for seg_1_id in range(len(poly_1)-1):
        seg_1 = np.array([poly_1[seg_1_id], poly_1[seg_1_id+1]])
        for seg_2_id in range(len(poly_2)-1):
            seg_2 = np.array([poly_2[seg_2_id], poly_2[seg_2_id + 1]])
            #print(seg_1, seg_2)
            intersected, intersection, tangent = segment_segment_intersection(seg_1, seg_2)
            #print(intersected, intersection)
            #print(intersected)
            if intersected:
                if intersection.shape == (3,):
                    intersections.append(intersection)
                    tangents.append(tangent)
                else:
                    intersections.append(intersection[0])
                    intersections.append(intersection[1])
                    tangents.append(tangent)
                    tangents.append(tangent)
                if VERBOSE:
                    print(seg_1_id, seg_2_id)
                    print(intersection.shape)
                    print(intersections)
                    ps.register_point_cloud("current intersection", np.array([intersection]))
    if VERBOSE:
        ps.show()
    return len(intersections) > 0, intersections, tangents

def print_pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def in_plane(origin, normal, pt, eps=None):
    return True

def unify_same_face_edges(edges, surface_ids):
    new_curves = []
    regrouped_edges = {}
    for i, id in enumerate(surface_ids):
        if id in regrouped_edges.keys():
            regrouped_edges[id].append(edges[i])
        else:
            regrouped_edges[id] = [edges[i]]
    print(regrouped_edges)
    for key in regrouped_edges.keys():
        print("KEY", key)
        key_edges = regrouped_edges[key]
        known_vertices = []
        vertex_ids = []
        for e in key_edges:
            e_vertex_ids = []
            for j in range(2):
                found_vertex = False
                for k_v_id, k_v in enumerate(known_vertices):
                    if np.all(np.isclose(e[j]-k_v, 0.0)):
                        e_vertex_ids.append(k_v_id)
                        found_vertex = True
                        break
                if not found_vertex:
                    e_vertex_ids.append(len(known_vertices))
                    known_vertices.append(e[j])
            vertex_ids.append(e_vertex_ids)
        if len(known_vertices) == 2:
            new_curves.append([[known_vertices[0], known_vertices[1]]])
            continue
        dists = squareform(pdist(known_vertices))
        G = nx.Graph()
        print(dists)
        for i in range(len(known_vertices)):
            nearest_neighbors = np.argsort(dists[i])
            print(nearest_neighbors)
            G.add_edge(i, nearest_neighbors[1])
            G.add_edge(i, nearest_neighbors[2])
        print(G.edges)
        exit()
    return new_curves

def unify_same_face_edges_2(edges, surface_ids):
    new_curves = []
    regrouped_edges = {}
    for i, id in enumerate(surface_ids):
        if id in regrouped_edges.keys():
            regrouped_edges[id].append(edges[i])
        else:
            regrouped_edges[id] = [edges[i]]
    #print(regrouped_edges)
    for key in regrouped_edges.keys():
        #print("KEY", key)
        key_edges = regrouped_edges[key]
        known_vertices = []
        vertex_ids = []
        G = nx.Graph()
        for e in key_edges:
            e_vertex_ids = []
            for j in range(2):
                found_vertex = False
                for k_v_id, k_v in enumerate(known_vertices):
                    if np.all(np.isclose(e[j]-k_v, 0.0)):
                        e_vertex_ids.append(k_v_id)
                        found_vertex = True
                        break
                if not found_vertex:
                    e_vertex_ids.append(len(known_vertices))
                    known_vertices.append(e[j])
            G.add_edge(e_vertex_ids[0], e_vertex_ids[1])
            vertex_ids.append(e_vertex_ids)

        #print(G.nodes)
        #print(G.edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        curves = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        #print(curves)
        for curve in curves:
            if nx.is_empty(curve):
                continue
            extremities = []
            #print(curve.edges)
            neighbor_lengths = []
            for node in curve.nodes():
                #print(node, len(list(nx.neighbors(curve, node))))
                neighbor_lengths.append(len(list(nx.neighbors(curve, node))))
                if len(list(nx.neighbors(curve, node))) == 1:
                    extremities.append(node)
            if len(extremities) == 1:
                print("Only one extremity!")
                continue
                #ps.init()
                #plot_curves(key_edges)
                #for v_id, v in enumerate(known_vertices):
                #    ps.register_point_cloud(str(v_id), np.array([v]))
                #ps.show()
            #print(extremities)
            if np.all(np.array(neighbor_lengths) == 2):
                # it's a circular path
                path = nx.find_cycle(curve)
                path = [p[0] for p in path]
                path.append(path[0])
            else:
                path = nx.shortest_path(curve, extremities[0], extremities[1])
            #print(path)
            new_curve = [known_vertices[path_i] for path_i in path]
            new_curves.append(new_curve)
    return new_curves

def in_faces_2(face_meshes, edge, plane_normal):
    #TODO: make this more efficient
    in_betweeners = [edge[0], edge[1]]
    #for i in range(len(edge) - 1):
    #    p_0 = edge[i]
    #    p_1 = edge[i + 1]
    #    vec = p_1 - p_0
    #    for t in np.linspace(0, 1, 5):
    #        in_betweeners.append(p_0 + vec * t)
    for face_id in face_meshes.keys():
        # TODO: discard faces which are planes and parallel to the current slice
        mesh = face_meshes[face_id]
        plane = Plane.best_fit(mesh.vertices)
        plane_dists = np.array([plane.distance_point(p) for p in mesh.vertices])
        if np.all(np.isclose(plane_dists, 0.0, atol=1e-4)) and np.isclose(np.abs(np.dot(plane.normal, plane_normal)), 1.0, atol=1e-4):
        #if (not np.all(np.isclose(plane_dists, 0.0, atol=1e-4))) or np.isclose(np.abs(np.dot(plane.normal, plane_normal)), 1.0, atol=1e-4):
            continue
        #dists, _, _ = pymesh.distance_to_mesh(mesh, in_betweeners)
        _, dists, _ = closest_point(mesh, in_betweeners)
        #in_face = np.all(np.isclose(dists, 0.0, atol=1e-5))
        in_face = np.all(np.isclose(dists, 0.0))
        if in_face:
            return True, face_id
    return False, -1
#def in_faces(face_meshes, edge):
#    in_betweeners = []
#    for i in range(len(edge) - 1):
#        p_0 = edge[i]
#        p_1 = edge[i + 1]
#        vec = p_1 - p_0
#        for t in np.linspace(0, 1, 5):
#            in_betweeners.append(p_0 + vec * t)
#    for face_id in face_meshes.keys():
#        # TODO: discard faces which are planes and parallel to the current slice
#        mesh = face_meshes[face_id]
#        dists, _, _ = pymesh.distance_to_mesh(mesh, in_betweeners)
#        in_face = np.all(np.isclose(dists, 0.0, atol=1e-5))
#        if in_face:
#            return True, face_id
#    return False, -1

def slice_mesh_2(mesh, direction, N, plane_origin=None):
    tmp_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    lines = mesh_plane(tmp_mesh, direction, plane_origin)
    #print(lines)
    #ps.show()
    return lines
    #exit()


#def slice_mesh(mesh, direction, N, plane_origin=None):
#    """ Slice a given 3D mesh N times along certain direciton.
#    Args:
#        mesh (:class:`Mesh`): The mesh to be sliced.
#        direction (:class:`numpy.ndaray`): Direction orthogonal to the slices.
#        N (int): Number of slices.
#    Returns:
#        A list of `N` :class:`Mesh` objects, each representing a single slice.
#    """
#    if mesh.dim != 3:
#        raise NotImplementedError("Only slicing 3D mesh is supported.")
#
#    print("plane_origin", plane_origin)
#    print("direction", direction)
#    bbox_min, bbox_max = mesh.bbox
#    center = 0.5 * (bbox_min + bbox_max)
#    radius = norm(bbox_max - center)
#    direction = np.array(direction)
#    direction = direction / norm(direction)
#
#    proj_len = np.dot(mesh.vertices, direction)
#    #if plane_origin is not None:
#    #    print("OK")
#    #    proj_len = np.array([np.dot(plane_origin, direction)])
#    #    print(proj_len)
#    min_val = np.amin(proj_len)
#    max_val = np.amax(proj_len)
#    mid_val = 0.5 * (min_val + max_val)
#    intercepts = np.linspace(min_val - mid_val, max_val - mid_val, N+2)[1:-1]
#    intercepts = np.array([np.dot(plane_origin, direction)])
#    print("intercepts", intercepts)
#    assert(len(intercepts) == N)
#    if N%2 == 1:
#        intercepts = np.append(intercepts, intercepts[-1]+radius)
#    print("intercepts", intercepts)
#
#    #exit()
#    #intercepts = np.array([np.dot(plane_origin, direction)])
#    boxes = []
#    for low, high in intercepts.reshape((-1, 2), order="C"):
#        min_corner = -np.ones(3) * (radius+1)
#        max_corner =  np.ones(3) * (radius+1)
#        min_corner[2] = low
#        max_corner[2] = high
#        box = pymesh.generate_box_mesh(min_corner, max_corner)
#        boxes.append(box)
#
#    num_boxes = len(boxes)
#    boxes = pymesh.merge_meshes(boxes)
#    # TODO: why???
#    rot = pymesh.Quaternion.fromData(
#            np.array([0.0, 0.0, 1.0]), np.array(direction)).to_matrix()
#    boxes = pymesh.form_mesh(np.dot(rot, boxes.vertices.T).T + center, boxes.faces)
#
#    return [boxes]
#    slabs = pymesh.boolean(boxes, mesh, "intersection")
#    print(slabs)
#    if len(slabs.vertices) == 0:
#        return []
#
#    print(len(slabs.vertices))
#    cross_secs = []
#    source = slabs.get_attribute("source").ravel()
#    selected = source == 1
#    cross_section_faces = slabs.faces[selected]
#    cross_section = pymesh.form_mesh(slabs.vertices, cross_section_faces)
#
#    intersects = np.dot(slabs.vertices, direction).ravel() - \
#            np.dot(center, direction)
#    eps = (max_val - min_val) / (2 * N)
#
#    for i,val in enumerate(intercepts[:N]):
#        selected_vertices = np.logical_and(
#                intersects > val - eps,
#                intersects < val + eps)
#        selected_faces = np.all(selected_vertices[cross_section_faces], axis=1).ravel()
#        faces = cross_section_faces[selected_faces]
#        if i%2 == 0:
#            faces = faces[:,[0, 2, 1]]
#        m = pymesh.form_mesh(slabs.vertices, faces)
#        m = pymesh.remove_isolated_vertices(m)[0]
#        cross_secs.append(m)
#
#    #print(len(cross_secs[0].vertices))
#    #exit()
#    return cross_secs


def plot_curves(curves, name_prefix="", color=(0, 0, 1), radius=0.005, enabled=True, type_ids=None, type_colors=None):
    for curve_id, curve_geom in enumerate(curves):
        if len(curve_geom) == 1:
            edges_array = np.array([[0, 0]])
        else:
            edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
        edge_color = color
        if type_ids is not None:
            edge_color = type_colors[type_ids[curve_id]]
        ps.init()
        ps.register_curve_network(name_prefix + "_" + str(curve_id), nodes=np.array(curve_geom),
                                  edges=edges_array, color=edge_color, radius=radius, enabled=enabled)

def xyz_list2dict(l):
    return OrderedDict({'x':l[0], 'y':l[1], 'z':l[2]})

def angle_from_vector_to_x(vec):
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle

#def fix_mesh(mesh, detail="normal"):
#    bbox_min, bbox_max = mesh.bbox
#    diag_len = norm(bbox_max - bbox_min)
#    if detail == "normal":
#        target_len = diag_len * 5e-3
#    elif detail == "high":
#        target_len = diag_len * 2.5e-3
#    elif detail == "low":
#        target_len = diag_len * 1e-2
#    print("Target resolution: {} mm".format(target_len))
#
#    count = 0
#    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
#    mesh, __ = pymesh.split_long_edges(mesh, target_len)
#    num_vertices = mesh.num_vertices
#    while True:
#        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
#        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
#                                               preserve_feature=True)
#        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
#        if mesh.num_vertices == num_vertices:
#            break
#
#        num_vertices = mesh.num_vertices
#        print("#v: {}".format(num_vertices))
#        count += 1
#        if count > 10: break
#
#    mesh = pymesh.resolve_self_intersection(mesh)
#    mesh, __ = pymesh.remove_duplicated_faces(mesh)
#    mesh = pymesh.compute_outer_hull(mesh)
#    mesh, __ = pymesh.remove_duplicated_faces(mesh)
#    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
#    mesh, __ = pymesh.remove_isolated_vertices(mesh)
#
#    return mesh

def get_boundary_edges(faces):
    edge_counts = {}
    for f in faces:
        for i in range(3):
            edge = (min(f[i], f[(i+1)%3]), max(f[i], f[(i+1)%3]))
            if edge in edge_counts.keys():
                edge_counts[edge] += 1
            else:
                edge_counts[edge] = 1
    boundary_edges = []
    for edge in edge_counts.keys():
        if edge_counts[edge] == 1:
            boundary_edges.append(edge)
    #print(faces)
    #print(edge_counts)
    #print(boundary_edges)
    #exit()
    return boundary_edges

# using sympy
def curve_is_absorbed_2(curr_curve, prev_curves):
    # check if all of the points of the curr_curve are withing the prev_curves
    curr_points = [sympy.Point3D(p) for p in curr_curve]
    print(curr_points)
    for prev_curve in prev_curves:
        prev_segments = [sympy.Segment(sympy.Point3D(prev_curve["geometry"][i]), sympy.Point3D(prev_curve["geometry"][i+1])) for i in range(len(prev_curve["geometry"])-1)]
        contained_points = 0
        for curr_pt in curr_points:
            for prev_seg in prev_segments:
                if prev_seg.contains(curr_pt):
                    contained_points += 1
        if contained_points == len(curr_points):
            print("is close")
            #ps.init()
            #ps.remove_all_structures()
            #ps.register_curve_network("curr_curve", np.array(curr_curve), np.array([[i, i+1] for i in range(len(curr_curve)-1)]))
            #ps.register_curve_network("prev_curve", np.array(prev_curve["geometry"]), np.array([[i, i+1] for i in range(len(prev_curve["geometry"])-1)]))
            #ps.show()
            return True
    return False

#def curve_is_absorbed(curr_curve, prev_curves):
#    # we have to scale everything up because of pymesh limitations
#    scale_factor = 100.0
#    is_absorbed = False
#    curr_wire = pymesh.form_wires(vertices=scale_factor*np.array(curr_curve),
#                                  edges=np.array([[i, i+1] for i in range(len(curr_curve)-1)]))
#    min_edge_length = np.min(curr_wire.wire_lengths)
#    if np.isclose(min_edge_length, 0.0):
#        print(np.array(curr_wire.vertices))
#    prev_wires = [pymesh.form_wires(vertices=scale_factor*np.array(prev_curve["geometry"]),
#                                      edges=np.array([[i, i + 1] for i in range(len(prev_curve["geometry"]) - 1)]))
#                  for prev_curve in prev_curves]
#    for prev_wire in prev_wires:
#        min_edge_length = np.minimum(min_edge_length, np.min(prev_wire.wire_lengths))
#        if np.isclose(min_edge_length, 0.0):
#            print(np.array(prev_wire.vertices))
#    print(min_edge_length)
#    inflation_thickness = min_edge_length/2
#    print(inflation_thickness)
#
#    curr_inflator = pymesh.wires.Inflator(curr_wire)
#    curr_inflator.inflate(inflation_thickness, per_vertex_thickness=True)
#    curr_mesh = curr_inflator.mesh
#    for prev_wire in prev_wires:
#        print(prev_wire)
#        prev_inflator = pymesh.wires.Inflator(prev_wire)
#        prev_inflator.inflate(inflation_thickness, per_vertex_thickness=True)
#        print("before mesh")
#        prev_mesh = prev_inflator.mesh
#        inter = pymesh.boolean(curr_mesh, prev_mesh, operation="intersection")
#        print("after inter")
#        print(inter.vertices)
#        if len(inter.vertices) == 0:
#            continue
#        curr_dists, _, _ = pymesh.distance_to_mesh(inter, curr_mesh.vertices)
#        print(curr_dists)
#        print(np.sum(curr_dists))
#        if np.isclose(np.sum(curr_dists), 0.0):
#            print("is close")
#            ps.init()
#            ps.remove_all_structures()
#            ps.register_surface_mesh("curr_mesh", curr_mesh.vertices, curr_mesh.faces)
#            ps.register_surface_mesh("prev_mesh", prev_mesh.vertices, prev_mesh.faces)
#            ps.register_surface_mesh("inter_mesh", inter.vertices, inter.faces)
#            ps.show()
#            return True
#        #prev_dists, _, _ = pymesh.distance_to_mesh(curr_mesh, inter.vertices)
#    return False

def remove_zero_length_edges(curve):
    curve = np.array(curve)
    new_curve = [curve[0]]
    for i in range(1, len(curve)):
        #print(i)
        #print(new_curve)
        #print(curve)
        if not np.isclose(np.linalg.norm(curve[i] - curve[i-1]), 0.0):
            new_curve.append(curve[i])
            #if i == len(curve)-2:
            #    new_curve.append(curve[i+1])
    if len(new_curve) == 1:
        return []

    return np.array(new_curve)

def polyline_pt_intersection(polyline, pt, line_dir):
    line_3d = sympy.Line3D(sympy.Point3D(pt), sympy.Point3D(pt+line_dir))
    #print(pt, line_dir)
    #print(polyline)
    for i in range(len(polyline)-1):
        if np.isclose(np.linalg.norm(np.array(polyline[i+1])-np.array(polyline[i])), 0.0):
            continue
        #print(polyline[i], polyline[i+1], np.array(polyline[i+1])-np.array(polyline[i]))
        try:
            seg = sympy.Segment3D(sympy.Point3D(polyline[i]), sympy.Point3D(polyline[i+1]))
            inter = seg.intersection(line_3d)
        except:
            continue
        print(inter)
        if len(inter) > 0:
            return True, np.array(inter)
        #print(inter)
    #exit()
    return False, np.zeros(3)
    #for seg in polyline:

def line_segment_from_points(points):
    new_points = np.array(points).reshape(-1, 3)
    if skspatial.objects.Points(new_points).are_concurrent():
        return [], False
    l = Line.best_fit(new_points)
    dots = np.dot(l.direction, new_points.T)
    is_good_line_fit = np.all([np.isclose(np.linalg.norm(l.project_point(p)-p), 0.0) for p in new_points])
    return np.array([new_points[np.argmin(dots)], new_points[np.argmax(dots)]]), is_good_line_fit

def intersect_lines(l1, l2):
    # Vector from line A to line B.
    vector_ab = skspatial.objects.Vector.from_points(l1.point, l2.point)

    # Vector perpendicular to both lines.
    vector_perpendicular = l1.direction.cross(l2.direction)

    num = vector_ab.cross(l2.direction).dot(vector_perpendicular)
    denom = vector_perpendicular.norm() ** 2

    # Vector along line A to the intersection point.
    vector_a_scaled = num / denom * l1.direction

    return l1.point + vector_a_scaled

def spherical_to_cartesian_coords(r, theta, phi):
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return np.array([x, y, z])
    #return np.array([x, z, y])

def polyline_length(l):
    l = np.array(l)
    return np.sum([np.linalg.norm(l[p_id]-l[p_id+1]) for p_id in range(len(l)-1)])

def icp_registration(l1, l2, with_scaling=False):
    initial_T = np.identity(4) # Initial transformation for ICP
    source = np.zeros([len(l1), 3])
    source[:, :2] = l1
    target = np.zeros([len(l2), 3])
    target[:, :2] = l2
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target)

    distance = 10000.0 # The threshold distance used for searching correspondences
    icp_type = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    iterations = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 1000000)
    result = o3d.pipelines.registration.registration_icp(source_pc, target_pc, distance, initial_T, icp_type, iterations)
    #print(result)
    trans = np.asarray(deepcopy(result.transformation))
    #if trans[0, 0]*trans[1, 1] < 0:
    #    trans[:, 0] *= -1
    source_pc.transform(trans)
    scale_x = np.linalg.norm(trans[:, 0])
    scale_y = np.linalg.norm(trans[:, 1])
    scale_z = np.linalg.norm(trans[:, 2])
    scale_z = trans[2, 2]
    reflection = scale_z < 0
    rot_mat = deepcopy(trans)
    rot_mat[:, 0] /= scale_z
    rot_mat[:, 1] /= scale_z
    rot_mat[:, 2] /= scale_z
    angle = np.rad2deg(np.arccos(rot_mat[0, 0]))
    return np.asarray(source_pc.points)[:, :2], trans, angle

    #rot_angle = 270*int(reflection) + angle
    #reject = True
    #if abs(rot_angle) < 20.0 or abs(abs(rot_angle) - 360) < 20.0 or abs(abs(rot_angle) - 180.0) < 20.0:
    #    reject = False
    #if abs(scale_z) < 0.2:
    #    reject = True

def apply_icp_transformation(l, transformation):
    source = np.zeros([len(l), 3])
    source[:, :2] = l
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)
    source_pc.transform(transformation)
    return np.asarray(source_pc.points)[:, :2]

def generate_silhouette_lines(curved_surfaces_per_feature_id, syn_draw_path,
                              cam_pos, obj_center, up_vec,
                              theta=60, phi=45,
                              VERBOSE=False):
    if VERBOSE:
        ps.init()
        ps.remove_all_structures()
    silhouette_lines_per_feature_id = {}
    for feature_id in curved_surfaces_per_feature_id.keys():
        for surface_id in curved_surfaces_per_feature_id[feature_id].keys():
            rnd_name = "".join(np.random.randint(0, 9, 4).astype(str))
            if VERBOSE:
                print("surface_id", surface_id)
            vertices = []
            faces_ids = []
            for tri in curved_surfaces_per_feature_id[feature_id][surface_id]:
                faces_ids += [[len(vertices), len(vertices)+1, len(vertices)+2]]
                vertices += tri
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces_ids)
            obj_file_content = export_obj(mesh)
            mesh_file_path = os.path.join(syn_draw_path, "curved_surface_mesh"+rnd_name+".obj")
            with open(mesh_file_path, "w") as fp:
                fp.write(obj_file_content)
            #with open(os.path.join(syn_draw_path, "../cam_smooth.properties"), "r") as fp:
            with open(os.path.join("./cam_smooth.properties"), "r") as fp:
                cam_properties = fp.readlines()
            new_cam_properties = ""
            for l in cam_properties:
                new_line = deepcopy(l)
                #if "chain_angle" in l:
                #    #new_line = "svg_out="+surface_id+".svg\n"
                #    new_line = "chain_angle=360\n"
                #if "intersections_tol" in l:
                #    #new_line = "svg_out="+surface_id+".svg\n"
                #    new_line = "intersections_tol=1.0e\n"
                #if "raycast_tol" in l:
                #    #new_line = "svg_out="+surface_id+".svg\n"
                #    new_line = "raycast_tol=1.0e-1\n"
                if "svg_out" in l:
                    #new_line = "svg_out="+surface_id+".svg\n"
                    #new_line = "svg_out=out.svg\n"
                    new_line = "svg_out="+rnd_name+".svg\n"
                if "mesh_in" in l:
                    new_line = "mesh_in="+os.path.abspath(mesh_file_path)+"\n"
                if "camera_target" in l:
                    new_line = "camera_target="+str(obj_center[0])+", "+str(obj_center[1])+", "+str(obj_center[2])+"\n"
                if "camera_position" in l:
                    new_cam_pos = np.array(cam_pos)+np.array(obj_center)
                    #new_cam_pos = np.array(cam_pos)
                    new_line = "camera_position="+str(new_cam_pos[0])+", "+str(new_cam_pos[1])+", "+str(new_cam_pos[2])+"\n"
                if "camera_up" in l:
                    new_line = "camera_up="+str(up_vec[0])+", "+str(up_vec[1])+", "+str(up_vec[2])+"\n"

                new_cam_properties += new_line
            #print(new_cam_properties)
            # generate a random file name, so that this step can be exectuted in parallel by multiple jobs
            #print(rnd_name)
            new_cam_file_path = os.path.abspath(os.path.join(syn_draw_path, rnd_name+".properties"))
            with open(new_cam_file_path, "w") as fp:
                fp.write(new_cam_properties)

            #os.system(os.path.join(os.path.abspath(syn_draw_path), "./SynDraw -p "+str(new_cam_file_path)))
            #os.system("rm -f "+os.path.join(os.path.abspath(syn_draw_path), "out.svg"))
            while os.system("mkdir ./syn_draw.lock >> /dev/null 2>&1") !=0:
                print("waiting 1")
                #exit()
            #os.system(os.path.join(os.path.abspath(syn_draw_path), "./SynDraw -p "+str(new_cam_file_path))+" > /dev/null")
            try:
                while os.system(os.path.join(os.path.abspath(syn_draw_path), "./SynDraw -p "+str(new_cam_file_path))+" > /dev/null") != 0:
                    print("waiting for syn_draw")
                #ret_val = os.system(os.path.join(os.path.abspath(syn_draw_path), "./SynDraw -p "+str(new_cam_file_path)))
                #print("SYN_DRAW ret_val", ret_val)
                os.remove(new_cam_file_path)
                os.system("rm -r ./syn_draw.lock")
            except:
                os.system("rm -r ./syn_draw.lock")
                exit()
            #paths, attributes = svg2paths(os.path.join(os.path.abspath(syn_draw_path), "out.svg"))
            #paths, attributes = svg2paths("out.svg")
            paths, attributes = svg2paths(rnd_name+".svg")
            os.remove(rnd_name+".svg")
            os.remove(mesh_file_path)
            silhouette_lines = []
            for path in attributes:
                if not "points3d" in path.keys():
                    continue
                points = path["points3d"].split(" ")
                l = []
                #print(points)
                for p in points:
                    if len(p) > 3:
                        l.append(np.array(p.split(","), dtype=float))
                        #print(p)
                        #print(l[-1])
                silhouette_line = np.array(l)
                if np.isclose(np.linalg.norm(silhouette_line[0] - silhouette_line[-1]), 0.0):
                    continue
                silhouette_lines.append(silhouette_line)
            if len(silhouette_lines) == 0:
                continue

            if not str(feature_id) in silhouette_lines_per_feature_id.keys():
                silhouette_lines_per_feature_id[str(feature_id)] = {}
            if not surface_id in silhouette_lines_per_feature_id[str(feature_id)].keys():
                silhouette_lines_per_feature_id[str(feature_id)][surface_id] = []
            for l in silhouette_lines:
                silhouette_lines_per_feature_id[str(feature_id)][surface_id].append(l.tolist())

            #print(silhouette_lines)
            #for i, sil in enumerate(silhouette_lines):
            #    ps.register_curve_network(str(i), sil, np.array([[i, i+1] for i in range(len(sil)-1)]))
            if VERBOSE:
                #ps.remove_all_structures()
                for l_id, l in enumerate(silhouette_lines):
                    ps.register_curve_network(str(feature_id)+"_"+str(l_id), l, np.array([[i, i+1] for i in range(len(l)-1)]))
                ps.register_surface_mesh(surface_id, np.array(vertices), np.array(faces_ids))
    if VERBOSE:
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("neg_z_up")
        ps.set_navigation_style("free")
        ps.show()
    return silhouette_lines_per_feature_id

def get_curved_surfaces(data_folder):
    curved_surfaces_per_feature_id = {}
    for body_file in os.listdir(data_folder):
        if not "bodydetails" in body_file:
            continue
        feature_id = int(body_file.split(".json")[0].split("bodydetails")[1])
        #print("feature_id", feature_id)
        curved_surfaces_per_feature_id[feature_id] = {}
        with open(os.path.join(data_folder, body_file), "r") as fp:
            bodydetails = json.load(fp)
        # get triangle meshes
        with open(os.path.join(data_folder, "feature_faces_"+str(feature_id)+".json"), "r") as fp:
            feature_faces = json.load(fp)
        for body in bodydetails["bodies"]:
            for face in body["faces"]:
                if face["surface"]["type"] == "plane" or (face["surface"]["type"] == "other" and len(face["loops"]) == 0):
                    continue
                curved_surfaces_per_feature_id[feature_id][face["id"]] = []
        for surface_id in curved_surfaces_per_feature_id[feature_id].keys():
            curved_surfaces_per_feature_id[feature_id][surface_id] = feature_faces[surface_id]
    # trim unchanged surfaces
    already_checked = set()
    for feature_id in range(max(list(curved_surfaces_per_feature_id.keys()))+1):
        #print(feature_id)
        remove_surface_ids = []
        for surface_id in curved_surfaces_per_feature_id[feature_id].keys():
            if surface_id in already_checked:
                remove_surface_ids.append(surface_id)
            else:
                already_checked.add(surface_id)
        for del_id in remove_surface_ids:
            curved_surfaces_per_feature_id[feature_id].pop(del_id)
    #for feature_id in range(max(list(curved_surfaces_per_feature_id.keys()))+1):
    #    print(list(curved_surfaces_per_feature_id[feature_id].keys()))
    #exit()
    return curved_surfaces_per_feature_id

# feature_lines is a dict
def project_points(feature_lines, cam_pos, obj_center, up_vec, img_dims=[1000, 1000]):
    #points = np.array([p for l in feature_lines for p in l])
    #max = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
    #min = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
    #bbox_diag = np.linalg.norm(max - min)
    ##print("bbox:", min, max)
    #obj_center = (max+min)/2
    #cam_pos = spherical_to_cartesian_coords(radius*bbox_diag, np.deg2rad(theta), np.deg2rad(phi))
    #print(cam_pos)
    #print(cam_pos+obj_center)
    #print(obj_center)
    #exit()

    #up_vec = np.array([0, 0, 1])
    #up_vec = np.array([0.51092751686519045,0.6654680932233632,-0.54415557463985587])
    view_mat = pyrr.matrix44.create_look_at(cam_pos,
                                            np.array([0, 0, 0]),
                                            up_vec)
    near = 0.001
    far = 1.0
    view_edges = []
    total_view_points = []
    for f_line in feature_lines:
        view_points = []
        for p in f_line.copy():
            p -= obj_center
            hom_p = np.ones(4)
            hom_p[:3] = p
            proj_p = np.matmul(view_mat.T, hom_p)
            view_points.append(proj_p)
            total_view_points.append(proj_p)
        view_edges.append(np.array(view_points))
    #for f_line in view_edges:
    #    plt.plot(f_line[:, 0], f_line[:, 1], c="black")
    #plt.show()
    total_view_points = np.array(total_view_points)
    max = np.array([np.max(total_view_points[:, 0]), np.max(total_view_points[:, 1]), np.max(total_view_points[:, 2])])
    min = np.array([np.min(total_view_points[:, 0]), np.min(total_view_points[:, 1]), np.min(total_view_points[:, 2])])

    #proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=min[0], right=max[0], bottom=min[1], top=max[1],
    #                                                                          near=near, far=far)
    max_dim = np.maximum(np.abs(max[0]-min[0]), np.abs(max[1]-min[1]))
    proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=-max_dim/2, right=max_dim/2, bottom=-max_dim/2, top=max_dim/2,
                                                                              near=near, far=far)

    total_projected_points = []
    projected_edges = []
    for f_line in view_edges:
        projected_points = []
        for p in f_line.copy():
            proj_p = np.matmul(proj_mat, p)
            proj_p[:3] /= proj_p[-1]
            total_projected_points.append(proj_p[:2])
            projected_points.append(proj_p[:2])
        projected_edges.append(np.array(projected_points))
    total_projected_points = np.array(total_projected_points)

    # screen-space
    # scale to take up 80% of the image
    max = np.array([np.max(total_projected_points[:, 0]), np.max(total_projected_points[:, 1])])
    min = np.array([np.min(total_projected_points[:, 0]), np.min(total_projected_points[:, 1])])
    bbox_diag = np.linalg.norm(max - min)
    screen_diag = np.sqrt(img_dims[0]*img_dims[0]+img_dims[1]*img_dims[1])
    scaled_edges = []
    for f_line in projected_edges:
        scaled_points = []
        for p in f_line:
            p[1] *= -1
            p *= 0.5*screen_diag/bbox_diag
            #p *= 0.8*500/bbox_diag
            p += np.array([img_dims[0]/2, img_dims[1]/2])
            scaled_points.append(p)
        scaled_edges.append(np.array(scaled_points))

    return scaled_edges

def cad_seq_last_id(data_folder):
    max_num = 0
    for f in os.listdir(data_folder):
        if not ".obj" in f:
            continue
        max_num = max(max_num, int(f.split("shape_")[1].split(".obj")[0]))
    return max_num

def load_last_mesh(data_folder):
    max_num = 0
    for f in os.listdir(data_folder):
        if not ".obj" in f or not "shape_" in f:
            continue
        max_num = max(max_num, int(f.split("shape_")[1].split(".obj")[0]))
    with open(os.path.join(data_folder, "shape_"+str(max_num)+".obj")) as fp:
        mesh = trimesh.load(fp, file_type="obj")
    return mesh

def load_last_bodydetails(data_folder):
    max_num = 0
    for f in os.listdir(data_folder):
        if not ".obj" in f:
            continue
        max_num = max(max_num, int(f.split("shape_")[1].split(".obj")[0]))
    with open(os.path.join(data_folder, "bodydetails"+str(max_num)+".json")) as fp:
        bd = json.load(fp)
    return bd

def abc_sanity_check(data_folder):
    bd = load_last_bodydetails(data_folder)
    if len(bd["bodies"]) > 1:
        print("too many bodies")
        return False
    mesh = load_last_mesh(data_folder)
    bbox = bbox_from_points(mesh.vertices)
    print(bbox)
    min = bbox[:3]
    max = bbox[3:]
    print(min, max)
    w, d, h = np.abs(max - min)
    print(w, d, h)
    diag = np.linalg.norm(max - min)
    print("diag ratios", diag/w, diag/d, diag/h)
    if diag/w > 5.0 or diag/d > 5.0 or diag/h > 5.0:
        return False
    return True

def load_faces_i(data_folder, i):
    max_num = 0
    with open(os.path.join(data_folder, "feature_faces_"+str(i)+".json"), "r") as fp:
        patches = json.load(fp)
    return patches

def load_last_faces(data_folder):
    max_num = 0
    for f in os.listdir(data_folder):
        if not "feature_faces" in f or not ".json" in f:
            continue
        max_num = max(max_num, int(f.split("feature_faces_")[1].split(".json")[0]))
    with open(os.path.join(data_folder, "feature_faces_"+str(max_num)+".json"), "r") as fp:
        patches = json.load(fp)
    return patches


def get_cam_pos_obj_center(points, radius=1, theta=60, phi=45):
    #points = np.array([p for l in feature_lines for p in l])
    max = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
    min = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
    bbox_diag = np.linalg.norm(max - min)
    obj_center = (max+min)/2
    cam_pos = spherical_to_cartesian_coords(radius*bbox_diag, np.deg2rad(theta), np.deg2rad(phi))
    return cam_pos, obj_center

def line_3d_length(polyline):
    if len(polyline) < 2:
        return 0.0
    l = np.array(polyline)
    return np.sum(np.linalg.norm(l[1:] - l[:len(l)-1], axis=1))

# normalized: t in [0, 1]
def interpolate_polyline(poly, t):

    if np.isclose(t, 0.0):
        return poly[0]
    if np.isclose(t, 1.0):
        return poly[-1]
    total_length = line_3d_length(poly)

    acc_length = 0
    for i in range(len(poly)-1):
        seg = np.array([poly[i], poly[i+1]])
        seg_length = line_3d_length(seg)
        start_t = acc_length/total_length
        end_t = (acc_length+seg_length)/total_length
        if t >= start_t and t <= end_t:
            vec = seg[1] - seg[0]
            return seg[0] + vec*((t-start_t)/(end_t-start_t))
        acc_length += seg_length
    return None

def get_ellipse_fittings(all_edges, cam_pos, obj_center, up_vec):
    proj_lines = project_points([e["geometry"] for e in all_edges], cam_pos, obj_center, up_vec)
    ell = EllipseModel()
    ellipse_fittings = [[] for l in proj_lines]
    for l_id, l in enumerate(proj_lines):
        linestring = LineString(l)
        if len(list(linestring.buffer(1.0).interiors)) > 0:
            ell.estimate(l)

            xc, yc, a, b, theta_orig = ell.params
            theta = np.rad2deg(theta_orig)
            center = np.array([xc, yc])
            ellipse_fittings[l_id] = [center, a, b, theta]

    return ellipse_fittings

def compute_visibility_score(all_edges, cam_pos, mesh, obj_center=None, up_vec=None, VERBOSE=False):

    bbox = bbox_from_points(mesh.vertices)
    eps = 0.05*np.linalg.norm(bbox[3:] - bbox[:3])
    #print("begin visibility scores")
    visibility_scores = []
    raycaster = ray_triangle.RayMeshIntersector(mesh)
    all_distances = []
    orig_feature_ids = []
    for edge_id, edge in enumerate(all_edges):
        point_hit = []
        # detect partial visibility
        if not edge["original_feature_line"]:
            visibility_scores.append(0.0)
            all_distances.append(0.0)
            continue
        if not "fitted_curve" in list(edge.keys()):
            resampled_pts = utils_occ.sample_fitted_curve(
                utils_occ.fit_curve(edge["geometry"]))
        else:
            resampled_pts = utils_occ.sample_fitted_curve(
                edge["fitted_curve"])
        all_distances.append(np.mean([np.linalg.norm(np.array(p) - np.array(cam_pos))
                                      for p in resampled_pts]))
        orig_feature_ids.append(edge_id)
        if edge["type"] == "silhouette_line":
            visibility_scores.append(1.0)
            continue
        #for p_id, p in enumerate(edge["geometry"]):
        for p_id, p in enumerate(resampled_pts):
            ray_origin = cam_pos
            ray_direction = np.array(p) - np.array(cam_pos)
            point_cam_dist = np.linalg.norm(ray_direction)
            ray_direction /= point_cam_dist
            trimesh.constants.tol.zero = 1e-5
            hits, _, _ = raycaster.intersects_location(
                ray_origins=[ray_origin], ray_directions=[ray_direction], multiple_hits=True)
            trimesh.constants.tol.__init__()
            # filter hits
            filtered_hits = []
            for hit in hits:
                hit_dist = np.linalg.norm(np.array(cam_pos) - np.array(hit))
                if (not np.isclose(abs(hit_dist - point_cam_dist), 0.0, atol=eps)) and hit_dist < point_cam_dist:
                    filtered_hits.append(hit)
            point_hit.append(len(filtered_hits) > 0)
        visibility_scores.append(1.0-np.sum(point_hit)/len(resampled_pts))
        #if np.all(point_hit):
        #    visibility_scores.append(0.0)
        #else:
        #    visibility_scores.append(1.0)
    visibility_scores = np.array(visibility_scores)
    all_distances = np.array(all_distances)
    all_distances[orig_feature_ids] -= np.min(all_distances[orig_feature_ids])
    all_distances[orig_feature_ids] /= np.max(all_distances[orig_feature_ids])
    all_distances[orig_feature_ids] = 1 - all_distances[orig_feature_ids]
    #print(visibility_scores)

    projected_edges = project_points([c["geometry"] for c in all_edges],
                                     cam_pos, obj_center, up_vec)
    stroke_lengths = np.array([LineString(p).length for p in projected_edges])
    stroke_lengths[orig_feature_ids] -= np.min(stroke_lengths[orig_feature_ids])
    stroke_lengths[orig_feature_ids] /= np.max(stroke_lengths[orig_feature_ids])

    #print(len(stroke_lengths))
    shape_score = 1.0/3.0*(visibility_scores + all_distances + stroke_lengths)
    #shape_score[orig_feature_ids] -= np.min(shape_score[orig_feature_ids])
    #shape_score[orig_feature_ids] /= np.max(shape_score[orig_feature_ids])
    mask = np.ones(len(shape_score), dtype=np.bool_)
    mask[orig_feature_ids] = 0
    shape_score[mask] = 0.0
    #print(len(shape_score))
    #for i, s in enumerate(shape_score):
    #    print(i, s)

    if VERBOSE:
        cmap = sns.color_palette("viridis", as_cmap=True)
        for p_id, p in enumerate(projected_edges):
            #if not all_edges[p_id]["type"] in ["feature_line", "silhouette_line"]:
            #    visibility_scores.append(0.0)
            #    continue
            #if not all_edges[p_id]["original_feature_line"]:
            #    continue

            #plt.plot(np.array(p)[:, 0], np.array(p)[:, 1], c=cmap(visibility_scores[p_id]), lw=3)
            #plt.plot(np.array(p)[:, 0], np.array(p)[:, 1], c=cmap(all_distances[p_id]), lw=3)
            plt.plot(np.array(p)[:, 0], np.array(p)[:, 1], c=cmap(shape_score[p_id]), lw=3)
            #plt.plot(np.array(p)[:, 0], np.array(p)[:, 1], c=cmap(stroke_lengths[p_id]), lw=3)
        plt.gca().invert_yaxis()
        plt.gca().axis("equal")
        plt.show()
    #exit()

    return shape_score

def compute_occlusions(all_edges, cam_pos, obj_center, up_vec, strokes_dict, intersections,
                       VERBOSE=True):
    projected_edges = project_points([c["geometry"] for c in all_edges],
                                     cam_pos, obj_center, up_vec)
    intersections_3d = np.array(project_points([[inter] for inter in intersections],
                                      cam_pos, obj_center, up_vec)).reshape(-1, 2)
    linestrings = [LineString(l) for l in projected_edges]
    intersections_2d = []
    intersections_2d_stroke_ids =[]
    for l1_id, l1 in enumerate(linestrings):
        for l2_id, l2 in enumerate(linestrings):
            if l2_id >= l1_id:
                continue
            if l1.intersects(l2):
                inter = l1.intersection(l2)
                if type(inter) != Point:
                    continue
                
                point_coords = [inter.x, inter.y]
                intersections_2d.append(point_coords)

                # intersections_2d += np.array(inter).reshape(-1, 2).tolist()
                intersections_2d_stroke_ids.append([l1_id, l2_id])
                #if len(intersections_2d) != len(intersections_2d_stroke_ids):
                #    print("here")
                #    print(intersections_2d)
                #    print(intersections_2d_stroke_ids)
                #    print(np.array(inter).reshape(-1, 2).tolist())
                #    print(l1_id, l2_id)
                #    exit()
    intersections_2d = np.array(intersections_2d)
    #exit()
    dists = cdist(intersections_2d, intersections_3d, metric="euclidean")
    occlusion_ids = [i for i in range(len(intersections_2d))
                     if not np.any(dists[i, :] < 2.0)]
    per_stroke_occlusions = [[] for i in range(len(strokes_dict))]
    for i in occlusion_ids:
        s_id_max = max(intersections_2d_stroke_ids[i])
        s_id_min = min(intersections_2d_stroke_ids[i])
        per_stroke_occlusions[s_id_max].append(s_id_min)
    for i in range(len(per_stroke_occlusions)):
        strokes_dict[i]["occlusions"] = per_stroke_occlusions[i]
    if VERBOSE:
        for p_id, p in enumerate(projected_edges):
            plt.plot(np.array(p)[:, 0], np.array(p)[:, 1], c="black", lw=2)
        plt.scatter(intersections_2d[:, 0], intersections_2d[:, 1], c="red", s=50)
        plt.scatter(intersections_3d[:, 0], intersections_3d[:, 1], c="green", s=100)
        plt.scatter(intersections_2d[occlusion_ids, 0], intersections_2d[occlusion_ids, 1], c="blue", s=50)
        plt.gca().invert_yaxis()
        plt.gca().axis("equal")
        plt.show()

def intersection_dag(strokes_dict):
    # return intersection dag and reachable substrokes per strokes
    dag = nx.DiGraph()
    for s in strokes_dict:
        s_id = s["id"]
        for anchor_i in s["anchor_intersections"]:
            for prev_s_id in anchor_i:
                dag.add_edge(prev_s_id, s_id)
    per_stroke_reachable_strokes = {}
    for s in strokes_dict:
        s_id = s["id"]
        if s_id in dag.nodes:
            per_stroke_reachable_strokes[s_id] = list(nx.descendants(dag, s_id))
        else:
            per_stroke_reachable_strokes[s_id] = []
    return dag, per_stroke_reachable_strokes

def cut_non_visible_points(edge, cam_pos, mesh, obj_center=None, up_vec=None, VERBOSE=False):

    geom = edge["geometry"]
    raycaster = ray_triangle.RayMeshIntersector(mesh)
    resampled_pts = utils_occ.sample_fitted_curve(
        utils_occ.fit_curve(geom), N=30)
    visible_segments = []
    visible_points = []
    last_p_id = -1
    #if edge["type"] == "silhouette_line":
    #    return [np.array(edge["geometry"])]

    for p_id, p in enumerate(resampled_pts):
        ray_origin = cam_pos
        ray_direction = np.array(p) - np.array(cam_pos)
        point_cam_dist = np.linalg.norm(ray_direction)
        ray_direction /= point_cam_dist
        trimesh.constants.tol.zero = 1e-5
        hits, _, _ = raycaster.intersects_location(
            ray_origins=[ray_origin], ray_directions=[ray_direction], multiple_hits=True)
        trimesh.constants.tol.__init__()
        # filter hits
        filtered_hits = []
        for hit in hits:
            hit_dist = np.linalg.norm(np.array(cam_pos) - np.array(hit))
            if VERBOSE:
                print(p_id, hit_dist, point_cam_dist, np.isclose(abs(hit_dist - point_cam_dist), 0.0, atol=1e-4), hit_dist<point_cam_dist)
            if (not np.isclose(abs(hit_dist - point_cam_dist), 0.0, atol=1e-4)) and hit_dist < point_cam_dist:
                filtered_hits.append(hit)
        if VERBOSE:
            ps.register_point_cloud(str(p_id), np.array(hits))
        #if edge["type"] == "silhouette_line":
        #    print(len(filtered_hits))
        if len(filtered_hits) == 0 or (edge["type"] == "silhouette_line" and len(filtered_hits) <= 1):
            if last_p_id != p_id - 1:
                if len(visible_points) > 1:
                    visible_segments.append(np.array(visible_points))
                visible_points = []
            if VERBOSE:
                print(p_id)
            visible_points.append(p)
            last_p_id = p_id
    if len(visible_points) > 1:
        visible_segments.append(np.array(visible_points))
    #print(visible_segments)
    return visible_segments

def add_visibility_label(all_edges, cam_pos, mesh, obj_center=None, up_vec=None, VERBOSE=False):

    bbox = bbox_from_points(mesh.vertices)
    #eps = 0.1*np.linalg.norm(bbox[3:] - bbox[:3])
    eps = 0.05*np.linalg.norm(bbox[3:] - bbox[:3])

    raycaster = ray_triangle.RayMeshIntersector(mesh)
    for edge_id, edge in enumerate(all_edges):
        all_edges[edge_id]["visibility_score"] = 0.0
    for edge_id, edge in enumerate(all_edges):
        #if edge["type"] == "silhouette_line":
        #    all_edges[edge_id]["visibility_score"] = 1.0
        #    continue
        geom = edge["geometry"]
        resampled_pts = utils_occ.sample_fitted_curve(
            utils_occ.fit_curve(geom), N=30)
        visible_points = 0
        visible_pts_ids = []
        trimesh.constants.tol.zero = 1e-8
        trimesh.constants.tol.__init__()

        for p_id, p in enumerate(resampled_pts):
            ray_origin = cam_pos
            ray_direction = np.array(p) - np.array(cam_pos)
            point_cam_dist = np.linalg.norm(ray_direction)
            ray_direction /= point_cam_dist
            hits, _, _ = raycaster.intersects_location(
                ray_origins=[ray_origin], ray_directions=[ray_direction], multiple_hits=True)
            # filter hits
            filtered_hits = []
            #print("p_id", p_id)
            for hit in hits:
                hit_dist = np.linalg.norm(np.array(cam_pos) - np.array(hit))
                #print(hit_dist, point_cam_dist)
                if (not np.isclose(abs(hit_dist - point_cam_dist), 0.0, atol=eps)) and hit_dist < point_cam_dist:
                    filtered_hits.append(hit)
            if len(filtered_hits) == 0:
                visible_points += 1
                visible_pts_ids.append(p_id)
        #print(visible_points/len(resampled_pts))
        #if edge_id == 115:
        #    ps.init()
        #    ps.register_point_cloud("resampled_pts", resampled_pts)
        #    ps.register_surface_mesh("mesh", mesh.vertices, mesh.faces)
        #    ps.register_point_cloud("visible_pts", resampled_pts[visible_pts_ids])
        #    ps.show()
        all_edges[edge_id]["visibility_score"] = visible_points/len(resampled_pts)

def to_scap(sketch, scap_file_name, stroke_ids):
    pts = np.array([p.coords for s in sketch.strokes for p in s.points_list])
    scap = "#"+str(int(np.max(pts[:, 0]))) + " " + str(int(np.max(pts[:, 1])))+"\n"
    scap += "@1.5\n"
    stroke_counter = 0
    for s_id, s in enumerate(sketch.strokes):
        if not s_id in stroke_ids:
            continue
        scap += "{\n"
        #scap += "#"+str(s_id)+" "+str(s_id)+"\n"
        scap += "#"+str(stroke_counter)+" "+str(stroke_counter)+"\n"
        stroke_counter += 1
        for p in s.points_list:
            scap += str(round(p.coords[0],ndigits=3))+" "+str(round(p.coords[1],ndigits=3))+" 0\n"
        scap += "}\n"
    with open(scap_file_name, "w") as fp:
        fp.write(scap)

def load_scap(scap_file_name):
    with open(scap_file_name, "r") as fp:
        lines = fp.readlines()
    cluster_ids = set()
    stroke_counter = 0
    for line_id, line in enumerate(lines):
        if "{" in line:
            stroke_counter += 1
            cluster_ids.add(lines[line_id+1].split("\t")[-1])
    print(len(list(cluster_ids)))
    print(stroke_counter)
    return len(list(cluster_ids))

def lineseg_dist(p, a, b):

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])
    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c)), c

def bbox_from_points(points):
    # min_x, min_y, min_z, max_x, max_y, max_z
    points = np.array(points)
    bbox = np.zeros(6, dtype=float)
    bbox[:3] = 1000.0
    bbox[3:] = -1000.0
    bbox[0] = np.minimum(bbox[0], np.min(points[:, 0]))
    bbox[1] = np.minimum(bbox[1], np.min(points[:, 1]))
    bbox[2] = np.minimum(bbox[2], np.min(points[:, 2]))
    bbox[3] = np.maximum(bbox[3], np.max(points[:, 0]))
    bbox[4] = np.maximum(bbox[4], np.max(points[:, 1]))
    bbox[5] = np.maximum(bbox[5], np.max(points[:, 2]))
    return bbox

def rescale_rotate_center(sketch):
    points = np.array([p.coords for s in sketch.strokes for p in s.points_list])
    bbox = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])])
    print(bbox)
    center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    print(sketch.width)
    print(sketch.height)
    sketch_scale = np.linalg.norm([sketch.width, sketch.height])/np.linalg.norm(bbox[2:]-bbox[:2])
    for s in sketch.strokes:
        for p_id, p in enumerate(s.points_list):
            coords = np.array(p.coords)
            coords -= center
            coords[1] = -coords[1]
            coords *= 0.8*sketch_scale
            coords += np.array([sketch.width/2, sketch.height/2])
            s.points_list[p_id].coords = coords
    return sketch

if __name__ == "__main__":
    strokes_dict_file_name = os.path.join("data/student8_house/60_125_1.4/strokes_dict.json")
    with open(strokes_dict_file_name, "r") as fp:
        strokes_dict = json.load(fp)
    dag, per_stroke_reachable_strokes = intersection_dag(strokes_dict)
    print(per_stroke_reachable_strokes)
    #seg_1 = np.array([[0.0, 0.0, 0], [1.0, 0, 0]])
    #seg_2 = np.array([[0.5, -1, 0], [0.5, 1, 0]])
    #seg_2 = np.array([[0.0, 0.0, 0], [1.0, 0, 0]])
    #start_time = time()
    #segment_segment_intersection_optim(seg_1, seg_2)
    #print("solve_time", time()-start_time)
    #start_time = time()
    #segment_segment_intersection(seg_1, seg_2)
    #print("solve_time", time()-start_time)
