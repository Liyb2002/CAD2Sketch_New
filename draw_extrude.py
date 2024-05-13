import json
from copy import deepcopy
import utils_occ
import networkx as nx
import os
import polyscope as ps
import trimesh
from trimesh.ray import ray_triangle
from perspective_grid import PerspectiveGrid
from skspatial.objects import Line

from track_bodydetails import get_new_feature_lines, get_new_and_modified_feature_lines
from render_shapes import features_lines_to_svg, typed_feature_lines_to_svg
import circle_3d
import utils, sketch_utils, extrude_utils
import seaborn as sns
import numpy as np
import igl
#import pymesh
from skspatial.objects import Plane

mesh_color = (139.0 / 255.0, 139.0 / 255.0, 230.0 / 255.0)

def draw_extrude(data_folder, silhouette_line_per_feature_id, VERBOSE=False,
                 output_svg=False, theta=60, phi=45, only_final_npr_lines=False,
                 include_fillet_lines=False):

    all_edges = []

    if VERBOSE:
        ps.init()
    with open(os.path.join(data_folder, "parsed_features.json"), "r") as f:
        parsed_features = json.load(f)
    #print(parsed_features)
    color_map = sns.color_palette("Set2", len(parsed_features["sequence"]))
    with open(os.path.join(data_folder, "sketch_info.json"), "r") as fp:
        sketch_info = json.load(fp)

    bodydetails_pool = []
    feature_lines_pool = []
    feature_faces_pool = []
    feature_faces_meshes_pool = []
    sketch_pool = []
    perspective_grids = {}

    print(parsed_features["sequence"])
    for feat_id, feat in enumerate(parsed_features["sequence"]):
        print("feat_id", feat_id)
        #if feat_id == 2:
        #    return all_edges
        if only_final_npr_lines and feat_id < len(parsed_features["sequence"]) - 1:
            continue
        with open(os.path.join(data_folder, "bodydetails"+str(feat_id)+".json"), "r") as f:
            bodydetails = json.load(f)
        if len(bodydetails["bodies"]) > 0:
            bodydetails_pool.append(bodydetails["bodies"])
            bodydetails = bodydetails["bodies"]
        else:
            bodydetails_pool.append({})
        with open(os.path.join(data_folder, "feature_lines_"+str(feat_id)+".json"), "r") as f:
            feature_lines = json.load(f)
        feature_lines_pool.append(feature_lines)
        #if len(list(feature_lines.values())) > 0:
        #    print(feature_lines)
        #    ps.init()
        #    utils.plot_curves(list(feature_lines.values()))
        #    ps.show()
        with open(os.path.join(data_folder, "feature_faces_"+str(feat_id)+".json"), "r") as f:
            feature_faces = json.load(f)
        feature_faces_meshes = {}
        for face_id in feature_faces.keys():
            v = []
            f = []
            for facet in feature_faces[face_id]:
                v.append(facet[0])
                v.append(facet[1])
                v.append(facet[2])
                f.append([len(v)-3, len(v)-2, len(v)-1])
            #face_mesh = pymesh.form_mesh(np.array(v), np.array(f))
            face_mesh = trimesh.Trimesh(vertices=v, faces=f)
            feature_faces_meshes[face_id] = face_mesh
        feature_faces_pool.append(feature_faces)
        feature_faces_meshes_pool.append(feature_faces_meshes)

        print(feat["type"])
        if feat["type"] == "Sketch":
            with open(os.path.join(data_folder, "sketch_"+feat["entity"]+".json"), "r") as f:
                sketch = json.load(f)
            # filter out handle lines
            handle_curves = {}
            for tmp_sketch in sketch_info["sketches"]:
                if tmp_sketch["featureId"] == feat["entity"]:
                    for tmp_ent in tmp_sketch["geomEntities"]:
                        if "Handle" in tmp_ent["id"] and tmp_ent["entityType"] == "lineSegment":
                            handle_curves[tmp_ent["id"]] = [np.array(tmp_ent["startPoint"], dtype=np.float64),
                                                            np.array(tmp_ent["endPoint"], dtype=np.float64)]
            del_ids = []
            for c_id, c in enumerate(sketch["curves"]):
                if len(c) == 2:
                    for handle_curve in handle_curves.values():
                        if (np.isclose(np.linalg.norm(c[0]-handle_curve[0]), 0.0, atol=1e-4) and np.isclose(np.linalg.norm(c[1]-handle_curve[1]), 0.0, atol=1e-4)) or \
                                (np.isclose(np.linalg.norm(c[0]-handle_curve[1]), 0.0, atol=1e-4) and np.isclose(np.linalg.norm(c[1]-handle_curve[0]), 0.0, atol=1e-4)):
                            del_ids.append(c_id)
            for del_id in reversed(np.unique(del_ids)):
                del sketch["curves"][del_id]

            sketch_pool.append(sketch)
        else:
            if only_final_npr_lines or len(bodydetails_pool) == 1:
                new_edge_ids = [edge["id"] for body in bodydetails_pool[-1] for edge in body["edges"]]
            else:
                new_edge_ids, new_edge_body_ids = get_new_and_modified_feature_lines(bodydetails_pool[-2], bodydetails_pool[-1])
            curves_to_plot = [[feature_lines[edge_id], edge_id] for edge_id in new_edge_ids]
            # add silhouette lines after feature lines
            silhouette_lines = []
            #print(feat_id)
            #print(silhouette_line_per_feature_id.keys())
            if str(feat_id) in silhouette_line_per_feature_id.keys():
                silhouette_lines = [l for tmp_f in silhouette_line_per_feature_id[str(feat_id)].keys()
                                    for l in silhouette_line_per_feature_id[str(feat_id)][tmp_f]]
            # look in previous feat_ids if any silhouette line of currently available face_ids are present
            if only_final_npr_lines:
                for face_id in feature_faces.keys():
                    for prev_feat_id in silhouette_line_per_feature_id.keys():
                        if str(feat_id) == prev_feat_id:
                            continue
                        if face_id in silhouette_line_per_feature_id[prev_feat_id].keys():
                            silhouette_lines += silhouette_line_per_feature_id[prev_feat_id][face_id]

            #print("len(silhouette_lines)", len(silhouette_lines))
            #if feat_id == len(parsed_features["sequence"]) - 1:
            #    for curve in feature_lines.values():
            #        curve = utils.remove_zero_length_edges(curve)
            #        all_edges.append({"geometry": curve, "type": "outline"})
            #else:
            if (feat["type"] != "extrude" and feat["type"] != "fillet") or only_final_npr_lines or (feat["type"] == "fillet" and not include_fillet_lines):
                for curve in curves_to_plot:
                    new_curve = utils.remove_zero_length_edges(curve[0])
                    if len(np.array(new_curve).shape) > 1:
                        all_edges.append({"geometry": new_curve, "type": "feature_line",
                                          "feature_id": feat_id,
                                          "fitted_curve": utils_occ.fit_curve(new_curve)})
                for curve in silhouette_lines:
                    new_curve = utils.remove_zero_length_edges(curve)
                    if len(np.array(new_curve).shape) > 1:
                        all_edges.append({"geometry": new_curve, "type": "silhouette_line",
                                          "feature_id": feat_id,
                                          "fitted_curve": utils_occ.fit_curve(new_curve)})
            if VERBOSE:
                if feat["type"] != "extrude":
                    utils.plot_curves(curves_to_plot, name_prefix=str(feat_id), color=color_map[feat_id])
                # plot mesh
                obj_mesh_file = os.path.join(data_folder, "shape_"+str(feat_id)+".obj")
                if os.path.exists(obj_mesh_file):
                    v, f = igl.read_triangle_mesh(obj_mesh_file)
                    #ps.register_surface_mesh("surface", vertices=v, faces=f, color=mesh_color)

        if feat["type"] == "Sketch" and not only_final_npr_lines:
            ent = parsed_features["entities"][feat["entity"]]
            persp_grid = PerspectiveGrid()
            persp_grid.parse_sketch(ent, sketch["curves"])
            if VERBOSE:
                persp_grid.plot_grid()
            perspective_grids[feat["entity"]] = persp_grid

            grid_lines = persp_grid.get_grid_lines()
            # intersect with previous grids
            sketch_anchor_lines = sketch_utils.grid_intersections(perspective_grids,
                                                                  feat["entity"],
                                                                  feat_id, data_folder, feature_faces_meshes_pool)

            for curve in sketch_anchor_lines:
                new_curve = utils.remove_zero_length_edges(curve)
                if len(np.array(new_curve).shape) > 1:
                    all_edges.append({"geometry": new_curve, "type": "grid_lines",
                                      "feature_id": feat_id,
                                      "fitted_curve": utils_occ.fit_curve(new_curve)
                                      })
            if VERBOSE:
                utils.plot_curves(sketch_anchor_lines,
                                  name_prefix=str(feat_id) + "_anchor_lines",
                                  color=color_map[feat_id])
            for curve in grid_lines:
                new_curve = utils.remove_zero_length_edges(curve)
                if len(np.array(new_curve).shape) > 1:
                    all_edges.append({"geometry": new_curve, "type": "grid_lines",
                                      "feature_id": feat_id,
                                      "fitted_curve": utils_occ.fit_curve(new_curve)
                                      })
            # compute loops
            sketch_utils.compute_sketch_loops(parsed_features, feat["entity"])
            # sort loops by convex hull surface
            for curve in sketch["curves"]:
                if len(curve) == 0:
                    continue
                new_curve = utils.remove_zero_length_edges(curve)
                if len(np.array(new_curve).shape) > 1:
                    all_edges.append({"geometry": new_curve, "type": "sketch",
                                      "feature_id": feat_id,
                                      "fitted_curve": utils_occ.fit_curve(new_curve)
                                      })
            #if VERBOSE:
            #ps.init()
            #utils.plot_curves(sketch["curves"], name_prefix=str(feat_id), color=color_map[feat_id])
            #ps.show()

            for const in ent["constraints"]:
                if const["constraintType"] == "MIDPOINT":
                    midpoint_curves, min_affected_stroke_id = \
                        sketch_utils.draw_midpoint_curves(const, sketch, grid_lines, all_edges,
                                                          VERBOSE=False)
                    for new_curve in reversed(midpoint_curves):
                        all_edges.insert(min_affected_stroke_id,
                                         {"geometry": new_curve, "type": "grid_lines",
                                          "feature_id": feat_id,
                                          "fitted_curve": utils_occ.fit_curve(new_curve)})
                        #all_edges.append({"geometry": new_curve, "type": "grid_lines",
                        #                  "feature_id": feat_id,
                        #                  "fitted_curve": utils_occ.fit_curve(new_curve)})


            #parsed_features["entities"][feat["entity"]]["sketch_constraints"] = constraints

        if feat["type"] == "extrude" and not only_final_npr_lines:
            ent = parsed_features["entities"][feat["entity"]]
            geo_ids = ent["entities"]
            faces, edges, vertices = sketch_utils.get_faces_edges_vertices(geo_ids, parsed_features)
            #    # TODO: loop through bodydetails
            # TODO: get faces, get vertices and normals for each face
            # let's assume, there's only one normal
            if len(faces) == 0:
                for curve in curves_to_plot:
                    new_curve = utils.remove_zero_length_edges(curve[0])
                    if len(np.array(new_curve).shape) > 1:
                        all_edges.append({"geometry": new_curve, "type": "feature_line",
                                          "feature_id": feat_id,
                                          "fitted_curve": utils_occ.fit_curve(new_curve)})
                for curve in silhouette_lines:
                    new_curve = utils.remove_zero_length_edges(curve)
                    if len(np.array(new_curve).shape) > 1:
                        all_edges.append({"geometry": new_curve, "type": "silhouette_line",
                                          "feature_id": feat_id,
                                          "fitted_curve": utils_occ.fit_curve(new_curve)})
                continue
            if "param" in faces[0].keys():
                extrude_normal = np.array(faces[0]["param"]["normal"])
            plane_normals = [np.cross(extrude_normal, np.array([1, 0, 0])),
                             np.cross(extrude_normal, np.array([0, 1, 0])),
                             np.cross(extrude_normal, np.array([0, 0, 1]))]
            # TODO: get depth
            extrude_depth_one = 0.0
            extrude_depth_two = 0.0
            if "param" in faces[0].keys():
                if ent["use_depth"]:
                    extrude_depth_one = ent["extent_one"]
                    extrude_depth_two = ent["extent_two"]
                #print(ent)
                if not ent["use_depth"] and ent["opposite_direction"]:
                    extrude_normal *= -1

            # TODO: extrude vertices along normals for the depth
            extrude_lines = []
            extrude_points = []
            extrude_convex_hull = []
            connect_extruded_points = []
            for face_id, face in enumerate(faces):
                # TODO: add edge-specific lines. e.g. for circles
                for edge_id, edge in enumerate(edges[face_id].values()):
                    if "param" in edge.keys():
                        if edge["param"]["type"] == "Circle" and len(edge["vertices"]) < 2: # only full-sweep circles
                            circle_lines = circle_3d.add_square(edge)
                            for curve in circle_lines:
                                all_edges.append({"geometry": curve, "type": "circle_square_line",
                                                  "feature_id": feat_id,
                                                  "fitted_curve": utils_occ.fit_curve(curve)
                                                  })
                            if VERBOSE:
                                utils.plot_curves(circle_lines, name_prefix=str(feat_id)+"_"+str(face_id)+"_"+str(edge_id)+"_circle_square", color=color_map[feat_id])
                            for line in circle_lines:
                                # add points which are on the circle
                                extrude_points.append(line[0])
                                extrude_points.append(line[1])
                                #connect_extruded_points.append([len(extrude_points)-2, len(extrude_points)-1])
                                extrude_convex_hull.append(line[0])
                                extrude_convex_hull.append(line[1])

                for vert in vertices[face_id].values():
                    extrude_convex_hull.append(vert)
                    extrude_points.append(vert)

            # look for all faces containing any new edge
            if not ent["use_depth"]:
                new_faces = []
                for edge_id in new_edge_ids:
                    for bd in bodydetails_pool:
                        for body in bd:
                            for face in body["faces"]:
                                for loop in face["loops"]:
                                    for coedge in loop["coedges"]:
                                        if edge_id == coedge["edgeId"]:
                                            new_faces.append(face["id"])
                for new_face_id in new_faces:
                    # find face in feature_faces
                    for feature_faces in reversed(feature_faces_pool):
                        if new_face_id in feature_faces.keys():
                            for facet in feature_faces[new_face_id]:
                                for p in facet:
                                    extrude_convex_hull.append(p)
                            break

            # go through extrude points and look for intersection with new lines
            #replace_extrude_line_ids = []
            #for ext_p in extrude_points:
            #    if not ent["use_depth"]:
            #        replace_extrude_line_ids.append(len(extrude_lines))
            #        extrude_lines.append([ext_p, ext_p+extrude_normal*0.1])
            #    else:
            #        extrude_lines.append([ext_p - extrude_normal*extrude_depth_two, ext_p + extrude_normal * extrude_depth_one])

            if not ent["use_depth"]:
                # filter out all points from the convex hull which are behind the extrude points
                #min_extrude_pts_depth = np.min(np.dot(extrude_normal, np.array(extrude_points).T)) - 0.001
                #extrude_convex_hull = np.array(extrude_convex_hull)
                #extrude_convex_hull = extrude_convex_hull[np.dot(extrude_normal, extrude_convex_hull.T) > min_extrude_pts_depth]
                #ps.register_point_cloud("convex_hull_pts", np.array(extrude_convex_hull))
                #extrude_convex_hull = trimesh.PointCloud(extrude_convex_hull).convex_hull
                obj_mesh_file = os.path.join(data_folder, "shape_"+str(feat_id-1)+".obj")
                v = []
                f = []
                if os.path.exists(obj_mesh_file):
                    v, f = igl.read_triangle_mesh(obj_mesh_file)
                if VERBOSE:
                    ps.register_surface_mesh("convex_hull", vertices=f,
                                             faces=extrude_convex_hull.faces)
                obj_mesh = trimesh.Trimesh(vertices=v, faces=f)
                #raycaster = ray_triangle.RayMeshIntersector(extrude_convex_hull)
                raycaster = ray_triangle.RayMeshIntersector(obj_mesh)
                for ext_p_id, ext_p in enumerate(extrude_points):
                    ray_origin = ext_p-extrude_normal
                    ray_direction = extrude_normal
                    trimesh.constants.tol.zero = 1e-4
                    hits, _, _ = raycaster.intersects_location(ray_origins=[ray_origin],
                                                               ray_directions=[ray_direction], multiple_hits=True)
                    trimesh.constants.tol.__init__()
                    # we expect two hits !
                    if len(hits) < 2:
                        print("Raycasting found only ", str(len(hits)), "hits!")
                        continue

                    if len(hits) > 0:
                        hits = hits[np.linalg.norm(hits - ext_p, axis=-1) > 1e-4]
                        hit_dot_prods = np.dot(extrude_normal, hits.T)
                        ext_p_prod = np.dot(extrude_normal, ext_p)
                        sorted_hit_prods = np.sort(hit_dot_prods)
                        sorted_hit_prods_args = np.argsort(hit_dot_prods)
                        if len(sorted_hit_prods_args[sorted_hit_prods > ext_p_prod]) == 0:
                            continue
                        first_inter_hit_id = sorted_hit_prods_args[sorted_hit_prods > ext_p_prod][0]
                        tmp_extrude_depth = np.linalg.norm(np.array(ext_p) - np.array(hits[first_inter_hit_id]))
                        extrude_depth_one = np.maximum(extrude_depth_one, tmp_extrude_depth)

                        #ps.init()
                        #ps.register_surface_mesh("mesh", obj_mesh.vertices, obj_mesh.faces)
                        #ps.register_point_cloud("hits", hits)
                        #ps.register_point_cloud("ext_p", np.array([ext_p]))
                        #ps.show()

                    #if len(hits) > 0:
                    #    #ext_line = [hits[0], hits[-1]]
                    #    #extrude_lines[replace_extrude_line_ids[ext_p_id]] = ext_line
                    #    extrude_depth_one = np.maximum(extrude_depth_one, np.linalg.norm(np.array(hits[0]) - np.array(hits[-1])))
            #for cnct_ext_pts in connect_extruded_points:
            #    extrude_lines.append([extrude_lines[cnct_ext_pts[0]][0], extrude_lines[cnct_ext_pts[1]][0]])
            #    extrude_lines.append([extrude_lines[cnct_ext_pts[0]][1], extrude_lines[cnct_ext_pts[1]][1]])
            #extrude_lines = extrude_utils.extrude_faces(geo_ids, parsed_features, perspective_grids,
            #                                            extrude_normal, extrude_depth_one, extrude_depth_two)

            for geo_id in geo_ids:
                found_face = False
                for sketch in parsed_features["sequence"]:
                    if sketch["type"] != "Sketch":
                        continue
                    sketch_ent = parsed_features["entities"][sketch["entity"]]
                    sketch_profiles = sketch_ent["profiles"]
                    for face in sketch_profiles["faces"]:
                        if face["id"] == geo_id:
                            # extrude this face
                            grid = perspective_grids[sketch["entity"]]
                            loops_edge_ids = face["loops_edge_ids"]
                            for loop_id, loop_edge_ids in enumerate(loops_edge_ids):
                                vertices_ids = []
                                for edge_id in sorted(loop_edge_ids):
                                    if sketch_ent["profiles"]["edges"][edge_id]["param"]["type"] == "Circle":
                                        vertices_ids.append(edge_id)
                                    vertices_ids += sketch_ent["profiles"]["edges"][edge_id]["vertices"]
                                #print("extrude_depth_two", extrude_depth_two)
                                grid_extrude_lines = grid.extrude_grids(vertices_ids, extrude_normal, extrude_depth_one,
                                                                        extrude_depth_two)
                                #if ent["name"] == "Extrude 2":
                                #    print(vertices_ids)
                                #    exit()
                                extrude_lines = grid.extrude_vertices(vertices_ids, extrude_normal, extrude_depth_one,
                                                                      extrude_depth_two)
                                # per-line extrude depth
                                if not ent["use_depth"]:
                                    #raycaster = ray_triangle.RayMeshIntersector(extrude_convex_hull)
                                    raycaster = ray_triangle.RayMeshIntersector(obj_mesh)
                                    for ext_line_id, ext_line in enumerate(extrude_lines):
                                    #for ext_p_id, ext_p in enumerate(extrude_points):
                                        ext_p = ext_line[0]
                                        ray_origin = ext_p - extrude_normal
                                        ray_direction = extrude_normal
                                        trimesh.constants.tol.zero = 1e-4
                                        hits, _, _ = raycaster.intersects_location(ray_origins=[ray_origin],
                                                                                   ray_directions=[ray_direction],
                                                                                   multiple_hits=True)
                                        trimesh.constants.tol.__init__()
                                        # we expect two hits !
                                        if len(hits) < 2:
                                            print("Raycasting found only ", str(len(hits)), "hits!")
                                            #ps.init()
                                            #ps.register_surface_mesh("obj_mesh", obj_mesh.vertices, obj_mesh.faces)
                                            #utils.plot_curves([[ext_p-0.01*extrude_normal, ext_p+0.01*extrude_normal]])
                                            #ps.show()
                                            continue

                                        if len(hits) > 0:
                                            #if VERBOSE:
                                            # ext_line = [hits[0], hits[-1]]
                                            # extrude_lines[replace_extrude_line_ids[ext_p_id]] = ext_line
                                            #tmp_extrude_depth = np.linalg.norm(np.array(hits[0]) - np.array(hits[-1]))
                                            # find first intersection from ext_p in extrude_normal direction
                                            hit_dot_prods = np.dot(extrude_normal, hits.T)
                                            ext_p_prod = np.dot(extrude_normal, ext_p)
                                            sorted_hit_prods = np.sort(hit_dot_prods)
                                            sorted_hit_prods_args = np.argsort(hit_dot_prods)
                                            if len(sorted_hit_prods_args[sorted_hit_prods > ext_p_prod+0.001]) == 0:
                                                continue
                                            first_inter_hit_id = sorted_hit_prods_args[sorted_hit_prods > ext_p_prod+0.001][0]
                                            tmp_extrude_depth = np.linalg.norm(np.array(ext_p) - np.array(hits[first_inter_hit_id]))
                                            #print(hit_dot_prods)
                                            #print(sorted_hit_prods)
                                            #print(sorted_hit_prods_args)
                                            #print(ext_p_prod)
                                            #print(sorted_hit_prods > ext_p_prod)
                                            #print(tmp_extrude_depth)
                                            extrude_depth_one = np.maximum(extrude_depth_one, tmp_extrude_depth)
                                            extrude_lines[ext_line_id][1] = ext_line[0]+tmp_extrude_depth*extrude_normal
                                            #if VERBOSE:
                                            #    ps.init()
                                            #    ps.register_surface_mesh("convex_hull", vertices=v,
                                            #                             faces=f)
                                            #    ps.register_point_cloud("hits", hits)
                                            #    ps.register_point_cloud("ext_p", np.array([ext_p]))
                                            #    print(ext_line)
                                            #    print(ext_p)
                                            #    print(hits)
                                            #    ps.show()
                                section_lines = extrude_utils.get_mesh_section_lines_v2(feat_id, data_folder, plane_normals,
                                                                                     extrude_lines, feature_faces_meshes_pool)

                                # get section lines for each extrude_line
                                # TODO: output after each loop
                                # return extrude_lines
                                if VERBOSE:
                                    utils.plot_curves(grid_extrude_lines, str(feat_id)+"_construction_"+geo_id+"_"+str(loop_id), color=(0, 0, 0), radius=0.00515)
                                for curve in grid_extrude_lines:
                                    new_curve = utils.remove_zero_length_edges(curve)
                                    if len(np.array(new_curve).shape) > 1:
                                        all_edges.append({"geometry": new_curve, "type": "grid_lines",
                                                          "feature_id": feat_id,
                                                          "fitted_curve": utils_occ.fit_curve(new_curve)
                                                          })
                                for curve in section_lines:
                                    new_curve = utils.remove_zero_length_edges(curve)
                                    if len(np.array(new_curve).shape) > 1:
                                        try:
                                            all_edges.append({"geometry": new_curve, "type": "section_lines",
                                                              "feature_id": feat_id,
                                                              "fitted_curve": utils_occ.fit_curve(new_curve)
                                                              })
                                        except:
                                            print(new_curve)
                                            ps.init()
                                            utils.plot_curves([new_curve], "curve")
                                            ps.show()
                                for curve in extrude_lines:
                                    new_curve = utils.remove_zero_length_edges(curve)
                                    if len(np.array(new_curve).shape) > 1:
                                        all_edges.append({"geometry": new_curve, "type": "extrude_line",
                                                          "feature_id": feat_id,
                                                          "fitted_curve": utils_occ.fit_curve(new_curve)
                                                          })
                                if VERBOSE:
                                    ps.show()
                                if output_svg:
                                    file_name = os.path.join(data_folder, str(feat_id)+"_"+feat["type"]+"_"+geo_id+"_"+str(loop_id))
                                    svg_file_name = file_name+".svg"
                                    typed_feature_lines_to_svg(deepcopy(all_edges), svg_file_name=svg_file_name,
                                                               theta=theta, phi=phi, title=feat["type"])
                                    #pdf_file_name = os.path.join("data", str(abc_id), "out.pdf")
                                    pdf_file_name = file_name + ".pdf"
                                    os.system("rsvg-convert -f pdf "+svg_file_name+" > "+pdf_file_name)

            for curve in curves_to_plot:
                new_curve = utils.remove_zero_length_edges(curve[0])
                if len(np.array(new_curve).shape) > 1:
                    all_edges.append({"geometry": new_curve, "type": "feature_line",
                                      "feature_id": feat_id,
                                      "fitted_curve": utils_occ.fit_curve(new_curve)})
            for curve in silhouette_lines:
                new_curve = utils.remove_zero_length_edges(curve)
                if len(np.array(new_curve).shape) > 1:
                    all_edges.append({"geometry": new_curve, "type": "silhouette_line",
                                      "feature_id": feat_id,
                                      "fitted_curve": utils_occ.fit_curve(new_curve)})

        if feat["type"] == "fillet" and not only_final_npr_lines and include_fillet_lines:
            ent = parsed_features["entities"][feat["entity"]]
            #if not ent["fillet_type"] == "EDGE" or not ent["cross_section"] == "CIRCULAR":
            #    for curve in curves_to_plot:
            #        new_curve = utils.remove_zero_length_edges(curve[0])
            #        if len(np.array(new_curve).shape) > 1:
            #            all_edges.append({"geometry": new_curve, "type": "feature_line",
            #                              "feature_id": feat_id,
            #                              "fitted_curve": utils_occ.fit_curve(new_curve)})
            #    for curve in silhouette_lines:
            #        new_curve = utils.remove_zero_length_edges(curve)
            #        if len(np.array(new_curve).shape) > 1:
            #            all_edges.append({"geometry": new_curve, "type": "silhouette_line",
            #                              "feature_id": feat_id,
            #                              "fitted_curve": utils_occ.fit_curve(new_curve)})
            #    continue
            # get new edges, created by the fillet
            plot_arcs = []
            edges_triplets = {}
            # define fillet design constraints as projection constraints
            tmp_edge_ids = []
            tmp_edge_body_ids = []
            for edge_id in ent["entities"]:
                if edge_id in feature_lines_pool[-2].keys():
                    tmp_edge_ids.append(edge_id)
                    for body in bodydetails_pool[-2]:
                        for tmp_tmp_edge in body["edges"]:
                            if tmp_tmp_edge["id"] == edge_id:
                                tmp_edge_body_ids.append(body["id"])
                else:
                    # check if it's a face id and add all edges
                    if edge_id in feature_faces_pool[-2]:
                        for body in bodydetails_pool[-2]:
                            for face in body["faces"]:
                                if face["id"] == edge_id:
                                    for loop_edge_id in face["loops"][0]["coedges"]:
                                        tmp_edge_ids.append(loop_edge_id["edgeId"])
                                        tmp_edge_body_ids.append(body["id"])
                                    break
                #print(tmp_edge_ids)
            ent["entities"], indices = np.unique(tmp_edge_ids, return_index=True)
            ent["entities"] = ent["entities"].tolist()
            tmp_edge_body_ids = np.array(tmp_edge_body_ids)[indices]
            #print(ent["entities"])
            constraints = []
            for old_vec_id, edge_id in enumerate(ent["entities"]):
                tmp_edge_triplet = {}
                tmp_edge_triplet["old_edge_id"] = edge_id
                tmp_edge_triplet["old_edge_3d"] = np.array(feature_lines_pool[-2][edge_id])
                if len(tmp_edge_triplet["old_edge_3d"]) > 2:
                    continue
                old_dir_vec = tmp_edge_triplet["old_edge_3d"][-1] - tmp_edge_triplet["old_edge_3d"][0]
                old_dir_vec /= np.linalg.norm(old_dir_vec)
                new_edge_dists = []
                new_edge_3ds = []
                for vec_id, new_edge_id in enumerate(new_edge_ids):
                    if new_edge_body_ids[vec_id] != tmp_edge_body_ids[old_vec_id]:
                        continue
                    new_edge_3d = np.array(feature_lines_pool[-1][new_edge_id])
                    if len(new_edge_3d) > 2:
                        continue
                    dir_vec = new_edge_3d[-1] - new_edge_3d[0]
                    dir_vec /= np.linalg.norm(dir_vec)
                    if not np.isclose(np.abs(np.dot(dir_vec, old_dir_vec)), 1.0):
                        continue
                    dist = Line(tmp_edge_triplet["old_edge_3d"][0], old_dir_vec).distance_line(Line(new_edge_3d[0], dir_vec))
                    new_edge_dists.append(dist)
                    new_edge_3ds.append(new_edge_3d)
                # pick closest two
                if len(new_edge_dists) < 2:
                    continue
                sorted_ids = np.argsort(new_edge_dists)
                tmp_edge_triplet["new_edge_0_3d"] = new_edge_3ds[sorted_ids[0]]
                tmp_edge_triplet["new_edge_1_3d"] = new_edge_3ds[sorted_ids[1]]

                # construct point triplet
                l0 = tmp_edge_triplet["new_edge_0_3d"]
                l1 = tmp_edge_triplet["new_edge_1_3d"]
                new_lines = [l0, l1]
                #utils.plot_curves(new_lines, name_prefix=str(feat_id)+"_new_lines", color=color_map[feat_id])
                lengths = [np.linalg.norm(l0[-1]-l0[0]), np.linalg.norm(l1[-1]-l1[0])]
                short_id = np.argmin(lengths)
                short_line = new_lines[short_id]
                dir_vec = short_line[-1] - short_line[0]
                dir_vec /= np.linalg.norm(dir_vec)
                edges_triplets[edge_id] = []
                for p in new_lines[short_id]:
                    cross_section = Plane(p, dir_vec)
                    other_line = new_lines[1-short_id]
                    other_dir_vec = other_line[-1] - other_line[0]
                    other_dir_vec /= np.linalg.norm(other_dir_vec)
                    p1 = cross_section.intersect_line(Line(other_line[0], other_dir_vec))
                    p2 = cross_section.intersect_line(Line(tmp_edge_triplet["old_edge_3d"][0], old_dir_vec))
                    circle_center = p + (p1 - p2)
                    first_axis = p - circle_center
                    snd_axis = p1 - circle_center

                    #ps.init()
                    #obj_mesh_file = os.path.join(data_folder, "shape_"+str(feat_id)+".obj")
                    #ps.register_point_cloud("circle_center", np.array([circle_center]))
                    #if os.path.exists(obj_mesh_file):
                    #    v, f = igl.read_triangle_mesh(obj_mesh_file)
                    #    ps.register_surface_mesh("surface", vertices=v, faces=f, color=mesh_color)
                    #ps.register_curve_network("l0", np.array(l0), np.array([[0, 1]]))
                    #ps.register_curve_network("l1", np.array(l1), np.array([[0, 1]]))
                    #ps.show()

                    arc = []
                    for t in np.linspace(0, 90, 20):
                        pt = np.array(circle_center + np.cos(np.deg2rad(t))*first_axis + np.sin(np.deg2rad(t))*snd_axis)
                        if len(arc) > 0:
                            if np.isclose(np.linalg.norm(arc[-1] - pt), 0.0):
                                continue
                        arc.append(pt)
                    #arc = np.cos(np.deg2rad(np.linspace(0, 90, 20)))*first_axis + np.sin(np.deg2rad(np.linspace(0, 90, 20)))*snd_axis
                    plot_arcs.append(arc)
                    plot_arcs.append([p, circle_center])
                    plot_arcs.append([p1, circle_center])
                    plot_arcs.append([p2, p])
                    plot_arcs.append([p2, p1])
                    edges_triplets[edge_id].append({"edge_3d": arc,
                                                    "support_lines": np.array([[p2, p1], [p2, p], [p, circle_center], [p1, circle_center]]).tolist()})
                    #all_edges.append({"geometry": arc, "type": "fillet_line", "feature_id": feat_id})
                    #all_edges.append({"geometry": np.array([p2, p1]), "type": "grid_lines", "feature_id": feat_id})
                    #all_edges.append({"geometry": np.array([p2, p]), "type": "grid_lines", "feature_id": feat_id})
                    #all_edges.append({"geometry": np.array([p, circle_center]), "type": "grid_lines", "feature_id": feat_id})
                    #all_edges.append({"geometry": np.array([p1, circle_center]), "type": "grid_lines", "feature_id": feat_id})
                    constraints.append({"constraintType": "HORIZONTAL",
                                        "first_3d": np.array(p2).tolist(),
                                        "second_3d": np.array(p1).tolist()})
                    constraints.append({"constraintType": "HORIZONTAL",
                                        "first_3d": np.array(p2).tolist(),
                                        "second_3d": np.array(p).tolist()})
                    constraints.append({"constraintType": "HORIZONTAL",
                                        "first_3d": np.array(p).tolist(),
                                        "second_3d": np.array(circle_center).tolist()})
                    constraints.append({"constraintType": "HORIZONTAL",
                                        "first_3d": np.array(p1).tolist(),
                                        "second_3d": np.array(circle_center).tolist()})

            # add supplemental fillet lines
            all_const_pts = np.array([p for const in constraints for p in [const["first_3d"], const["second_3d"]]])
            for const in constraints:
                dir = np.array(const["first_3d"]) - np.array(const["second_3d"])
                if len(const["first_3d"]) == 0:
                    continue
                if np.isclose(np.linalg.norm(dir), 0.0):
                    continue
                l = Line(const["first_3d"], dir)
                dists = [l.distance_point(p) for p in all_const_pts]
                projections = np.dot(dir, all_const_pts.T)
                valid_pts = np.isclose(dists, 0.0, atol=1e-4)
                new_line = [all_const_pts[valid_pts][np.argmin(projections[valid_pts])],
                            all_const_pts[valid_pts][np.argmax(projections[valid_pts])]]
                all_edges.append(
                    {"geometry": np.array(new_line).tolist(),
                     "type": "grid_lines",
                     "feature_id": feat_id,
                     "fitted_curve": utils_occ.fit_curve(new_line)
                     }
                )

            # add fillet support lines
            for edge_triplet in edges_triplets.values():
                for edge in edge_triplet:
                    for supp in edge["support_lines"]:
                        all_edges.append(
                            {"geometry": np.array(supp).tolist(),
                             "type": "grid_lines",
                             "feature_id": feat_id,
                             "fitted_curve": utils_occ.fit_curve(supp)
                             }
                        )

            # add actual fillet arcs
            for edge_triplet in edges_triplets.values():
                for edge in edge_triplet:
                    all_edges.append(
                        {"geometry": np.array(edge["edge_3d"]).tolist(),
                         "type": "fillet_line",
                         "feature_id": feat_id,
                         "support_lines": edge["support_lines"],
                         "fitted_curve": utils_occ.fit_curve(edge["edge_3d"])
                         }
                    )
            utils.plot_curves(plot_arcs, name_prefix=str(feat_id)+"_construction", color=color_map[feat_id])

            parsed_features["entities"][feat["entity"]]["fillet_projection_constraints"] = constraints
            #parsed_features["entities"][feat["entity"]]["fillet_projection_constraints"] = []
            with open(os.path.join(data_folder, "parsed_features.json"), "w") as fp:
                json.dump(parsed_features, fp, indent=4)
                #parsed_features = json.load(f)
            # associate two new edges to each old edge
            # form point triplets between old edge and new edges
            # draw quarter circles in these point triplets
            for curve in curves_to_plot:
                new_curve = utils.remove_zero_length_edges(curve[0])
                if len(np.array(new_curve).shape) > 1:
                    all_edges.append({"geometry": new_curve, "type": "feature_line",
                                      "feature_id": feat_id,
                                      "fitted_curve": utils_occ.fit_curve(new_curve)})
            for curve in silhouette_lines:
                new_curve = utils.remove_zero_length_edges(curve)
                if len(np.array(new_curve).shape) > 1:
                    all_edges.append({"geometry": new_curve, "type": "silhouette_line",
                                      "feature_id": feat_id,
                                      "fitted_curve": utils_occ.fit_curve(new_curve)})
        if VERBOSE:
            ps.show()
        if output_svg:
            file_name = os.path.join(data_folder, str(feat_id)+"_"+feat["type"])
            svg_file_name = file_name+".svg"
            typed_feature_lines_to_svg(deepcopy(all_edges), svg_file_name=svg_file_name,
                                       theta=theta, phi=phi, title=feat["type"])
            #pdf_file_name = os.path.join("data", str(abc_id), "out.pdf")
            pdf_file_name = file_name + ".pdf"
            os.system("rsvg-convert -f pdf "+svg_file_name+" > "+pdf_file_name)
    return all_edges

if __name__ == "__main__":
    abc_id = 24
    theta = 60
    phi = -90-35
    phi = 50
    #np.seterr(all='raise')
    data_folder = os.path.join("data", str(abc_id))
    #all_edges = draw_extrude(abc_id, theta=theta, phi=phi, VERBOSE=True)
    all_edges = draw_extrude(data_folder, theta=theta, phi=phi)
    svg_file_name = os.path.join("data", str(abc_id), "out.svg")
    typed_feature_lines_to_svg(deepcopy(all_edges), svg_file_name=svg_file_name,
                               theta=theta, phi=phi, title="Final drawing")
    pdf_file_name = os.path.join("data", str(abc_id), "out.pdf")
    os.system("rsvg-convert -f pdf "+svg_file_name+" > "+pdf_file_name)
    all_edges_file_name = os.path.join("data", str(abc_id), "all_edges.json")
    for edge_id in range(len(all_edges)):
        all_edges[edge_id]["geometry"] = np.array(all_edges[edge_id]["geometry"]).tolist()
    with open(all_edges_file_name, "w") as f:
        json.dump(all_edges, f)
