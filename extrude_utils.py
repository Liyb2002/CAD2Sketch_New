import os
import polyscope as ps
from trimesh.exchange.obj import load_obj
import trimesh
#import pymesh
import numpy as np
from skspatial.objects import Plane
import utils

def extrude_faces(face_ids, parsed_features, perspective_grids,
                  extrude_normal, extrude_depth_one, extrude_depth_two):
    #print(face_ids)
    extrude_lines = []
    for geo_id in face_ids:
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
                    #print(grid)
                    loops_edge_ids = face["loops_edge_ids"]
                    for loop_edge_ids in loops_edge_ids:
                        vertices_ids = []
                        for edge_id in loop_edge_ids:
                            if sketch_ent["profiles"]["edges"][edge_id]["param"]["type"] == "Circle":
                                vertices_ids.append(edge_id)
                            vertices_ids += sketch_ent["profiles"]["edges"][edge_id]["vertices"]
                        #print(vertices_ids)
                        extrude_lines += grid.extrude_vertices(vertices_ids, extrude_normal, extrude_depth_one,
                                                               extrude_depth_two)
                        # TODO: output after each loop
                        #return extrude_lines
    return extrude_lines

def get_mesh_section_lines(feat_id, data_folder, plane_normals, extrude_lines, feature_faces_meshes_pool):

    # draw section line for each extrude line
    # get 3d mesh from previous step
    print("get_mesh_section_lines")
    section_lines = []
    already_sectioned_planes = []
    sliced_lines_cnt = 0
    for j in range(1, feat_id):
        print(j)
        print(len(extrude_lines))
        obj_mesh_file = os.path.join(data_folder, "shape_" + str(feat_id-j) + ".obj")
        if os.path.exists(obj_mesh_file):
            #mesh_faces = pymesh.load_mesh(obj_mesh_file)
            with open(obj_mesh_file, "r") as fp:
                obj_dict = load_obj(fp)
                mesh_faces = trimesh.Trimesh(vertices=obj_dict["vertices"],
                                             faces=obj_dict["faces"])
            for plane_normal_id, plane_normal in enumerate(plane_normals):
                if np.all(np.isclose(plane_normal, 0.0)):
                    continue
                plane_normal /= np.linalg.norm(plane_normal)
                #print(len(extrude_lines))
                for ext_line_id, ext_line in enumerate(extrude_lines):
                    # construct section plane

                    # check for redundant planes
                    plane_origin = np.array(ext_line[0])
                    tmp_plane = Plane(point=plane_origin, normal=plane_normal)
                    already_sectioned = False
                    for check_plane in already_sectioned_planes:
                        if check_plane.is_close(tmp_plane):
                            already_sectioned = True
                            break
                    if already_sectioned:
                        #print("already sectioned")
                        continue
                    already_sectioned_planes.append(tmp_plane)

                    boundary_edges = utils.slice_mesh_2(mesh_faces, plane_normal, 1, plane_origin=plane_origin)
                    #if VERBOSE:
                    #    if len(boundary_edges) > 0:
                    #        ps.register_curve_network("sliced_lines_"+str(sliced_lines_cnt), nodes=np.array(boundary_edges).reshape(-1, 3),
                    #                                  edges=np.array(
                    #                                      [[2 * i, 2 * i + 1] for i in range(len(boundary_edges))]))
                    sliced_lines_cnt += 1
                    surface_ids = []
                    tmp_edges = []
                    print("len(boundary_edges)", len(boundary_edges))
                    if len(boundary_edges) > 100:
                        ps.init()
                        ps.register_surface_mesh("mesh", mesh_faces.vertices, mesh_faces.faces)
                        ps.show()
                    for tmp_edge in boundary_edges:
                        print(tmp_edge)
                        in_face, surface_id = utils.in_faces_2(feature_faces_meshes_pool[-1-j], tmp_edge, plane_normal)
                        if in_face:
                            tmp_edges.append(tmp_edge)
                            surface_ids.append(surface_id)
                    tmp_edges = utils.unify_same_face_edges_2(tmp_edges, surface_ids)
                    for curve in tmp_edges:
                        new_curve = utils.remove_zero_length_edges(curve)
                        if len(new_curve) > 0:
                            new_line, is_good_line_fit = utils.line_segment_from_points(new_curve)
                            if is_good_line_fit:
                                section_lines.append(new_line)
                            else:
                                section_lines.append(new_curve)
                            #all_edges.append({"geometry": curve, "type": "section_line"})
                    #if VERBOSE:
                    #    utils.plot_curves(tmp_edges, str(feat_id)+"_"+str(plane_normal_id)+"_"+str(ext_line_id)+"_tmp_edges",
                    #                      color=(0, 1, 0))
    return section_lines

def get_mesh_section_lines_v2(feat_id, data_folder, plane_normals, extrude_lines, feature_faces_meshes_pool):

    # draw section line for each extrude line
    # get 3d mesh from previous step
    #print("get_mesh_section_lines")
    sliced_lines_cnt = 0
    new_section_lines = []
    for feature_faces_meshes in feature_faces_meshes_pool[:-1]:
        #print(feature_faces_meshes)
        #if len(feature_faces_meshes.keys()) > 0:
        #    exit()
        for face_id in feature_faces_meshes.keys():
            # ignore non-planar faces
            mesh_faces = feature_faces_meshes[face_id]
            plane = Plane.best_fit(mesh_faces.vertices)
            plane_dists = np.array([plane.distance_point(p) for p in mesh_faces.vertices])
            if not np.all(np.isclose(plane_dists, 0.0, atol=1e-4)):
                continue
            for plane_normal_id, plane_normal in enumerate(plane_normals):
                if np.isclose(np.abs(np.dot(plane.normal, plane_normal)), 1.0, atol=1e-4):
                    continue
                already_sectioned_planes = []
                if np.all(np.isclose(plane_normal, 0.0)):
                    continue
                plane_normal /= np.linalg.norm(plane_normal)
                #print(len(extrude_lines))
                for ext_line_id, ext_line in enumerate(extrude_lines):
                    # construct section plane

                    # check for redundant planes
                    plane_origin = np.array(ext_line[0])
                    tmp_plane = Plane(point=plane_origin, normal=plane_normal)
                    already_sectioned = False
                    for check_plane in already_sectioned_planes:
                        if check_plane.is_close(tmp_plane):
                            already_sectioned = True
                            break
                    if already_sectioned:
                        #print("already sectioned")
                        continue
                    already_sectioned_planes.append(tmp_plane)

                    boundary_edges = utils.slice_mesh_2(mesh_faces, plane_normal, 1, plane_origin=plane_origin)
                    #print("len(boundary_edges)", len(boundary_edges))

                    surface_ids = [face_id for i in range(len(boundary_edges))]

                    tmp_edges = utils.unify_same_face_edges_2(boundary_edges, surface_ids)
                    for curve in tmp_edges:
                        new_curve = utils.remove_zero_length_edges(curve)
                        if len(new_curve) > 0:
                            new_line, is_good_line_fit = utils.line_segment_from_points(new_curve)
                            #if is_good_line_fit:
                            # lines here should only come from planar faces, so they are always straight lines
                            new_section_lines.append(new_line)
                            #else:
                            #    new_section_lines.append(new_curve)

    #section_lines = []
    #for j in range(1, feat_id):
    #    print(j)
    #    print(len(extrude_lines))
    #    obj_mesh_file = os.path.join(data_folder, "shape_" + str(feat_id-j) + ".obj")
    #    already_sectioned_planes = []
    #    if os.path.exists(obj_mesh_file):
    #        #mesh_faces = pymesh.load_mesh(obj_mesh_file)
    #        with open(obj_mesh_file, "r") as fp:
    #            obj_dict = load_obj(fp)
    #            mesh_faces = trimesh.Trimesh(vertices=obj_dict["vertices"],
    #                                         faces=obj_dict["faces"])
    #        for plane_normal_id, plane_normal in enumerate(plane_normals):
    #            if np.all(np.isclose(plane_normal, 0.0)):
    #                continue
    #            plane_normal /= np.linalg.norm(plane_normal)
    #            #print(len(extrude_lines))
    #            for ext_line_id, ext_line in enumerate(extrude_lines):
    #                # construct section plane

    #                # check for redundant planes
    #                plane_origin = np.array(ext_line[0])
    #                tmp_plane = Plane(point=plane_origin, normal=plane_normal)
    #                already_sectioned = False
    #                for check_plane in already_sectioned_planes:
    #                    if check_plane.is_close(tmp_plane):
    #                        already_sectioned = True
    #                        break
    #                if already_sectioned:
    #                    continue
    #                already_sectioned_planes.append(tmp_plane)

    #                boundary_edges = utils.slice_mesh_2(mesh_faces, plane_normal, 1, plane_origin=plane_origin)
    #                #if VERBOSE:
    #                #    if len(boundary_edges) > 0:
    #                #        ps.register_curve_network("sliced_lines_"+str(sliced_lines_cnt), nodes=np.array(boundary_edges).reshape(-1, 3),
    #                #                                  edges=np.array(
    #                #                                      [[2 * i, 2 * i + 1] for i in range(len(boundary_edges))]))
    #                sliced_lines_cnt += 1
    #                surface_ids = []
    #                tmp_edges = []
    #                print("len(boundary_edges)", len(boundary_edges))
    #                if len(boundary_edges) > 100:
    #                    #ps.init()
    #                    ps.register_surface_mesh("mesh", mesh_faces.vertices, mesh_faces.faces)
    #                    #ps.show()
    #                for tmp_edge in boundary_edges:
    #                    in_face, surface_id = utils.in_faces_2(feature_faces_meshes_pool[-1-j], tmp_edge, plane_normal)
    #                    if in_face:
    #                        tmp_edges.append(tmp_edge)
    #                        surface_ids.append(surface_id)
    #                        print("surface_id", surface_id)
    #                tmp_edges = utils.unify_same_face_edges_2(tmp_edges, surface_ids)
    #                for curve in tmp_edges:
    #                    new_curve = utils.remove_zero_length_edges(curve)
    #                    if len(new_curve) > 0:
    #                        new_line, is_good_line_fit = utils.line_segment_from_points(new_curve)
    #                        if is_good_line_fit:
    #                            section_lines.append(new_line)
    #                        else:
    #                            section_lines.append(new_curve)
    #                        #all_edges.append({"geometry": curve, "type": "section_line"})
    #                #if VERBOSE:
    #                #    utils.plot_curves(tmp_edges, str(feat_id)+"_"+str(plane_normal_id)+"_"+str(ext_line_id)+"_tmp_edges",
    #                #                      color=(0, 1, 0))
    return new_section_lines
    #print("comparison")
    #print(new_section_lines)
    #print(section_lines)
    #if len(new_section_lines) != len(section_lines):
    #    ps.init()
    #    ps.remove_all_structures()
    #    utils.plot_curves(new_section_lines, "new_section_line_", (0, 0, 1))
    #    utils.plot_curves(section_lines, "section_line_", (1, 0, 0))
    #    ps.show()
    #    #exit()

    #return section_lines
