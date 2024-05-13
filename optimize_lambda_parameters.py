import utils
from utils import lineseg_dist
import polyscope as ps
import matplotlib.pyplot as plt
from render_training_data import get_normal_map_single_mesh
from time import time
import pickle
import json
import numpy as np
import os, sys
from get_intermediate_shapes import get_intermediate_shapes
from measure_proximity_labels import count_labels_fast
from opacity_optimization import optimize_opacities
from onshape.call import delete_document
from draw_extrude import draw_extrude
from prepare_decluttering import filter_identical, filter_identical_bvh, prepare_decluttering_v2, extract_strokes_dict
from declutter_gurobi import declutter
from copy import deepcopy
from render_shapes import features_lines_to_svg, typed_feature_lines_to_svg, \
    typed_feature_lines_to_svg_successive, indexed_lines_to_svg
from line_rendering import geometry_match, match_strokes, get_stroke_dataset, subdivide_long_curves, perturbate_sketch, get_opacity_profiles
from pylowstroke.sketch_io import SketchSerializer as skio
sys.setrecursionlimit(10000)

import logging
logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

def optimize_lambda_parameters(per_view_data_folder, strokes, all_edges, lambda_0,
                               stroke_lengths,
                               ellipse_fittings, visibility_scores, constraints,
                               intersection_dag, per_stroke_descendants,
                               cam_pos, obj_center, up_vec, VERBOSE=False):

    lambda_folder = os.path.join(per_view_data_folder, "lambda_optimization")
    if not os.path.exists(lambda_folder):
        os.mkdir(lambda_folder)

    ours_labels = []
    max_feature_id = 0
    for s_id in range(len(all_edges)):
        max_feature_id = max(max_feature_id, np.max([l["feature_id"] for l in all_edges[s_id]["original_labels"]]))
    #print("max_feature_id", max_feature_id)

    for edge_id, edge in enumerate(all_edges):
        tmp_max_feature_id = np.max([l["feature_id"] for l in edge["original_labels"]])
        all_line_types = [l["type"] for l in edge["original_labels"]]
        if tmp_max_feature_id == max_feature_id and "silhouette_line" in all_line_types and edge["visibility_score"] > 0.9:
            ours_labels.append("silhouette")
            continue
        if tmp_max_feature_id == max_feature_id and "feature_line" in all_line_types or "silhouette_line" in all_line_types:
            if edge["visibility_score"] > 0.5:
                ours_labels.append("vis_edge")
            else:
                ours_labels.append("occ_edge")
            continue
        ours_labels.append("scaffold")
    ours_labels = np.array(ours_labels)
    #print("ours_labels", ours_labels)
    if VERBOSE:
        scaffold_file_name = os.path.join(per_view_data_folder, "scaffold_lines.svg")
        indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                        "type": edge["type"],
                                        "feature_id": edge["feature_id"]}
                                       for edge in all_edges]),
                             [i for i, l in enumerate(all_edges)
                              if ours_labels[i] == "scaffold"],
                             cam_pos, obj_center, up_vec,
                             svg_file_name=scaffold_file_name,
                             title="Scaffold lines")
        vis_lines_file_name = os.path.join(per_view_data_folder, "vis_lines.svg")
        indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                        "type": edge["type"],
                                        "feature_id": edge["feature_id"]}
                                       for edge in all_edges]),
                             [i for i, l in enumerate(all_edges)
                              if ours_labels[i] == "vis_edge"],
                             cam_pos, obj_center, up_vec,
                             svg_file_name=vis_lines_file_name,
                             title="Visible feature lines")
        occ_lines_file_name = os.path.join(per_view_data_folder, "occ_lines.svg")
        indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                        "type": edge["type"],
                                        "feature_id": edge["feature_id"]}
                                       for edge in all_edges]),
                             [i for i, l in enumerate(all_edges)
                              if ours_labels[i] == "occ_edge"],
                             cam_pos, obj_center, up_vec,
                             svg_file_name=occ_lines_file_name,
                             title="Occluded feature lines")
        sil_lines_file_name = os.path.join(per_view_data_folder, "sil_lines.svg")
        indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                        "type": edge["type"],
                                        "feature_id": edge["feature_id"]}
                                       for edge in all_edges]),
                             [i for i, l in enumerate(all_edges)
                              if ours_labels[i] == "silhouette"],
                             cam_pos, obj_center, up_vec,
                             svg_file_name=sil_lines_file_name,
                             title="Silhouette lines")

    opensketch_segment = np.array([[0.18558724, 0.80672288, 0.01736725],
                                   [0.91795817, 0.0021275,  0.05278179]])
    max_nb_vis_lines = np.sum(ours_labels == "vis_edge")
    #print("max_nb_vis_lines", max_nb_vis_lines)
    counter = 0
    min_ratio_dist = 100.0
    dists = []
    lambdas = []
    vis_line_ratios = []
    for lambda_1 in np.linspace(0.0, 1.0, 500):
        for lambda_2 in np.linspace(0.0, 1.0, 500):
            lambda_file = os.path.join(lambda_folder, str(counter)+".json")
            print("counter", counter)
            counter += 1
            #print("lambda_1", lambda_1, "lambda_2", lambda_2)
            #if os.path.exists(lambda_file):
            #    continue
            #for lambda_1 in np.linspace(0.0, 1.0, 10):
            #    for lambda_2 in np.linspace(0.0, lambda_1, 10):
            selected_stroke_ids, half_a_constructed, a_constructed, pos_constructed, \
            paths, coplanar_graphs = declutter(strokes,
                                               lambda_0=lambda_0,
                                               lambda_1=lambda_1,
                                               lambda_2=lambda_2,
                                               lambda_3=0.0,
                                               stroke_lengths=stroke_lengths,
                                               ellipse_fittings=ellipse_fittings,
                                               visibility_scores=visibility_scores,
                                               constraints=constraints,
                                               intersection_dag=intersection_dag,
                                               per_stroke_descendants=per_stroke_descendants,
                                               timeout=True)
            if len(selected_stroke_ids) == 0:
                continue
            ratio_dist, ref_ratios, ours_ratios = count_labels_fast([], ours_labels[selected_stroke_ids])
            sketch_pt = np.array([ours_ratios[2], ours_ratios[0], ours_ratios[1]])
            nb_vis_lines = np.sum(ours_labels[selected_stroke_ids] == "vis_edge")
            dist, _ = lineseg_dist(sketch_pt, opensketch_segment[0], opensketch_segment[1])
            #print("ratio_dist", ratio_dist)
            dists.append(dist)
            lambdas.append([lambda_1, lambda_2])
            vis_line_ratios.append(nb_vis_lines/max_nb_vis_lines)
            #if dist+(nb_vis_lines/max_nb_vis_lines) < min_ratio_dist:
            #    min_lambda_1 = lambda_1
            #    min_lambda_2 = lambda_2
            #    min_ratio_dist = dist
            if VERBOSE:
                with open(lambda_file, "w") as fp:
                    json.dump({"lambda_1":lambda_1, "lambda_2":lambda_2,
                               "selected_stroke_ids": selected_stroke_ids.tolist(),
                               "ratio_dist": ratio_dist, "ref_ratios":ref_ratios.tolist(),
                               "ours_ratios": ours_ratios.tolist()}, fp)
    dists = np.array(dists)
    dists -= np.min(dists)
    dists /= np.max(dists)
    vis_line_ratios = np.array(vis_line_ratios)
    #print(vis_line_ratios)
    if not np.isclose(np.min(vis_line_ratios), 1.0):
        vis_line_ratios -= np.min(vis_line_ratios)
        vis_line_ratios /= np.max(vis_line_ratios)
    vis_line_ratios = 1 - vis_line_ratios
    final_dists = dists + vis_line_ratios
    #print(dists)
    #print(vis_line_ratios)
    #print(lambdas)
    min_lambda_1, min_lambda_2 = lambdas[np.argmin(final_dists)]
    min_ratio_dist = np.min(final_dists)
    #print(final_dists)
    print(min_ratio_dist, "min_lambda_1", min_lambda_1, "min_lambda_2", min_lambda_2)
    return min_lambda_1, min_lambda_2


if __name__ == "__main__":

    collect_data = False
    generate_silhouette_lines = False
    recompute_all_construction_lines = False
    declutter_construction_lines = False
    npr_rendering = False
    display = [1000, 1000]
    radius = 1.4
    lambda_0 = 10
    lambda_1 = 2.5
    lambda_1 = 5
    lambda_2 = 5
    lambda_3 = 40
    theta = 60
    phi = -125
    stroke_dataset_designer_name = "student4"
    stylesheet_designer_name = "student4"
    stroke_dataset_designer_name = "Professional6"
    stylesheet_designer_name = "Professional6"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="", type=str, help="Onshape document url")
    parser.add_argument("--data_folder", default="", type=str, help="data_folder path")
    parser.add_argument("--collect_data", default="false", type=str, help="Download data from onshape")
    parser.add_argument("--generate_silhouette_lines", default="false", type=str, help="Generate silhouette lines from SynDraw")
    parser.add_argument("--recompute_all_construction_lines", default="false", type=str, help="Generate all construction lines from CAD-sequence")
    parser.add_argument("--declutter_construction_lines", default="false", type=str, help="Decluttering step")
    parser.add_argument("--npr_rendering", default="false", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--display", default=[512, 512], nargs=2, type=int, help="Image size of rendered normal map and drawing")
    parser.add_argument("--theta", default=60, type=float, help="Spherical coordinate theta of camera position around the object")
    parser.add_argument("--phi", default=125, type=float, help="Spherical coordinate phi of camera position around the object")
    parser.add_argument("--radius", default=1.4, type=float, help="Radius of camera position around the object")
    parser.add_argument("--only_feature_lines", default="false", type=str, help="Only generate intermediate feature lines")
    parser.add_argument("--only_final_npr_lines", default="false", type=str, help="Only generate feature lines from final shape and contour lines")
    parser.add_argument("--include_fillet_lines", default="true", type=str, help="")
    parser.add_argument("--per_view_folder_prefix", default="", type=str, help="")
    parser.add_argument("--verbose", default="true", type=str, help="")
    parser.add_argument("--designer", default="Professional6", type=str, help="For style")
    parser.add_argument("--stylesheet_file", default="", type=str, help="stylesheet file path")
    parser.add_argument("--clean_rendering", default="false", type=str, help="")
    parser.add_argument("--prep_full_sketch", default="false", type=str, help="")
    parser.add_argument("--use_full_sketch", default="false", type=str, help="")
    parser.add_argument("--only_sketch_rendering", default="false", type=str, help="")
    parser.add_argument("--cut_non_visible_points", default="false", type=str, help="")
    parser.add_argument("--lambda_0", default=1.0, type=float, help="")
    parser.add_argument("--lambda_1", default=0.5, type=float, help="")
    parser.add_argument("--lambda_2", default=0.1, type=float, help="")
    parser.add_argument("--cad_sequence_rendering", default="false", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--jitter_cam_pos", default="false", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--ref_labels_file_name", default="", type=str, help="Render drawing in a concept sketch style")
    args = parser.parse_args()
    labels_file_name = args.ref_labels_file_name
    stroke_dataset_designer_name = args.designer
    stylesheet_designer_name = args.designer
    style_sheet_file_name = args.stylesheet_file
    collect_data = args.collect_data == "True" or args.collect_data == "true"
    generate_silhouette_lines = args.generate_silhouette_lines == "True" or args.generate_silhouette_lines == "true"
    recompute_all_construction_lines = args.recompute_all_construction_lines == "True" or args.recompute_all_construction_lines == "true"
    declutter_construction_lines = args.declutter_construction_lines == "True" or args.declutter_construction_lines == "true"
    npr_rendering = args.npr_rendering == "True" or args.npr_rendering == "true"
    include_fillet_lines = args.include_fillet_lines == "True" or args.include_fillet_lines == "true"
    only_feature_lines = args.only_feature_lines == "True" or args.only_feature_lines == "true"
    only_final_npr_lines = args.only_final_npr_lines == "True" or args.only_final_npr_lines == "true"
    verbose = args.verbose == "True" or args.verbose == "true"
    clean_rendering = args.clean_rendering == "True" or args.clean_rendering == "true"
    prep_full_sketch = args.prep_full_sketch == "True" or args.prep_full_sketch == "true"
    use_full_sketch = args.use_full_sketch == "True" or args.use_full_sketch == "true"
    only_sketch_rendering = args.only_sketch_rendering == "True" or args.only_sketch_rendering == "true"
    cad_sequence_rendering = args.cad_sequence_rendering == "True" or args.cad_sequence_rendering == "true"
    cut_non_visible_points = args.cut_non_visible_points == "True" or args.cut_non_visible_points == "true"
    theta = args.theta
    phi = args.phi
    radius = args.radius
    display = [args.display[0], args.display[1]]
    lambda_0 = args.lambda_0
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2

    # create a view-dependent folder
    if args.url != "":
        url = args.url
    if args.data_folder == "":
        obj_id = 0
        while os.path.exists(os.path.join("data", str(obj_id))):
            obj_id += 1
        #obj_id = 184
        data_folder = os.path.join("data", str(obj_id))
    else:
        data_folder = args.data_folder

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    print("Data folder: ", data_folder)
    per_view_data_folder = os.path.join(data_folder, args.per_view_folder_prefix+str(theta)+"_"+str(phi)+"_"+str(radius))
    if not os.path.exists(per_view_data_folder):
        os.mkdir(per_view_data_folder)

    print("lambdas", lambda_0, lambda_1, lambda_2)

    # collect data
    if collect_data:
        print("COLLECT_DATA")
        get_intermediate_shapes(url, data_folder)

    mesh = utils.load_last_mesh(data_folder)

    cam_pos, obj_center = utils.get_cam_pos_obj_center(mesh.vertices, radius=radius, theta=theta, phi=phi)
    up_vec = np.array([0, 0, 1])
    for x in os.listdir(data_folder):
        if "camparam.json" in x:
            with open(os.path.join(data_folder, x), "r") as fp:
                cam_params = json.load(fp)["restricted"]
                #cam_params = json.load(fp)["general"]
                cam_pos = np.array(cam_params["C"]) - obj_center
                up_vec = np.array(cam_params["up"])

    training_data_folder = os.path.join(per_view_data_folder, "training_data")
    if not os.path.exists(training_data_folder):
        os.mkdir(training_data_folder)

    print("DECLUTTER_CONSTRUCTION_LINES")
    #if not recompute_all_construction_lines:
    strokes_dict_file_name = os.path.join(per_view_data_folder, "strokes_dict.json")
    with open(strokes_dict_file_name, "r") as f:
        strokes_dict = json.load(f)
    unique_edges_file_name = os.path.join(per_view_data_folder, "unique_edges.json")
    with open(unique_edges_file_name, "r") as f:
        all_edges = json.load(f)
    strokes = [utils.Stroke(id=s["id"], intersections=s["intersections"], planes=s["planes"], type=s["type"],
                            previous_strokes=s["previous_strokes"], anchor_intersections=s["anchor_intersections"],
                            tangent_intersections=s["tangent_intersections"],
                            overlapping_stroke_ids=s["overlapping_stroke_ids"],
                            projection_constraint_ids=s["projection_constraint_ids"],
                            original_feature_line=s["original_feature_line"],
                            occlusions=s["occlusions"],
                            feature_id=s["feature_id"]) for s in strokes_dict]
    visibility_score_file_name = os.path.join(per_view_data_folder, "visibility_scores")
    with open(visibility_score_file_name, "rb") as fp:
        visibility_scores = np.load(fp)
    #for s_id, s in enumerate(strokes):
    #    all_edges[s_id]["original_feature_line"] = s.original_feature_line

    for s_id, v in enumerate(visibility_scores):
        print(s_id, v)
    constraints_file_name = os.path.join(per_view_data_folder, "constraints.json")
    with open(constraints_file_name, "r") as f:
        constraints = json.load(f)
    intersection_dag_file_name = os.path.join(per_view_data_folder, "intersection_dag.pkl")
    with open(intersection_dag_file_name, "rb") as f:
        intersection_dag = pickle.load(f)
    per_stroke_descendants_file_name = os.path.join(per_view_data_folder, "per_stroke_descendants.json")
    with open(per_stroke_descendants_file_name, "r") as f:
        per_stroke_descendants = json.load(f)


    ellipse_fittings = utils.get_ellipse_fittings(all_edges, cam_pos, obj_center, up_vec)
    if verbose:
        # print individual lines
        individual_lines_folder = os.path.join(per_view_data_folder, "individual_lines")
        if not os.path.exists(individual_lines_folder):
            os.mkdir(individual_lines_folder)
        #for s_id, s in enumerate(all_edges):
        #    if s["type"] == "feature_line" or s["type"] == "sketch" or s["type"] == "silhouette_line":
        #        print("keep line", s_id, s["type"])
        #typed_feature_lines_to_svg_successive(deepcopy([{"geometry": edge["geometry"],
        #                      "type": edge["type"],
        #                      "feature_id": edge["feature_id"]}
        #                     for edge in all_edges]), cam_pos, obj_center, up_vec,
        #                                      os.path.join(individual_lines_folder, "line.svg"))
        #print("Plotted individual_lines")

    stroke_lengths = np.array([utils.polyline_length(l["geometry"]) for l in all_edges])
    stroke_lengths /= np.max(stroke_lengths)
    #print("stroke_lengths", stroke_lengths)
    with open(labels_file_name, "r") as fp:
        labels_opensketch = json.load(fp)["strokes_line_types"]
    print("Decluttering ...")
    ours_labels = []
    max_feature_id = 0
    for s_id in range(len(all_edges)):
        max_feature_id = max(max_feature_id, np.max([l["feature_id"] for l in all_edges[s_id]["original_labels"]]))
    for edge_id, edge in enumerate(all_edges):
        tmp_max_feature_id = np.max([l["feature_id"] for l in edge["original_labels"]])
        all_line_types = [l["type"] for l in edge["original_labels"]]
        if tmp_max_feature_id == max_feature_id and "silhouette_line" in all_line_types:
            ours_labels.append("silhouette")
            continue
        if tmp_max_feature_id == max_feature_id and "feature_line" in all_line_types:
            if edge["visibility_score"] > 0.5:
                ours_labels.append("vis_edge")
            else:
                ours_labels.append("occ_edge")
            continue
        ours_labels.append("scaffold")
    ours_labels = np.array(ours_labels)
    scaffold_file_name = os.path.join(per_view_data_folder, "scaffold_lines.svg")
    indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                    "type": edge["type"],
                                    "feature_id": edge["feature_id"]}
                                   for edge in all_edges]),
                         [i for i, l in enumerate(all_edges)
                          if ours_labels[i] == "scaffold"],
                         cam_pos, obj_center, up_vec,
                         svg_file_name=scaffold_file_name,
                         title="Scaffold lines")
    vis_lines_file_name = os.path.join(per_view_data_folder, "vis_lines.svg")
    indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                    "type": edge["type"],
                                    "feature_id": edge["feature_id"]}
                                   for edge in all_edges]),
                         [i for i, l in enumerate(all_edges)
                          if ours_labels[i] == "vis_edge"],
                         cam_pos, obj_center, up_vec,
                         svg_file_name=vis_lines_file_name,
                         title="Visible feature lines")
    occ_lines_file_name = os.path.join(per_view_data_folder, "occ_lines.svg")
    indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                    "type": edge["type"],
                                    "feature_id": edge["feature_id"]}
                                   for edge in all_edges]),
                         [i for i, l in enumerate(all_edges)
                          if ours_labels[i] == "occ_edge"],
                         cam_pos, obj_center, up_vec,
                         svg_file_name=occ_lines_file_name,
                         title="Occluded feature lines")
    sil_lines_file_name = os.path.join(per_view_data_folder, "sil_lines.svg")
    indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                    "type": edge["type"],
                                    "feature_id": edge["feature_id"]}
                                   for edge in all_edges]),
                         [i for i, l in enumerate(all_edges)
                          if ours_labels[i] == "silhouette"],
                         cam_pos, obj_center, up_vec,
                         svg_file_name=sil_lines_file_name,
                         title="Silhouette lines")
    exit()

    labels_opensketch = np.array(labels_opensketch)
    min_ratio_dist = 10.0
    min_lambda_1 = 0.0
    min_lambda_2 = 0.0
    lambda_folder = os.path.join(per_view_data_folder, "lambda_optimization")
    if not os.path.exists(lambda_folder):
        os.mkdir(lambda_folder)

    counter = 0
    for lambda_1 in np.linspace(0.0, 1.0, 100):
        for lambda_2 in np.linspace(0.0, 1.0, 100):
            lambda_file = os.path.join(lambda_folder, str(counter)+".json")
            counter += 1
            print("lambda_1", lambda_1, "lambda_2", lambda_2)
            #if os.path.exists(lambda_file):
            #    continue
            #for lambda_1 in np.linspace(0.0, 1.0, 10):
            #    for lambda_2 in np.linspace(0.0, lambda_1, 10):
            selected_stroke_ids, half_a_constructed, a_constructed, pos_constructed, \
            paths, coplanar_graphs = declutter(strokes,
                                               lambda_0=lambda_0,
                                               lambda_1=lambda_1,
                                               lambda_2=lambda_2,
                                               lambda_3=lambda_3,
                                               stroke_lengths=stroke_lengths,
                                               ellipse_fittings=ellipse_fittings,
                                               visibility_scores=visibility_scores,
                                               constraints=constraints,
                                               intersection_dag=intersection_dag,
                                               per_stroke_descendants=per_stroke_descendants)
            ratio_dist, ref_ratios, ours_ratios = count_labels_fast(labels_opensketch, ours_labels[selected_stroke_ids])
            print("ratio_dist", ratio_dist)
            if ratio_dist < min_ratio_dist:
                min_lambda_1 = lambda_1
                min_lambda_2 = lambda_2
                min_ratio_dist = ratio_dist
            with open(lambda_file, "w") as fp:
                json.dump({"lambda_1":lambda_1, "lambda_2":lambda_2,
                           "selected_stroke_ids": selected_stroke_ids.tolist(),
                           "ratio_dist": ratio_dist, "ref_ratios":ref_ratios.tolist(),
                           "ours_ratios": ours_ratios.tolist()}, fp)
    print(min_ratio_dist, "min_lambda_1", min_lambda_1, "min_lambda_2", min_lambda_2)
    exit()
    #ps.init()
    ##utils.plot_curves([e["geometry"] for e_id, e in enumerate(all_edges)], color=(0, 0, 0))
    #curves = [e["geometry"] for e_id, e in enumerate(all_edges)]
    #for curve_id, curve_geom in enumerate(curves):
    #    if np.isclose(visibility_scores[curve_id], 0.0):
    #        continue
    #    if len(curve_geom) == 1:
    #        edges_array = np.array([[0, 0]])
    #    else:
    #        edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
    #    ps_c = ps.register_curve_network(str(curve_id), nodes=np.array(curve_geom),
    #                              edges=edges_array, color=(0, 0, 0))
    #    ps_c.add_scalar_quantity("visibility_score", np.repeat(visibility_scores[curve_id], len(edges_array)),defined_on="edges", enabled=True, vminmax=(0, 1))

    #ps.show()
    #ps.remove_all_structures()
    #utils.plot_curves([e["geometry"] for e_id, e in enumerate(all_edges) if e_id in selected_stroke_ids], color=(0, 0, 0))
    #ps.show()
    #exit()

    # extract final edges
    final_edges = [edge for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
    for edge_id in range(len(selected_stroke_ids)):
        final_edges[edge_id]["id"] = selected_stroke_ids[edge_id]
    decluttered_strokes_file_name = os.path.join(per_view_data_folder, "decluttered_lambda0_"+str(lambda_0)+".json")
    final_edges_dict = {}
    last_edge_id = 0
    for edge_id, edge in enumerate(final_edges):
        #final_edges_dict[str(edge_id)] = [int(selected_stroke_ids[edge_id]), list(edge["geometry"])]
        if cut_non_visible_points:
            visible_segments = utils.cut_non_visible_points(edge, cam_pos, mesh, obj_center, up_vec)
            for vis_seg in visible_segments:
                final_edges_dict[str(last_edge_id)] = {"id": int(selected_stroke_ids[edge_id]),
                                                       "geometry_3d": vis_seg.tolist(),
                                                       "line_type": edge["type"]}
                last_edge_id += 1
        else:
            final_edges_dict[str(edge_id)] = edge
            final_edges_dict[str(edge_id)]["id"] = int(selected_stroke_ids[edge_id])
            final_edges_dict[str(edge_id)]["geometry_3d"] = list(edge["geometry"])
            final_edges_dict[str(edge_id)]["line_type"] = edge["type"]
            final_edges_dict[str(edge_id)]["visibility_score"] = edge["visibility_score"]
            #final_edges_dict[str(edge_id)] = {"id": int(selected_stroke_ids[edge_id]),
            #                                  "geometry_3d": list(edge["geometry"]),
            #                                  "line_type": edge["type"]}


    with open(decluttered_strokes_file_name, "w") as fp:
        json.dump(final_edges_dict, fp, indent=4)
    print("Saved ", decluttered_strokes_file_name)
    if verbose:
        svg_file_name = os.path.join(per_view_data_folder, "decluttered_lambda0_"+str(lambda_0)+".svg")
        typed_feature_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                              "type": edge["type"],
                              "feature_id": edge["feature_id"]}
                             for edge in final_edges]),
                                   cam_pos, obj_center, up_vec,
                                   svg_file_name=svg_file_name,
                                   title="Final drawing")
        print("Plotted ", svg_file_name)
        individual_lines_folder = os.path.join(per_view_data_folder, "individual_lines_result")
        if not os.path.exists(individual_lines_folder):
            os.mkdir(individual_lines_folder)
        typed_feature_lines_to_svg_successive(deepcopy([{"geometry": edge["geometry"],
                              "type": edge["type"],
                              "feature_id": edge["feature_id"]}
                             for edge in final_edges]),
                                              cam_pos, obj_center, up_vec,
                                              os.path.join(individual_lines_folder, "line.svg"))
        print("Plotted individual_lines")
    # Final outline plot
    feat_line_id = 0
    last_feat_lines_file_name = os.path.join(data_folder, "feature_lines_"+str(feat_line_id)+".json")
    while os.path.exists(last_feat_lines_file_name):
        feat_line_id += 1
        last_feat_lines_file_name = os.path.join(data_folder, "feature_lines_"+str(feat_line_id)+".json")
    feat_line_id -= 1
    last_feat_lines_file_name = os.path.join(data_folder, "feature_lines_"+str(feat_line_id)+".json")
    with open(last_feat_lines_file_name, "r") as f:
        feat_lines = json.load(f)

    #for l in feat_lines.values():
    #    final_edges.append({"geometry": l, "type": "outline"})
    #    final_edges_dict[str(len(final_edges)-1)] = {"id": int(len(final_edges))-1,
    #                                                 "geometry_3d": list(l),
    #                                                 "line_type": "outline"}

    if verbose:
        final_drawing_file_name = os.path.join(per_view_data_folder, "final_drawing.svg")
        #indexed_lines_to_svg(deepcopy(final_edges),
        #                     [0], svg_file_name=final_drawing_file_name,
        #                     theta=theta, phi=phi, title="Outlined drawing")
        indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                        "type": edge["type"]}
                                       for edge in final_edges]),
                             [i for i, l in enumerate(final_edges)
                              if l["type"] == "lol"],
                             cam_pos, obj_center, up_vec,
                             svg_file_name=final_drawing_file_name,
                             title="Outlined drawing")
        os.system("rsvg-convert -f pdf " + final_drawing_file_name + " > " + final_drawing_file_name.replace("svg", "pdf"))
    final_edges_file_name = os.path.join(per_view_data_folder, "final_edges.json")
    with open(final_edges_file_name, "w") as fp:
        json.dump(final_edges_dict, fp, indent=4)
