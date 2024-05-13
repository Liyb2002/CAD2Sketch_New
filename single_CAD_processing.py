from distutils.command.clean import clean
from pyrr import Matrix44
import utils
from optimize_lambda_parameters import optimize_lambda_parameters
import polyscope as ps
import matplotlib.pyplot as plt
from render_training_data import get_normal_map_single_mesh
from time import time
import pickle
import json
import numpy as np
import os, sys
from get_intermediate_shapes import get_intermediate_shapes
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
from get_best_viewpoint import get_best_viewpoint
sys.setrecursionlimit(10000)
import shutil

import logging
logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

SYN_DRAW_PATH = "../contour-detect/SynDraw/build"

if __name__ == "__main__":

    collect_data = False
    generate_silhouette_lines = True
    recompute_all_construction_lines = True
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
    stroke_dataset_designer_name = "Professional6"
    stylesheet_designer_name = "Professional6"


    mesh = utils.load_last_mesh(data_folder)
    cam_pos, obj_center = utils.get_cam_pos_obj_center(mesh.vertices, radius=radius, theta=theta, phi=phi)


    up_vec = np.array([0, 0, 1])
    #up_vec = np.array([0, 1, 0])
    for x in os.listdir(data_folder):
        if "camparam.json" in x:
            with open(os.path.join(data_folder, x), "r") as fp:
                cam_params = json.load(fp)["restricted"]
                #cam_params = json.load(fp)["general"]
                cam_pos = np.array(cam_params["C"]) - obj_center
                up_vec = np.array(cam_params["up"])
    #up_vec = np.array([0.51092751686519045,0.6654680932233632,-0.54415557463985587])

    #ps.init()
    #ps.register_surface_mesh("mouse", mesh.vertices, mesh.faces)
    #ps.register_point_cloud("cam_pos", np.array([cam_pos]))
    #ps.show()
    
    # abc-data sanity check
    #if not utils.abc_sanity_check(data_folder):
    #    exit()

    training_data_folder = os.path.join(per_view_data_folder, "training_data")
    if only_final_npr_lines:
        training_data_folder = os.path.join(per_view_data_folder, "training_data_only_final_npr_lines")
    #if only_feature_lines:
    #    training_data_folder = os.path.join(per_view_data_folder, "training_data_only_feature_lines")
    #if only_final_npr_lines:
    #    training_data_folder = os.path.join(per_view_data_folder, "training_data_only_final_npr_lines")
    if not os.path.exists(training_data_folder):
        os.mkdir(training_data_folder)
    if generate_silhouette_lines:
        print("GENERATE_SILHOUETTE_LINES")
        # go through all curved surfaces and generate silhouette lines
        curved_surfaces_per_feature_id = utils.get_curved_surfaces(data_folder)
        #if not declutter_construction_lines:
        #    final_edges_file_name = os.path.join(per_view_data_folder, "final_edges.json")
        #    with open(final_edges_file_name, "r") as fp:
        #        final_edges_dict = json.load(fp)
        silhouette_lines_per_feature_id = utils.generate_silhouette_lines(
            curved_surfaces_per_feature_id, SYN_DRAW_PATH, cam_pos, obj_center, up_vec,
        VERBOSE=False)
        silhouette_file_name = os.path.join(per_view_data_folder, "silhouette_line_per_feature_id.json")
        with open(silhouette_file_name, "w") as fp:
            json.dump(silhouette_lines_per_feature_id, fp)

    # construction lines
    strokes_dict_file_name = os.path.join(per_view_data_folder, "strokes_dict.json")
    if recompute_all_construction_lines:
        start_time = time()
        print("RECOMPUTE_ALL_CONSTRUCTION_LINES")
        if not generate_silhouette_lines:

            silhouette_file_name = os.path.join(per_view_data_folder, "silhouette_line_per_feature_id.json")
            with open(silhouette_file_name, "r") as fp:
                silhouette_lines_per_feature_id = json.load(fp)

        # get all edges
        #print("Compute all edges")
        #print(silhouette_lines_per_feature_id)
        path = './draw_extrude.lock'
        if os.path.exists(path):
            shutil.rmtree(path)

        while os.system("mkdir ./draw_extrude.lock >> /dev/null 2>&1") !=0:
            print("waiting")
        #try:
        all_edges = draw_extrude(data_folder, silhouette_lines_per_feature_id,
                                 only_final_npr_lines=only_final_npr_lines, include_fillet_lines=include_fillet_lines)
        final_npr_lines = draw_extrude(data_folder, silhouette_lines_per_feature_id,
                                       only_final_npr_lines=True, include_fillet_lines=False)
        max_feature_id = np.max([edge["feature_id"] for edge in all_edges])
        for edge_id in range(len(final_npr_lines)):
            final_npr_lines[edge_id]["feature_id"] = int(max_feature_id)+1
        all_edges += final_npr_lines
        os.system("rm -r ./draw_extrude.lock")
        #except:
        #    os.system("rm -r ./draw_extrude.lock")
        #    exit()
        #for edge_id, edge in enumerate(all_edges):
        #    if edge["fitted_curve"].NbKnots() < 2:
        #        print(edge_id)
        #        print(edge["fitted_curve"].NbKnots())
        #all_edges = draw_extrude(data_folder, silhouette_lines_per_feature_id, VERBOSE=True, only_final_npr_lines=only_final_npr_lines)
        #for s in all_edges:
        #    if s["type"] == "silhouette_line":
        #        print(s)
        print(len(all_edges))
        all_edges_file_name = os.path.join(per_view_data_folder, "all_edges.json")
        for edge_id in range(len(all_edges)):
            all_edges[edge_id]["geometry"] = np.array(all_edges[edge_id]["geometry"]).tolist()
        with open(all_edges_file_name, "w") as f:
            json.dump([{"geometry": edge["geometry"],
                        "type": edge["type"],
                        "feature_id": edge["feature_id"]}
                        for edge in all_edges], f, indent=4)
        print("Filter identical edges")
        #all_edges = filter_identical(all_edges, only_feature_lines or only_final_npr_lines)
        individual_lines_folder = os.path.join(per_view_data_folder, "individual_lines")
        if not os.path.exists(individual_lines_folder):
            os.mkdir(individual_lines_folder)
        #typed_feature_lines_to_svg_successive(deepcopy([{"geometry": edge["geometry"],
        #                                                 "type": edge["type"],
        #                                                 "feature_id": edge["feature_id"]}
        #                                                for edge in all_edges]), cam_pos, obj_center,
        #                                      os.path.join(individual_lines_folder, "line.svg"))
        all_edges = filter_identical_bvh(all_edges, only_feature_lines or only_final_npr_lines)
        print(len(all_edges))
        #exit()
        utils.add_visibility_label(all_edges, cam_pos+obj_center, mesh, obj_center, up_vec)
        unique_edges_file_name = os.path.join(per_view_data_folder, "unique_edges.json")
        unique_edges = deepcopy([{"geometry": edge["geometry"],
                                  "type": edge["type"],
                                  "feature_id": edge["feature_id"],
                                  "original_labels": edge["original_labels"],
                                  "visibility_score": edge["visibility_score"]}
                                 for edge in all_edges])
        for edge_id in range(len(unique_edges)):
            unique_edges[edge_id]["geometry"] = np.array(unique_edges[edge_id]["geometry"]).tolist()
        print("line generation time", time() - start_time)
        with open(unique_edges_file_name, "w") as f:
            json.dump(unique_edges, f, indent=4)
        if verbose:
            all_strokes_file_name = os.path.join(per_view_data_folder, "all_lines.svg")
            typed_feature_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                  "type": edge["type"],
                                  "feature_id": edge["feature_id"]}
                                 for edge in all_edges]), cam_pos, obj_center, up_vec,
                                       svg_file_name=all_strokes_file_name,
                                       title="All lines")
            print("Plotted ", all_strokes_file_name)
            os.system("rsvg-convert -f pdf " + all_strokes_file_name + " > " + all_strokes_file_name.replace("svg", "pdf"))
            extrude_lines_file_name = os.path.join(per_view_data_folder, "extrude_lines.svg")
            indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                  "type": edge["type"],
                                  "feature_id": edge["feature_id"]}
                                 for edge in all_edges]),
                                 [i for i, l in enumerate(all_edges)
                                  if l["type"] == "extrude_line"],
                                 cam_pos, obj_center, up_vec,
                                 svg_file_name=extrude_lines_file_name,
                                 title="Extrude lines")
            print("Plotted ", extrude_lines_file_name)
            os.system("rsvg-convert -f pdf " + extrude_lines_file_name + " > " + extrude_lines_file_name.replace("svg", "pdf"))
            silhouette_lines_file_name = os.path.join(per_view_data_folder, "silhouette_lines.svg")
            indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                  "type": edge["type"],
                                  "feature_id": edge["feature_id"]}
                                 for edge in all_edges]),
                                 [i for i, l in enumerate(all_edges)
                                  if l["type"] == "silhouette_line"],
                                 cam_pos, obj_center, up_vec,
                                 svg_file_name=silhouette_lines_file_name)
            print("Plotted ", silhouette_lines_file_name)
            os.system("rsvg-convert -f pdf " + silhouette_lines_file_name + " > " + silhouette_lines_file_name.replace("svg", "pdf"))
        with open(os.path.join(data_folder, "parsed_features.json"), "r") as f:
            parsed_features = json.load(f)
        #for s_id, s in enumerate(all_edges):
        #    if s["type"] == "feature_line" or s["type"] == "sketch" or s["type"] == "silhouette_line":
        #        print("keep line", s_id, s["type"])
        if verbose:
            typed_feature_lines_to_svg_successive(deepcopy([{"geometry": edge["geometry"],
                                                             "type": edge["type"],
                                                             "feature_id": edge["feature_id"]}
                                                            for edge in all_edges]), cam_pos, obj_center, up_vec,
                                                  os.path.join(individual_lines_folder, "line.svg"))
        #print("Plotted individual_lines")
        print("Prepare decluttering")
        #start_time = time()
        #old_strokes = prepare_decluttering(all_edges, cam_pos, obj_center,
        #                               parsed_features=parsed_features, VERBOSE=False)
        #print("old time", time() - start_time)
        start_time = time()
        strokes, intersections, constraints = prepare_decluttering_v2(all_edges, cam_pos, obj_center, up_vec,
                                                                      parsed_features=parsed_features, VERBOSE=False)
        print("prepare decluttering time", time() - start_time)
        #exit()
        #for s in strokes:
        #    if s.type == "silhouette_line":
        #        print("keep line", s.id)
        strokes_dict = extract_strokes_dict(strokes)
        if verbose:
            projection_constraint_lines_file_name = os.path.join(per_view_data_folder, "projection_constraint_lines.svg")
            indexed_lines_to_svg(deepcopy([{"geometry": edge["geometry"],
                                  "type": edge["type"],
                                  "feature_id": edge["feature_id"]}
                                 for edge in all_edges]),
                                 [i for i, l in enumerate(strokes)
                                  if len(l.projection_constraint_ids) > 0],
                                 cam_pos, obj_center, up_vec,
                                 svg_file_name=projection_constraint_lines_file_name,
                                 title="Projection constraint lines")
            print("Plotted ", projection_constraint_lines_file_name)
        visibility_scores = utils.compute_visibility_score(all_edges, cam_pos+obj_center, mesh, obj_center, up_vec, VERBOSE=False)
        visibility_score_file_name = os.path.join(per_view_data_folder, "visibility_scores")
        with open(visibility_score_file_name, "wb") as fp:
            np.save(fp, visibility_scores)
        occlusions = utils.compute_occlusions(all_edges, cam_pos, obj_center, up_vec, strokes_dict, intersections,
                                              VERBOSE=False)
        occlusions_file_name = os.path.join(per_view_data_folder, "occlusions")
        with open(occlusions_file_name, "wb") as fp:
            np.save(fp, occlusions)
        with open(strokes_dict_file_name, "w") as f:
            json.dump(strokes_dict, f, indent=4)
        constraints_file_name = os.path.join(per_view_data_folder, "constraints.json")
        with open(constraints_file_name, "w") as f:
            #print(constraints)
            json.dump(constraints, f, indent=4)
        print("Saved ", strokes_dict_file_name)

        intersection_dag, per_stroke_descendants = utils.intersection_dag(strokes_dict)
        intersection_dag_file_name = os.path.join(per_view_data_folder, "intersection_dag.pkl")
        with open(intersection_dag_file_name, "wb") as f:
            pickle.dump(intersection_dag, f)
        per_stroke_descendants_file_name = os.path.join(per_view_data_folder, "per_stroke_descendants.json")
        with open(per_stroke_descendants_file_name, "w") as f:
            json.dump(per_stroke_descendants, f)

    if prep_full_sketch:
        if not recompute_all_construction_lines:
            unique_edges_file_name = os.path.join(per_view_data_folder, "unique_edges.json")
            with open(unique_edges_file_name, "r") as f:
                all_edges = json.load(f)
        #style_sheet_file_name = os.path.join("data/stylesheets/"+stylesheet_designer_name+".json")
        with open(style_sheet_file_name, "r") as fp:
            stylesheet = json.load(fp)

        opacity_profiles_name = os.path.join("data/opacity_profiles", stroke_dataset_designer_name+".json")
        if os.path.exists(opacity_profiles_name):
            with open(opacity_profiles_name, "r") as fp:
                opacity_profiles = json.load(fp)
        else:
            opacity_profiles = get_opacity_profiles(stroke_dataset_designer_name)
            with open(opacity_profiles_name, "w") as fp:
                json.dump(opacity_profiles, fp)

        stroke_dataset_name = os.path.join("data/stroke_datasets", stroke_dataset_designer_name+".pkl")
        if os.path.exists(stroke_dataset_name):
            with open(stroke_dataset_name, "rb") as fp:
                stroke_dataset = pickle.load(fp)
        else:
            stroke_dataset = get_stroke_dataset(stroke_dataset_designer_name)
            for s in stroke_dataset:
                if s.is_ellipse():
                    continue
            with open(stroke_dataset_name, "wb") as fp:
                pickle.dump(stroke_dataset, fp)
        final_edges_dict = {}
        for edge_id, edge in enumerate(all_edges):
            final_edges_dict[str(edge_id)] = edge
            final_edges_dict[str(edge_id)]["id"] = edge_id
            final_edges_dict[str(edge_id)]["geometry_3d"] = list(edge["geometry"])
            final_edges_dict[str(edge_id)]["line_type"] = edge["type"]
            final_edges_dict[str(edge_id)]["visibility_score"] = edge["visibility_score"]
            #final_edges_dict[str(edge_id)] = {"id": int(edge_id),
            #                                  "geometry_3d": list(edge["geometry"]),
            #                                  "line_type": edge["type"]}

        new_opacities = optimize_opacities(final_edges_dict, stylesheet, cam_pos, obj_center, up_vec, mesh, VERBOSE=False)
        syn_sketch = geometry_match(final_edges_dict, stylesheet,
                                    cam_pos, obj_center, up_vec, display,
                                    clean_rendering=clean_rendering)
        for s_id in range(len(syn_sketch.strokes)):
            for p_id in range(len(syn_sketch.strokes[s_id].points_list)):
                syn_sketch.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])

        syn_sketch = perturbate_sketch(syn_sketch)

        npr_sketch = match_strokes(syn_sketch, stroke_dataset, opacity_profiles,
                                   opacity_threshold=0.1,
                                   straight_line_nearest_neighbor_range=[0, 10])

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                            bottom=0.0,
                            top=1.0)
        npr_sketch.display_strokes_2(fig, ax,
                                     linewidth_data=lambda p: p.get_data("pressure")+0.5,
                                     color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
                                                              for p in s.points_list])
        ax.set_xlim(0, display[0])
        ax.set_ylim(display[1], 0)
        ax.set_aspect("equal")
        plt.gca().invert_yaxis()
        ax.axis("off")
        fig.set_size_inches(display[0]/100, display[1]/100)
        npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_full.png")
        npr_sketch_file_name = os.path.join(per_view_data_folder, "npr_sketch_full.svg")
        skio.save(npr_sketch, npr_sketch_file_name)
        plt.savefig(npr_sketch_name, dpi=100)
        plt.close(fig)
        with open(os.path.join(training_data_folder, "npr_sketch_full.pkl"), "wb") as fp:
            pickle.dump(npr_sketch, fp)

    if declutter_construction_lines:
        start_time = time()
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

        #for s_id, v in enumerate(visibility_scores):
        #    print(s_id, v)
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
        print("Decluttering ...")
        if keep_all_lines:
            selected_stroke_ids = [s.id for s_id, s in enumerate(strokes)]
        elif only_feature_lines or only_final_npr_lines:
            selected_stroke_ids = [s.id for s_id, s in enumerate(strokes)
                                   if s.type == "feature_line" or s.type == "silhouette_line"]
        else:
            if find_best_lambdas:
                lambda_1, lambda_2 = optimize_lambda_parameters(
                    per_view_data_folder, strokes, all_edges, lambda_0,
                    stroke_lengths, ellipse_fittings, visibility_scores,
                    constraints, intersection_dag, per_stroke_descendants,
                    cam_pos, obj_center, up_vec, VERBOSE=verbose)

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
            if verbose:
                print("selected_stroke_ids", selected_stroke_ids)
                #print("len(selected_stroke_ids)", len(selected_stroke_ids))
                #for graph in coplanar_graphs:
                #    graph_folder = os.path.join(data_folder, "coplanar_graph"+str(graph[0])+"_"+str(graph[1]))
                #    if not os.path.exists(graph_folder):
                #        os.mkdir(graph_folder)
                #    graph_file_name = os.path.join(graph_folder, "coplanar_graph"+str(graph[0])+".svg")
                #    indexed_lines_to_svg(deepcopy(all_edges), graph[2], svg_file_name=graph_file_name,
                #                         theta=theta, phi=phi, title="Final drawing")
                #    for edge_id, edge in enumerate(graph[3]):
                #        graph_file_name = os.path.join(graph_folder, "edge"+str(edge[0])+"_"+str(edge[1])+".svg")
                #        indexed_lines_to_svg(deepcopy(all_edges), edge, svg_file_name=graph_file_name,
                #                             theta=theta, phi=phi)

                #pos_constructed [10 12 13 30 31 32 33 48 55 58 59 60 61 62 63 64 65 66 67 68 69]
                #pos_constructed [10 12 13 30 31 32 33 37 41 58 59 60 61 62 63 64 65 66 67 68 69]

                print("a_constructed", a_constructed)
                print("pos_constructed", pos_constructed)
                print(paths)
        print("declutter_time", time() - start_time)
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

        # a hidden silhouette line is a feature line (for the rendering)
        for edge_id, edge in enumerate(all_edges):
            if edge["type"] == "silhouette_line" and edge["visibility_score"] < 0.9:
                all_edges[edge_id]["type"] = "feature_line"

        # extract final edges
        final_edges = [edge for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
        for edge_id in range(len(selected_stroke_ids)):
            final_edges[edge_id]["id"] = selected_stroke_ids[edge_id]

        decluttered_strokes_file_name = os.path.join(per_view_data_folder, "decluttered_lambda0_"+str(lambda_0)+".json")
        final_edges_dict = {}
        last_edge_id = 0
        #ps.init()
        #ps.remove_all_structures()
        for edge_id, edge in enumerate(final_edges):
            #final_edges_dict[str(edge_id)] = [int(selected_stroke_ids[edge_id]), list(edge["geometry"])]
            if cut_non_visible_points:
                visible_segments = utils.cut_non_visible_points(edge, cam_pos+obj_center, mesh, obj_center, up_vec)
                for vis_seg in visible_segments:
                    final_edges_dict[str(last_edge_id)] = {"id": int(selected_stroke_ids[edge_id]),
                                                           "geometry_3d": vis_seg.tolist(),
                                                           "line_type": edge["type"],
                                                           "visibility_score": 1.0,
                                                           "original_labels": edge["original_labels"],
                                                           "feature_id": edge["feature_id"],
                                                           #"geometry": edge["geometry"],
                                                           "type": edge["type"]
                                                           }
                    last_edge_id += 1
                utils.plot_curves(visible_segments, name_prefix=str(edge_id)+"_")
            else:
                final_edges_dict[str(edge_id)] = edge
                final_edges_dict[str(edge_id)]["id"] = int(selected_stroke_ids[edge_id])
                final_edges_dict[str(edge_id)]["geometry_3d"] = list(edge["geometry"])
                final_edges_dict[str(edge_id)]["line_type"] = edge["type"]
                final_edges_dict[str(edge_id)]["visibility_score"] = edge["visibility_score"]
                #final_edges_dict[str(edge_id)] = {"id": int(selected_stroke_ids[edge_id]),
                #                                  "geometry_3d": list(edge["geometry"]),
                #                                  "line_type": edge["type"]}
        #ps.set_ground_plane_mode("shadow_only")
        #ps.set_navigation_style("free")
        #ps.set_up_dir("neg_z_up")
        #ps.look_at_dir(camera_location=cam_pos+obj_center,
        #           target=obj_center,
        #           up_dir=np.array([0, 0, 1]))

        #ps.register_surface_mesh("mesh", mesh.vertices, mesh.faces)
        #ps.show()


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
            #individual_lines_folder = os.path.join(per_view_data_folder, "individual_lines_result")
            #if not os.path.exists(individual_lines_folder):
            #    os.mkdir(individual_lines_folder)
            #typed_feature_lines_to_svg_successive(deepcopy([{"geometry": edge["geometry"],
            #                      "type": edge["type"],
            #                      "feature_id": edge["feature_id"]}
            #                     for edge in final_edges]),
            #                                      cam_pos, obj_center, up_vec,
            #                                      os.path.join(individual_lines_folder, "line.svg"))
            #print("Plotted individual_lines")
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

    if only_sketch_rendering:
        with open(os.path.join(training_data_folder, "npr_sketch_full.pkl"), "rb") as fp:
            npr_sketch = pickle.load(fp)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                            bottom=0.0,
                            top=1.0)
        npr_sketch.display_strokes_2(fig, ax,
                                     linewidth_data=lambda p: p.get_data("pressure")+0.5,
                                     color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
                                                              for p in s.points_list],
                                     display_strokes=selected_stroke_ids)
        ax.set_xlim(0, display[0])
        ax.set_ylim(display[1], 0)
        ax.set_aspect("equal")
        plt.gca().invert_yaxis()
        ax.axis("off")
        fig.set_size_inches(display[0]/100, display[1]/100)
        npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_lambda0_"+str(lambda_0)+"_lambda1_"+str(lambda_1)+"_lambda2_"+str(lambda_2)+".png")
        plt.savefig(npr_sketch_name, dpi=100)
        plt.close(fig)

    if baseline_1 or baseline_0:

        print("BASELINE_1")
        #if not recompute_all_construction_lines:
        strokes_dict_file_name = os.path.join(per_view_data_folder, "strokes_dict.json")
        with open(strokes_dict_file_name, "r") as f:
            strokes_dict = json.load(f)
        unique_edges_file_name = os.path.join(per_view_data_folder, "unique_edges.json")
        with open(unique_edges_file_name, "r") as f:
            all_edges = json.load(f)
        
        # select all feature and silhouette lines
        selected_stroke_ids = [s_id for s_id, s in enumerate(all_edges)
                               if np.any([l["type"] in ["feature_line", "silhouette_line"] for l in s["original_labels"]])]
        if baseline_0:
            max_feature_id = np.max([np.max([l["feature_id"] for l in edge["original_labels"]]) for edge in all_edges])
            #max_feature_id = np.max([edge["feature_id"] for edge in all_edges])
            selected_stroke_ids = []
            for s_id, s in enumerate(all_edges):
                for l in s["original_labels"]:
                    if l["type"] in ["feature_line", "silhouette_line"] and l["feature_id"] == max_feature_id:
                        selected_stroke_ids.append(s_id)
            selected_stroke_ids = np.unique(selected_stroke_ids)
            #selected_stroke_ids = [s_id for s_id, s in enumerate(all_edges)
            #                       if s["feature_id"] == max_feature_id and 
            #                       np.any([l["type"] in ["feature_line", "silhouette_line"] for l in s["original_labels"]])]

        for edge_id, edge in enumerate(all_edges):
            if edge["type"] == "silhouette_line" and edge["visibility_score"] < 0.9:
                all_edges[edge_id]["type"] = "feature_line"

        # extract final edges
        final_edges = [edge for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
        for edge_id in range(len(selected_stroke_ids)):
            final_edges[edge_id]["id"] = selected_stroke_ids[edge_id]

        final_edges_dict = {}
        last_edge_id = 0
        for edge_id, edge in enumerate(final_edges):
            #final_edges_dict[str(edge_id)] = [int(selected_stroke_ids[edge_id]), list(edge["geometry"])]
            final_edges_dict[str(edge_id)] = edge
            final_edges_dict[str(edge_id)]["id"] = int(selected_stroke_ids[edge_id])
            final_edges_dict[str(edge_id)]["geometry_3d"] = list(edge["geometry"])
            final_edges_dict[str(edge_id)]["line_type"] = edge["type"]
            final_edges_dict[str(edge_id)]["visibility_score"] = edge["visibility_score"]
            #final_edges_dict[str(edge_id)] = {"id": int(selected_stroke_ids[edge_id]),
            #                                  "geometry_3d": list(edge["geometry"]),
            #                                  "line_type": edge["type"]}
        
        #style_sheet_file_name = os.path.join("data/stylesheets/"+stylesheet_designer_name+".json")
        with open(style_sheet_file_name, "r") as fp:
            stylesheet = json.load(fp)

        stroke_dataset_name = os.path.join("data/stroke_datasets", stroke_dataset_designer_name+".pkl")
        if os.path.exists(stroke_dataset_name):
            with open(stroke_dataset_name, "rb") as fp:
                stroke_dataset = pickle.load(fp)
        else:
            stroke_dataset = get_stroke_dataset(stroke_dataset_designer_name)
            for s in stroke_dataset:
                if s.is_ellipse():
                    continue
            with open(stroke_dataset_name, "wb") as fp:
                pickle.dump(stroke_dataset, fp)

        opacity_profiles_name = os.path.join("data/opacity_profiles", stroke_dataset_designer_name+".json")
        if os.path.exists(opacity_profiles_name):
            with open(opacity_profiles_name, "r") as fp:
                opacity_profiles = json.load(fp)
        else:
            opacity_profiles = get_opacity_profiles(stroke_dataset_designer_name)
            with open(opacity_profiles_name, "w") as fp:
                json.dump(opacity_profiles, fp)

        #import polyscope as ps
        #import polyscope_bindings as ps_b
        #ps.init()
        #for edge_id, edge in enumerate(final_edges_dict.values()):
        #    l = np.array(edge["geometry_3d"])
        #    ps.register_curve_network(str(edge_id), l,
        #                              np.array([[i, i+1] for i in range(len(l)-1)]))
        #ps.show()
        #exit()

        # DEBUG
        # normal map rendering
        patches = utils.load_last_faces(data_folder)
        normal_pixels = get_normal_map_single_mesh(patches, display, cam_pos, obj_center, up_vec)
        plt.gcf().subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                                  bottom=0.0,
                                  top=1.0)
        plt.gca().imshow(normal_pixels)
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.gcf().set_size_inches(display[0]/100, display[1]/100)
        normal_map_name = os.path.join(training_data_folder, "normal_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        plt.savefig(normal_map_name, dpi=100)
        plt.close(plt.gcf())

        # get opacities
        new_opacities = optimize_opacities(final_edges_dict, stylesheet, cam_pos, obj_center, up_vec, mesh, VERBOSE=False)
        #print("new_opacities")
        #for i, n in enumerate(new_opacities):
        #    print(i, n)

        width = 972
        syn_sketch = geometry_match(final_edges_dict, stylesheet,
                                    cam_pos, obj_center, up_vec, display,
                                    clean_rendering=clean_rendering)
        scale_factor = np.minimum(width, display[0])/width
        if not clean_rendering:
            for s_id in range(len(syn_sketch.strokes)):
                for p_id in range(len(syn_sketch.strokes[s_id].points_list)):
                    syn_sketch.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])

            # DEBUG
            syn_sketch = perturbate_sketch(syn_sketch)

            npr_sketch = match_strokes(syn_sketch, stroke_dataset, opacity_profiles,
                                       opacity_threshold=0.1,
                                       straight_line_nearest_neighbor_range=[0, 10],
                                       target_smoothness=0.3,
                                       scale_factor=scale_factor,
                                       optimize_stroke_length=match_stroke_length)

        pen_width = 1.5*scale_factor
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                            bottom=0.0,
                            top=1.0)
        # new rendering pipeline
        if clean_rendering:
            npr_sketch = syn_sketch
        for s_i, s in enumerate(npr_sketch.strokes):
            for p_id in range(len(s.points_list)-1):
                pressure = s.points_list[p_id+1].get_data("pressure")
                #print(pen_width*pressure)
                ax.plot([s.points_list[p_id].coords[0], s.points_list[p_id+1].coords[0]],
                        [s.points_list[p_id].coords[1], s.points_list[p_id+1].coords[1]],
                        c=(0, 0, 0, pressure),
                        lw=pen_width*pressure)
        ax.set_xlim(0, display[0])
        ax.set_ylim(display[1], 0)
        ax.set_aspect("equal")
        plt.gca().invert_yaxis()
        ax.axis("off")
        fig.set_size_inches(display[0]/100, display[1]/100)
        npr_sketch_name = os.path.join(training_data_folder, "npr_baseline1_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        if clean_rendering:
            npr_sketch_name = os.path.join(training_data_folder, "npr_baseline1_clean_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        if baseline_0:
            npr_sketch_name = os.path.join(training_data_folder, "npr_baseline0_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        print(npr_sketch_name)
        plt.savefig(npr_sketch_name, dpi=100)
        plt.close(fig)
        try:
            obj_id = int(data_folder.split("/")[-1])
            out_name = os.path.join(training_data_folder, "out_baseline1_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_npr.png")
            if clean_rendering:
                out_name = os.path.join(training_data_folder, "out_baseline1_clean_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_npr.png")
            if baseline_0:
                out_name = os.path.join(training_data_folder, "out_baseline0_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_npr.png")
            print(out_name)
            os.system("montage "+npr_sketch_name+" "+normal_map_name+" -tile 2x1 -geometry +0+0 "+out_name)
        except:
            print("data_folder ", data_folder, "is not an integer")
        
    #if clean_rendering:
    #    print("CLEAN_RENDERING")
    #    if not declutter_construction_lines:
    #        final_edges_file_name = os.path.join(per_view_data_folder, "final_edges.json")
    #        with open(final_edges_file_name, "r") as fp:
    #            final_edges_dict = json.load(fp)
    #    syn_sketch = geometry_match(final_edges_dict, [],
    #                                cam_pos, obj_center, up_vec, display,
    #                                clean_rendering=clean_rendering)

    #    fig, ax = plt.subplots(nrows=1, ncols=1)
    #    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
    #                        bottom=0.0,
    #                        top=1.0)
    #    syn_sketch.display_strokes_2(fig, ax,
    #                                 linewidth_data=lambda p: 1.0,
    #                                 color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
    #                                                          for p in s.points_list])
    #    #for s in syn_sketch.strokes:
    #    #    l = np.array(s.linestring.linestring)
    #    #    plt.plot(np.array(l)[:, 0], np.array(l)[:, 1], c="black")
    #    ax.set_xlim(0, display[0])
    #    ax.set_ylim(display[1], 0)
    #    ax.set_aspect("equal")
    #    plt.gca().invert_yaxis()
    #    ax.axis("off")
    #    fig.set_size_inches(display[0]/100, display[1]/100)
    #    npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
    #    plt.savefig(npr_sketch_name, dpi=100)
    #    plt.close(fig)
    #    print("Saved", npr_sketch_name)
    #    ours_per_view_data_folder = os.path.join(data_folder, "ours_"+str(theta)+"_"+str(phi)+"_"+str(radius))
    #    ours_training_data_folder = os.path.join(ours_per_view_data_folder, "training_data")
    #    construction_name = os.path.join(ours_training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
    #    #print(construction_name)
    #    obj_id = int(data_folder.split("/")[-1])
    #    out_name = os.path.join(training_data_folder, "out_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
    #    print("before montage", out_name)
    #    if os.path.exists(construction_name):
    #        os.system("montage "+construction_name+" "+npr_sketch_name+" -tile 2x1 -geometry +0+0 "+out_name)
    #    else:
    #        print(construction_name, "doesn't exist")

    if npr_rendering:
        print("NPR_RENDERING")
        start_time = time()

        if not declutter_construction_lines:
            final_edges_file_name = os.path.join(per_view_data_folder, "final_edges.json")
            with open(final_edges_file_name, "r") as fp:
                final_edges_dict = json.load(fp)

        #style_sheet_file_name = os.path.join("data/stylesheets/"+stylesheet_designer_name+".json")
        with open(style_sheet_file_name, "r") as fp:
            stylesheet = json.load(fp)

        stroke_dataset_name = os.path.join("data/stroke_datasets", stroke_dataset_designer_name+".pkl")
        if os.path.exists(stroke_dataset_name):
            with open(stroke_dataset_name, "rb") as fp:
                stroke_dataset = pickle.load(fp)
        else:
            stroke_dataset = get_stroke_dataset(stroke_dataset_designer_name)
            with open(stroke_dataset_name, "wb") as fp:
                pickle.dump(stroke_dataset, fp)

        opacity_profiles_name = os.path.join("data/opacity_profiles", stroke_dataset_designer_name+".json")
        if os.path.exists(opacity_profiles_name):
            with open(opacity_profiles_name, "r") as fp:
                opacity_profiles = json.load(fp)
        else:
            opacity_profiles = get_opacity_profiles(stroke_dataset_designer_name)
            with open(opacity_profiles_name, "w") as fp:
                json.dump(opacity_profiles, fp)

        #import polyscope as ps
        #import polyscope_bindings as ps_b
        #ps.init()
        #for edge_id, edge in enumerate(final_edges_dict.values()):
        #    l = np.array(edge["geometry_3d"])
        #    ps.register_curve_network(str(edge_id), l,
        #                              np.array([[i, i+1] for i in range(len(l)-1)]))
        #ps.show()
        #exit()

        # DEBUG
        # normal map rendering
        patches = utils.load_last_faces(data_folder)
        normal_pixels = get_normal_map_single_mesh(patches, display, cam_pos, obj_center, up_vec)
        plt.gcf().subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                                  bottom=0.0,
                                  top=1.0)
        plt.gca().imshow(normal_pixels)
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.gcf().set_size_inches(display[0]/100, display[1]/100)
        normal_map_name = os.path.join(training_data_folder, "normal_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        plt.savefig(normal_map_name, dpi=100)
        plt.close(plt.gcf())

        # get opacities
        new_opacities = optimize_opacities(final_edges_dict, stylesheet, cam_pos, obj_center, up_vec, mesh, VERBOSE=False)
        #print("new_opacities")
        #for i, n in enumerate(new_opacities):
        #    print(i, n)


        width = 972
        syn_sketch = geometry_match(final_edges_dict, stylesheet,
                                    cam_pos, obj_center, up_vec, display,
                                    clean_rendering=clean_rendering)
        for s_id in range(len(syn_sketch.strokes)):
            for p_id in range(len(syn_sketch.strokes[s_id].points_list)):
                syn_sketch.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])

        #fig, ax = plt.subplots(nrows=1, ncols=1)
        #fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
        #                    bottom=0.0,
        #                    top=1.0)
        #syn_sketch.display_strokes_2(fig, ax,
        #                             linewidth_data=lambda p: p.get_data("pressure")+0.5,
        #                             color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
        #                                                      for p in s.points_list])
        #ax.set_xlim(0, display[0])
        #ax.set_ylim(display[1], 0)
        #ax.set_aspect("equal")
        #ax.axis("off")
        #fig.set_size_inches(display[0]/100, display[1]/100)
        #plt.savefig(os.path.join(training_data_folder, "test_strokes.png"), dpi=100)
        #plt.close(fig)

        #subdivide_long_curves(syn_sketch, VERBOSE=True)
        # DEBUG
        syn_sketch = perturbate_sketch(syn_sketch)

        if verbose:
            syn_sketch_file_name = os.path.join(per_view_data_folder, "synthetic_sketch.svg")
            skio.save(syn_sketch, syn_sketch_file_name)
            os.system("rsvg-convert -f pdf " + syn_sketch_file_name + " > " + syn_sketch_file_name.replace("svg", "pdf"))

        scale_factor = np.minimum(width, display[0])/width
        #syn_sketch = utils.rescale_rotate_center(syn_sketch)
        #npr_sketch_file_name = os.path.join(per_view_data_folder, "synthetic_sketch_v2.svg")
        #skio.save(syn_sketch, npr_sketch_file_name)
        npr_sketch = match_strokes(syn_sketch, stroke_dataset, opacity_profiles,
                                   opacity_threshold=0.1,
                                   straight_line_nearest_neighbor_range=[0, 10],
                                   target_smoothness=0.3,
                                   scale_factor=scale_factor,
                                   optimize_stroke_length=match_stroke_length)
        time_step = 0.05
        time_counter = 0.0
        for s_id, s in enumerate(npr_sketch.strokes):
            s.add_avail_data("time")
            for p in s.points_list:
                p.add_data("time", time_counter)
                time_counter += time_step
            time_counter += time_step
        #npr_sketch = utils.rescale_rotate_center(npr_sketch)
        npr_sketch_file_name = os.path.join(per_view_data_folder, "npr_sketch.json")
        skio.save(npr_sketch, npr_sketch_file_name)
        npr_sketch_file_name = os.path.join(per_view_data_folder, "npr_sketch.png")
        skio.save(npr_sketch, npr_sketch_file_name)
        #exit()
        npr_sketch_file_name = os.path.join(per_view_data_folder, "npr_sketch.svg")
        skio.save(npr_sketch, npr_sketch_file_name)
        os.system("rsvg-convert -f pdf " + npr_sketch_file_name + " > " + npr_sketch_file_name.replace("svg", "pdf"))

        pen_width = 1.5*scale_factor
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                            bottom=0.0,
                            top=1.0)
        ax.set_xlim(0, display[0])
        ax.set_ylim(display[1], 0)
        plt.gca().invert_yaxis()
        ax.axis("off")
        fig.set_size_inches(display[0]/100, display[1]/100)
        # old rendering pipeline
        #npr_sketch.display_strokes_2(fig, ax,
        #                             linewidth_data=lambda p: p.get_data("pressure")+0.5,
        #                             color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
        #                                                      for p in s.points_list])
        # new rendering pipeline
        for s_i, s in enumerate(npr_sketch.strokes):
            for p_id in range(len(s.points_list)-1):
                pressure = s.points_list[p_id+1].get_data("pressure")
                #print(pen_width*pressure)
                ax.plot([s.points_list[p_id].coords[0], s.points_list[p_id+1].coords[0]],
                        [s.points_list[p_id].coords[1], s.points_list[p_id+1].coords[1]],
                        c=(0, 0, 0, min(1.0, pressure)),
                        lw=pen_width*pressure)
        obj_id = data_folder.split("/")[-1]
        npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        npr_sketch_name = os.path.join("only_visible_sketches", "npr_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        #npr_sketch_name = os.path.join(training_data_folder, str(s_i)+".png")
        plt.savefig(npr_sketch_name, dpi=100)
        print("saved sketch in ", npr_sketch_name)
        #plt.close(fig)
        #exit()
        #for s in syn_sketch.strokes:
        #    l = np.array(s.linestring.linestring)
        #    plt.plot(np.array(l)[:, 0], np.array(l)[:, 1], c="black")
        #ax.set_aspect("equal")
        #npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_professional3.png")
        #npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        npr_sketch_name = os.path.join(training_data_folder, "ours_lambda1_"+str(lambda_1)+"_lambda2_"+str(lambda_2)+"_style_"+stroke_dataset_designer_name+".png")
        plt.savefig(npr_sketch_name, dpi=100)
        plt.close(fig)
        try:
            #obj_id = int(data_folder.split("/")[-1])
            obj_id = data_folder.split("/")[-1]
            out_name = os.path.join(training_data_folder, "out_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_professional3_npr.png")
            out_name = os.path.join(training_data_folder, "out_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_npr.png")
            os.system("montage "+npr_sketch_name+" "+normal_map_name+" -tile 2x1 -geometry +0+0 "+out_name)
        except:
            print("data_folder ", data_folder, "is not an integer")
        print("stylization_time", time() - start_time)

    if cad_sequence_rendering:
        print("CAD SEQUENCE RENDERING")

        if not declutter_construction_lines:
            final_edges_file_name = os.path.join(per_view_data_folder, "final_edges.json")
            with open(final_edges_file_name, "r") as fp:
                final_edges_dict = json.load(fp)
            all_edges_file_name = os.path.join(per_view_data_folder, "unique_edges.json")
            with open(all_edges_file_name, "r") as fp:
                all_edges = json.load(fp)
        #style_sheet_file_name = os.path.join("data/stylesheets/"+stylesheet_designer_name+".json")
        with open(style_sheet_file_name, "r") as fp:
            stylesheet = json.load(fp)

        stroke_dataset_name = os.path.join("data/stroke_datasets", stroke_dataset_designer_name+".pkl")
        if os.path.exists(stroke_dataset_name):
            with open(stroke_dataset_name, "rb") as fp:
                stroke_dataset = pickle.load(fp)
        else:
            stroke_dataset = get_stroke_dataset(stroke_dataset_designer_name)
            with open(stroke_dataset_name, "wb") as fp:
                pickle.dump(stroke_dataset, fp)

        opacity_profiles_name = os.path.join("data/opacity_profiles", stroke_dataset_designer_name+".json")
        if os.path.exists(opacity_profiles_name):
            with open(opacity_profiles_name, "r") as fp:
                opacity_profiles = json.load(fp)
        else:
            opacity_profiles = get_opacity_profiles(stroke_dataset_designer_name)
            with open(opacity_profiles_name, "w") as fp:
                json.dump(opacity_profiles, fp)

        #import polyscope as ps
        #import polyscope_bindings as ps_b
        #ps.init()
        #for edge_id, edge in enumerate(final_edges_dict.values()):
        #    l = np.array(edge["geometry_3d"])
        #    ps.register_curve_network(str(edge_id), l,
        #                              np.array([[i, i+1] for i in range(len(l)-1)]))
        #ps.show()
        #exit()

        syn_sketch = geometry_match(final_edges_dict, stylesheet,
                                    cam_pos, obj_center, up_vec, display,
                                    clean_rendering=clean_rendering)

        new_opacities = optimize_opacities(final_edges_dict, stylesheet, cam_pos, obj_center, up_vec, mesh)
        for s_id in range(len(syn_sketch.strokes)):
            for p_id in range(len(syn_sketch.strokes[s_id].points_list)):
                syn_sketch.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])

        #syn_sketch = perturbate_sketch(syn_sketch)
        npr_sketch = match_strokes(syn_sketch, stroke_dataset, opacity_profiles,
                                   opacity_threshold=0.1,
                                   straight_line_nearest_neighbor_range=[0, 10])

        patches = utils.load_faces_i(data_folder, 0)
        normal_pixels = get_normal_map_single_mesh(patches, display, cam_pos, obj_center, up_vec)
        plt.gcf().subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                                  bottom=0.0,
                                  top=1.0)
        plt.gca().imshow(normal_pixels)
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.gcf().set_size_inches(display[0]/100, display[1]/100)
        normal_map_name = os.path.join(training_data_folder, "normal_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_cad_"+str(0)+".png")
        plt.savefig(normal_map_name, dpi=100)
        plt.close(plt.gcf())
        with open(os.path.join(data_folder, "parsed_features.json"), "r") as f:
            parsed_features = json.load(f)
        cad_i_real = 0
        max_stroke_id_from_last_step = 0
        for cad_i in range(utils.cad_seq_last_id(data_folder)+1):
            if parsed_features["sequence"][cad_i]["type"] == "Sketch":
                continue

            patches = utils.load_faces_i(data_folder, cad_i)
            normal_pixels = get_normal_map_single_mesh(patches, display, cam_pos, obj_center, up_vec)
            plt.gcf().subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                                      bottom=0.0,
                                      top=1.0)
            plt.gca().imshow(normal_pixels)
            plt.gca().invert_yaxis()
            plt.axis("off")
            plt.gcf().set_size_inches(display[0]/100, display[1]/100)
            normal_map_name = os.path.join(training_data_folder, "normal_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_cad_"+str(cad_i)+".png")
            plt.savefig(normal_map_name, dpi=100)
            plt.close(plt.gcf())

            cad_i_line_ids = [edge_id for edge_id, edge in enumerate(all_edges)
                              if edge["feature_id"] == cad_i]
            final_cad_i_line_ids = [edge_id for edge_id, edge in enumerate(final_edges_dict.values())
                                    if edge["id"] in cad_i_line_ids]
            #print("cad_i", cad_i)
            #print("cad_i_line_ids")
            #print(cad_i_line_ids)
            #print("final_cad_i_line_ids")
            #print(final_cad_i_line_ids)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                                bottom=0.0,
                                top=1.0)
            display_strokes = np.arange(max_stroke_id_from_last_step, np.max(final_cad_i_line_ids)+1)
            #print(display_strokes)
            npr_sketch.display_strokes_2(fig, ax,
                                         linewidth_data=lambda p: p.get_data("pressure")+0.5,
                                         color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
                                                                  for p in s.points_list],
                                         display_strokes=display_strokes)
            max_stroke_id_from_last_step = np.max(final_cad_i_line_ids)+1
            #for s in syn_sketch.strokes:
            #    l = np.array(s.linestring.linestring)
            #    plt.plot(np.array(l)[:, 0], np.array(l)[:, 1], c="black")
            ax.set_xlim(0, display[0])
            ax.set_ylim(display[1], 0)
            ax.set_aspect("equal")
            plt.gca().invert_yaxis()
            ax.axis("off")
            fig.set_size_inches(display[0]/100, display[1]/100)
            npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_cad_"+str(cad_i)+".png")
            plt.savefig(npr_sketch_name, dpi=100)
            plt.close(fig)
            obj_id = int(data_folder.split("/")[-1])
            out_name = os.path.join(training_data_folder, "out_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_cad_"+str(cad_i)+".png")
            cad_i_prev = cad_i_real
            prev_normal_map_name = os.path.join(training_data_folder, "normal_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_cad_"+str(cad_i_prev)+".png")
            os.system("montage "+npr_sketch_name+" " + prev_normal_map_name + " " + normal_map_name + " -tile 3x1 -geometry +0+0 "+out_name)
            cad_i_real = cad_i
