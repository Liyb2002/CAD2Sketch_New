import os
from time import process_time
from copy import deepcopy
from render_shapes import features_lines_to_svg, typed_feature_lines_to_svg, \
    typed_feature_lines_to_svg_successive, indexed_lines_to_svg
import json
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

import numpy as np

import utils

M = 100

def construct_same_plan_graph(graph, strokes, s_id, plane_id):
    for prev_s_id in strokes[s_id].previous_strokes:
        if plane_id in strokes[prev_s_id].planes:
            graph.add_edge(prev_s_id, s_id)
            construct_same_plan_graph(graph, strokes, prev_s_id, plane_id)

def construct_per_plane_graphs(strokes):
    per_plane_graphs = {}
    for s in strokes:
        for p_id in s.planes:
            for prev_s_id in s.previous_strokes:
                if p_id in strokes[prev_s_id].planes:
                    if not (p_id in per_plane_graphs.keys()):
                        per_plane_graphs[p_id] = nx.Graph()
                    per_plane_graphs[p_id].add_edge(prev_s_id, s.id)
    return per_plane_graphs

def extract_sub_plane_graph(per_plane_graphs, plane_id, max_s_id):
    if not plane_id in per_plane_graphs.keys():
        return nx.Graph()
    return nx.Graph(per_plane_graphs[plane_id].subgraph([n for n in per_plane_graphs[plane_id].nodes() if n <= max_s_id]))


def declutter(strokes, lambda_0=10.0, lambda_1=5.0, lambda_2=1.0, lambda_3=10.0, stroke_lengths=[],
              ellipse_fittings=[], visibility_scores=[], constraints=[],
              intersection_dag=None, per_stroke_descendants=None, timeout=False):
    #print("declutter model creation")

    m = gp.Model("declutter")
    m.setParam("OutputFlag", 0)
    if timeout:
        m.setParam("TimeLimit", 20)

    coplanar_graphs = []

    # variable definitions
    stroke_names = [s.id for s in strokes]
    #print(stroke_names)
    #intersection_names = [(s.id, i) for s in strokes for i in range(len(s.intersections)) ]
    intersection_names = [(s.id, i) for s in strokes for i in range(len(s.anchor_intersections)) ]
    tan_intersection_names = [(s.id, i) for s in strokes for i in range(len(s.tangent_intersections)) ]
    occlusion_names = [(s.id, i) for s in strokes for i in range(len(s.occlusions))]

    x_vars = m.addVars(stroke_names, vtype=GRB.BINARY)
    pos_vars = m.addVars(stroke_names, vtype=GRB.BINARY)
    a_vars = m.addVars(stroke_names, vtype=GRB.BINARY)
    tan_vars = m.addVars(stroke_names, vtype=GRB.BINARY)
    half_a_vars = m.addVars(stroke_names, vtype=GRB.BINARY) # half-anchored strokes, typically only one intersection
    i_vars = m.addVars(intersection_names, vtype=GRB.BINARY)
    o_vars = m.addVars(occlusion_names, vtype=GRB.BINARY)
    tan_i_vars = m.addVars(tan_intersection_names, vtype=GRB.BINARY)
    pos_i_vars = m.addVars(intersection_names, vtype=GRB.BINARY)
    final_c_vars = m.addVars(stroke_names, vtype=GRB.BINARY)
    path_vars_dict = {}
    per_plane_graphs = construct_per_plane_graphs(strokes)

    per_stroke_descendants_ids = [(int(s_id), i) for s_id in per_stroke_descendants.keys() for i in per_stroke_descendants[s_id]
                                  if len(strokes[int(s_id)].anchor_intersections) > 1 and len(ellipse_fittings[int(s_id)]) == 0]
    per_stroke_descendants_vars = m.addVars(per_stroke_descendants_ids, vtype=GRB.BINARY)
    #print(per_stroke_descendants_ids)
    #exit()

    projection_constraint_indices = set()
    per_projection_constraint_stroke_ids = {}
    for s_id, s in enumerate(strokes):
        for pc_id in s.projection_constraint_ids:
            projection_constraint_indices.add(pc_id)
            if pc_id in per_projection_constraint_stroke_ids.keys():
                if not s_id in per_projection_constraint_stroke_ids[pc_id]:
                    per_projection_constraint_stroke_ids[pc_id].append(s_id)
            else:
                per_projection_constraint_stroke_ids[pc_id] = [s_id]
    projection_constraint_indices = list(projection_constraint_indices)
    #print(projection_constraint_indices)
    projection_constraint_vars = m.addVars(projection_constraint_indices, vtype=GRB.BINARY)
    #print(projection_constraint_vars)
    #print(per_projection_constraint_stroke_ids)

    # midpoint constraint variables
    const_vars = {}
    for const_id in constraints.keys():
        const = constraints[const_id]
        if const["type"] != "midpoint":
            continue
        midpoint_vars = m.addVars([("main"),
                                   ("midpoint"),
                                   ("midpoint_perp"),
                                   ("p0_perp"),
                                   ("p0_diag"),
                                   ("p1_perp"),
                                   ("p1_diag"),
                                   ("last")],
                                  vtype=GRB.BINARY)
        midpoint_x_ids = const["midpoint_line_ids"] + const["midpoint_perp_line_ids"] + \
                         const["p0_perp_line_ids"] + const["p0_diag_line_ids"] + \
                         const["p1_perp_line_ids"] + const["p1_diag_line_ids"] + \
                         const["last_line_ids"]
        midpoint_x_vars = m.addVars(midpoint_x_ids, vtype=GRB.BINARY)

        midpoint_intersection_ids = []
        midpoint_triple_intersection_ids = []
        for s_id in midpoint_x_ids:
            # add intersections
            #for i in range(len(strokes[s_id].anchor_intersections)):
            #    # how many are in midpoint_x_ids?
            #    in_x_ids = [tmp_s_id for tmp_s_id in strokes[s_id].anchor_intersections[i]
            #                if tmp_s_id in midpoint_x_ids]
            for i in range(len(strokes[s_id].intersections)):
                # how many are in midpoint_x_ids?
                in_x_ids = [tmp_s_id for tmp_s_id in strokes[s_id].intersections[i]
                            if tmp_s_id in midpoint_x_ids]
                if len(in_x_ids) > 0:
                    midpoint_intersection_ids.append((strokes[s_id].id, i))
                if len(in_x_ids) > 1:
                    #print(s_id, in_x_ids)
                    midpoint_triple_intersection_ids.append((strokes[s_id].id, i))
        midpoint_intersection_vars = m.addVars(midpoint_intersection_ids, vtype=GRB.BINARY)
        midpoint_triple_intersection_vars = m.addVars(midpoint_triple_intersection_ids, vtype=GRB.BINARY)

        const_vars[const_id] = {
            "midpoint_vars": midpoint_vars,
            "midpoint_x_ids": midpoint_x_ids,
            "midpoint_x_vars": midpoint_x_vars,
            "midpoint_intersection_vars": midpoint_intersection_vars,
            "midpoint_triple_intersection_vars": midpoint_triple_intersection_vars}

    m.update()

    for const_id in constraints.keys():
        const = constraints[const_id]
        if const["type"] != "midpoint":
            continue
        # select in total 7 strokes
        midpoint_vars = const_vars[const_id]["midpoint_vars"]
        midpoint_x_ids = const_vars[const_id]["midpoint_x_ids"]
        midpoint_x_vars = const_vars[const_id]["midpoint_x_vars"]
        midpoint_intersection_vars = const_vars[const_id]["midpoint_intersection_vars"]
        midpoint_triple_intersection_vars = const_vars[const_id]["midpoint_triple_intersection_vars"]
        m.addConstr(midpoint_x_vars.sum("*") >= 7-M*(1-midpoint_vars.sum("main")))
        #m.addConstr(midpoint_vars.sum("main") == 1)
        # select at least 7 intersections
        m.addConstr(midpoint_intersection_vars.sum("*") >= 7-M*(1-midpoint_vars.sum("main")))
        # select at least 5 triple intersections
        m.addConstr(midpoint_triple_intersection_vars.sum("*") >= 5-M*(1-midpoint_vars.sum("main")))

        # select at most one stroke of each category
        m.addConstr(midpoint_x_vars.sum(const["midpoint_line_ids"]) <= 1)
        m.addConstr(midpoint_x_vars.sum(const["midpoint_perp_line_ids"]) <= 1)
        m.addConstr(midpoint_x_vars.sum(const["p0_perp_line_ids"]) <= 1)
        m.addConstr(midpoint_x_vars.sum(const["p0_diag_line_ids"]) <= 1)
        m.addConstr(midpoint_x_vars.sum(const["p1_perp_line_ids"]) <= 1)
        m.addConstr(midpoint_x_vars.sum(const["p1_diag_line_ids"]) <= 1)
        m.addConstr(midpoint_x_vars.sum(const["last_line_ids"]) <= 1)
        for s_id in midpoint_x_ids:
            # a stroke must be selected by x_vars
            m.addConstr(midpoint_x_vars.sum(s_id) <= x_vars.sum(s_id))
            # a stroke must have at least one intersection
            #m.addConstr(midpoint_intersection_vars.sum(s_id, "*") >= 1 + M(1-midpoint_x_vars.sum(s_id)))

        # an intersection can only happen if both strokes are intersected
        for s_id in midpoint_x_ids:
            # add intersections
            for i in range(len(strokes[s_id].intersections)):
                in_x_ids = [tmp_s_id for tmp_s_id in strokes[s_id].intersections[i]
                            if tmp_s_id in midpoint_x_ids]
                if len(in_x_ids) > 0:
                    m.addConstr(midpoint_x_vars.sum(s_id) + midpoint_x_vars.sum(strokes[s_id].intersections[i]) >= 2-M*(1-midpoint_intersection_vars.sum(s_id, i)))
                    m.addConstr(midpoint_intersection_vars.sum(s_id, i) <= midpoint_x_vars.sum(s_id))
                if len(in_x_ids) > 1:
                    m.addConstr(midpoint_x_vars.sum(s_id) + midpoint_x_vars.sum(strokes[s_id].intersections[i]) >= 3-M*(1-midpoint_triple_intersection_vars.sum(s_id, i)))
                    m.addConstr(midpoint_triple_intersection_vars.sum(s_id, i) <= midpoint_x_vars.sum(s_id))
        #for s_id in midpoint_x_ids:
        #    # add intersections
        #    for i in range(len(strokes[s_id].anchor_intersections)):
        #        in_x_ids = [tmp_s_id for tmp_s_id in strokes[s_id].anchor_intersections[i]
        #                    if tmp_s_id in midpoint_x_ids]
        #        if len(in_x_ids) > 0:
        #            m.addConstr(midpoint_x_vars.sum(s_id) + midpoint_x_vars.sum(strokes[s_id].anchor_intersections[i]) >= 2-M*(1-midpoint_intersection_vars.sum(s_id, i)))
        #            m.addConstr(midpoint_intersection_vars.sum(s_id, i) <= midpoint_x_vars.sum(s_id))
        #        if len(in_x_ids) > 1:
        #            print(s_id, strokes[s_id].anchor_intersections[i])
        #            m.addConstr(midpoint_x_vars.sum(s_id) + midpoint_x_vars.sum(strokes[s_id].anchor_intersections[i]) >= 3-M*(1-midpoint_triple_intersection_vars.sum(s_id, i)))
        #            m.addConstr(midpoint_triple_intersection_vars.sum(s_id, i) <= midpoint_x_vars.sum(s_id))

    # turn on all feature_lines
    #visibility_sum = gp.LinExpr()
    #for s in strokes:
    #    #if s.type == "feature_line" or s.type == "sketch" or s.type == "silhouette_line":
    #    #    #print("keep line", s.id, s.type)
    #    #    x_vars[s.id].ub = 1
    #    #    x_vars[s.id].lb = 1
    #    visibility_sum += x_vars[s.id]*visibility_scores[s.id]

    # constraint definitions
    per_stroke_mutually_exlusive_set_vars = {}
    for s in strokes:
        per_stroke_mutually_exlusive_set_vars[s.id] = 0
        #print("stroke_id", s.id)
        # well-anchored constraint
        m.addConstr(a_vars[s.id] <= x_vars[s.id])
        m.addConstr(pos_vars[s.id] <= x_vars[s.id])
        m.addConstr(half_a_vars[s.id] <= x_vars[s.id])
        # deactivate half_a if a is 1
        m.addConstr(a_vars[s.id] <= 0 + M*(1-half_a_vars[s.id]))
        if len(s.anchor_intersections) > 1 and len(ellipse_fittings[s.id]) == 0:
            m.addConstr(i_vars.sum(s.id, "*") >= len(s.anchor_intersections) - M*(1-a_vars[s.id]))
            m.addConstr(pos_i_vars.sum(s.id, "*") + x_vars.sum(s.overlapping_stroke_ids) >= 1 - M*(1-pos_vars[s.id]))
            m.addConstr(i_vars.sum(s.id, "*") >= 1 - M * (1 - half_a_vars[s.id]))
        elif len(s.anchor_intersections) > 0 and len(ellipse_fittings[s.id]) == 0:
            m.addConstr(i_vars.sum(s.id, "*") >= 1 - M * (1 - half_a_vars[s.id]))
            m.addConstr(pos_i_vars.sum(s.id, "*") >= 1 - M*(1-pos_vars[s.id]))
            a_vars[s.id].ub = 0
            a_vars[s.id].lb = 0
            #a_vars[s.id].ub = 1
            #a_vars[s.id].lb = 1
        elif len(s.anchor_intersections) == 0:# or len(ellipse_fittings[s.id]) > 0:
            #half_a_vars[s.id].ub = 1
            #half_a_vars[s.id].lb = 1
            #a_vars[s.id].ub = 1
            #a_vars[s.id].lb = 1
            #pos_vars[s.id].ub = 1
            #pos_vars[s.id].lb = 1
            half_a_vars[s.id].ub = 0
            half_a_vars[s.id].lb = 0
            a_vars[s.id].ub = 0
            a_vars[s.id].lb = 0
            pos_vars[s.id].ub = 0
            pos_vars[s.id].lb = 0
        elif len(ellipse_fittings[s.id]) > 0:
            m.addConstr(a_vars.sum(s.id) <= tan_vars.sum(s.id))
        if len(s.tangent_intersections) > 0:
            m.addConstr(tan_i_vars.sum(s.id, "*") >= len(s.tangent_intersections) - M*(1-tan_vars[s.id]))
        else:
            m.addConstr(tan_vars[s.id] == 0)
        #i_vars[ a_vars[s.id]
        #for i in range(len(s.intersections)):
        for i in range(len(s.tangent_intersections)):
            #print(s.id)
            #print(gp.quicksum(x_vars.select(list(set(s.tangent_intersections[i]) - set(s.overlapping_stroke_ids)))) >= 1 - M*(1-tan_i_vars[(s.id, i)]))
            m.addConstr(gp.quicksum(x_vars.select(list(set(s.tangent_intersections[i]) - set(s.overlapping_stroke_ids)))) >= 1 - M*(1-tan_i_vars[(s.id, i)]))
        for i in range(len(s.anchor_intersections)):
            # intersection constraint
            #print(s.anchor_intersections[i], gp.quicksum(x_vars.select(s.intersections[i])))
            #print("stroke_id", s.id)
            #print(s.overlapping_stroke_ids)
            #print(x_vars.select(s.anchor_intersections[i]))
            #print(x_vars.select(list(set(s.anchor_intersections[i]) - set(s.overlapping_stroke_ids))))
            prev_i_ids = list(set(s.anchor_intersections[i]) - set(s.overlapping_stroke_ids))
            #prev_a_ids = [prev_i_id for prev_i_id in prev_i_ids if strokes[prev_i_id].feature_id > 1]
            #prev_not_a_ids = [prev_i_id for prev_i_id in prev_i_ids if strokes[prev_i_id].feature_id <= 1]
            #print(prev_i_ids)
            #print(prev_a_ids)
            #print(prev_not_a_ids)
            #print(a_vars.select(prev_a_ids))
            #print(x_vars.select(prev_not_a_ids))
            #print(gp.quicksum(a_vars.select(prev_a_ids) + x_vars.select(prev_not_a_ids)))
            # An intersection can only happen with previously SELECTED strokes
            m.addConstr(gp.quicksum(x_vars.select(list(set(s.anchor_intersections[i]) - set(s.overlapping_stroke_ids)))) >= 1 - M*(1-i_vars[(s.id, i)]))
            # An intersection can only happen with previously WELL CONSTRUCTED strokes
            #m.addConstr(gp.quicksum(a_vars.select(prev_a_ids) + x_vars.select(prev_not_a_ids)) >= 1 - M*(1-i_vars[(s.id, i)]))
            # for each positional intersection constraint, we have to get rid of overlapping strokes
            mutually_exclusive_set_vars = []
            overlapping_ids_adj_mat = np.zeros([len(s.anchor_intersections[i]), len(s.anchor_intersections[i])], dtype=int)
            if len(s.anchor_intersections[i]) < 2:
                m.addConstr(pos_i_vars[(s.id, i)] == 0)
            else:
                #print("s.anchor_intersections[i]")
                #print(s.anchor_intersections[i])
                for vec_id, s_i in enumerate(s.anchor_intersections[i]):
                    for overlapping_s_i in strokes[s_i].overlapping_stroke_ids:
                        tmp_s_i = np.argwhere(np.array(s.anchor_intersections[i]) == overlapping_s_i).flatten()
                        if len(tmp_s_i) > 0:
                            overlapping_ids_adj_mat[vec_id, tmp_s_i[0]] = 1
                            overlapping_ids_adj_mat[tmp_s_i[0], vec_id] = 1
                    #if len(strokes[s_i].overlapping_stroke_ids) > 0:
                    #    print(strokes[s_i].overlapping_stroke_ids)
                    #    overlapping_ids_adj_mat[s_i, np.array(s.anchor_intersections[i]) == strokes[s_i].overlapping_stroke_ids] = 1
                    #    overlapping_ids_adj_mat[np.array(s.anchor_intersections[i]) == strokes[s_i].overlapping_stroke_ids, s_i] = 1
                overlapping_ids_clusters = [np.array(s.anchor_intersections[i])[list(c)] for c in nx.connected_components(nx.from_numpy_matrix(overlapping_ids_adj_mat))]
                per_s_id_cluster_id = {}
                for s_i in s.anchor_intersections[i]:
                    per_s_id_cluster_id[s_i] = []
                for c_id, c in enumerate(overlapping_ids_clusters):
                    for s_i in c:
                        per_s_id_cluster_id[s_i] = c_id
                for s_i in s.anchor_intersections[i]:
                    new_set = list(set(s.anchor_intersections[i]) - set(overlapping_ids_clusters[per_s_id_cluster_id[s_i]])) + [s_i]
                    if len(new_set) < 2:
                        continue
                    mutually_exclusive_set_vars.append(m.addVar(vtype=GRB.BINARY))
                    #if s.id == 178:
                    #    print(s_i, s.anchor_intersections[i], strokes[s_i].overlapping_stroke_ids)
                    #    print(list(set(s.anchor_intersections[i]) - set(overlapping_ids_clusters[per_s_id_cluster_id[s_i]])) + [s_i])
                    #    print(gp.quicksum(x_vars.select(list(set(s.anchor_intersections[i]) - set(strokes[s_i].overlapping_stroke_ids)))) >= 2 - M*(1-mutually_exclusive_set_vars[-1]))
                    #m.addConstr(gp.quicksum(x_vars.select(list(set(s.anchor_intersections[i]) - set(strokes[s_i].overlapping_stroke_ids)))) >= 2 - M*(1-mutually_exclusive_set_vars[-1]))
                    m.addConstr(gp.quicksum(x_vars.select(new_set)) >= 2 - M*(1-mutually_exclusive_set_vars[-1]))

                    per_stroke_mutually_exlusive_set_vars[s.id] = max(per_stroke_mutually_exlusive_set_vars[s.id], len(new_set))

            m.addConstr(gp.quicksum(mutually_exclusive_set_vars) >= pos_i_vars[(s.id, i)])
            #m.addConstr(gp.quicksum(x_vars.select(s.anchor_intersections[i])) >= 2 - M*(1-pos_i_vars[(s.id, i)]))
        for i in range(len(s.occlusions)):
            m.addConstr(x_vars[s.id]+x_vars[s.occlusions[i]] <= 1 + M*o_vars[(s.id, i)])

        if s.type == "extrude_line":
            #print(s.id)
            # same_plane path existence
            # get all plane ids
            #print("EXTRUDE LINE", s.id)
            if len(s.anchor_intersections) < 2:
                #a_vars[s.id].ub = 1
                #a_vars[s.id].lb = 1
                a_vars[s.id].ub = 0
                a_vars[s.id].lb = 0
                continue
            #print(s.previous_strokes)
            same_plane_graphs = []
            plane_graph_ids = []
            all_nodes = [(-1, "start"), (-1, "end")]
            path_exists = False
            for plane_id in s.planes:
                if plane_id == -1:
                    continue
                #print("plane_id", plane_id)
                #breadth_graph = nx.Graph()
                # do not add the second intersection strokes
                #construct_same_plan_graph(breadth_graph, strokes, s.id, plane_id)
                breadth_graph = extract_sub_plane_graph(per_plane_graphs, plane_id, s.id)
                #print(plane_id, breadth_graph.nodes)
                #print(breadth_graph.edges)
                for s_id in s.anchor_intersections[1]:
                    if plane_id in strokes[s_id].planes:
                        if (s.id, s_id) in breadth_graph.edges.keys():
                            breadth_graph.remove_edge(s.id, s_id)
                for over_n in s.overlapping_stroke_ids:
                    if over_n in breadth_graph.nodes():
                        breadth_graph.remove_node(over_n)
                if len(breadth_graph.edges) == 0:
                    continue
                breadth_first_edges = list(nx.edge_bfs(breadth_graph, source=s.id))
                #print("bfs_edges", breadth_first_edges)
                #exit()
                g = nx.DiGraph()
                for edge in breadth_first_edges:
                    #print(edge)
                    g.add_edge(edge[0], edge[1])
                #print("g.edges", g.edges)
                #print(g.nodes)
                #construct_same_plan_graph(g, strokes, s.id, plane_id)
                if len(g.nodes) == 0:
                    continue
                # check if there exists a path
                for start_s_id in s.anchor_intersections[0]:
                    if not start_s_id in g.nodes:
                        continue
                    for end_s_id in s.anchor_intersections[1]:
                        if not end_s_id in g.nodes:
                            continue
                        if nx.has_path(g, start_s_id, end_s_id):
                            path_exists = True
                if not path_exists:
                    continue
                #print(g.nodes)
                g.remove_node(s.id)
                #print(g.edges)
                #print(g.nodes)
                coplanar_graphs.append([s.id, plane_id, g.nodes, g.edges])
                sorted_nodes = sorted(g.nodes)
                for n in sorted_nodes:
                    all_nodes.append((plane_id, n))
                #all_nodes += sorted_nodes
                same_plane_graphs.append(g)
                plane_graph_ids.append(plane_id)
                #if s.id == 47:
                #    exit()
            if not path_exists:
                print("no path", s.id)
                continue
            # TODO: a node can appear in multiple planes
            # TODO: introduce double naming scheme
            path_node_vars = m.addVars(all_nodes, vtype=GRB.BINARY)
            m.update()
            #print(path_node_vars)

            # path constraints
            m.addConstr(path_node_vars[(-1, "start")] <= x_vars[s.id])
            m.addConstr(path_node_vars[(-1, "end")] <= x_vars[s.id])
            m.addConstr(a_vars[s.id] <= path_node_vars[(-1, "end")])
            m.addConstr(a_vars[s.id] <= path_node_vars[(-1, "start")])
            for g_id, g in enumerate(same_plane_graphs):
                for s_id in g.nodes:
                    m.addConstr(path_node_vars[(plane_graph_ids[g_id], s_id)] <= x_vars[s_id])
            # start-node constraint
            start_sum = gp.LinExpr()
            for outgoing_s_id in s.anchor_intersections[0]:
                for graph_id, plane_id in enumerate(plane_graph_ids):
                    if plane_id in strokes[outgoing_s_id].planes:
                        # TODO: if outgoing_s_id is present in graph
                        if outgoing_s_id in same_plane_graphs[graph_id].nodes:
                            #if s.id == 60:
                            #    print("outgoing_s_id", outgoing_s_id)
                            start_sum += path_node_vars[(plane_id, outgoing_s_id)]
            m.addConstr(start_sum <= 1 + M*(1-a_vars[s.id]))
            m.addConstr(start_sum >= 1 - M*(1-a_vars[s.id]))
            # end-node constraint
            end_sum = gp.LinExpr()
            for incoming_s_id in s.anchor_intersections[1]:
                #if plane_id in strokes[incoming_s_id].planes:
                for graph_id, plane_id in enumerate(plane_graph_ids):
                #for plane_id in plane_graph_ids:
                    if plane_id in strokes[incoming_s_id].planes:
                        # TODO: if incoming_s_id is present in graph
                        #if s.id == 60:
                        #    print(incoming_s_id, same_plane_graphs[graph_id].nodes)
                        if incoming_s_id in same_plane_graphs[graph_id].nodes:
                            #if s.id == 60:
                            #    print("incoming_s_id", incoming_s_id)
                            end_sum += path_node_vars[(plane_id, incoming_s_id)]
            m.addConstr(end_sum >= 1 - M*(1-a_vars[s.id]))
            m.addConstr(end_sum <= 1 + M*(1-a_vars[s.id]))
            # all other nodes
            for g_id, g in enumerate(same_plane_graphs):
                for s_id in g.nodes:
                    #print(s_id)
                    node_sum = gp.LinExpr()
                    if s_id in s.anchor_intersections[0]:
                        #print("start_connection")
                        node_sum -= path_node_vars[(-1, "start")]
                    if s_id in s.anchor_intersections[1]:
                        #print("end_connection")
                        node_sum += path_node_vars[(-1, "end")]
                    in_nodes_vars = [path_node_vars[(plane_graph_ids[g_id], edge[0])] for edge in g.in_edges(s_id)]
                    #if s.id == 60:
                    #    print(g.in_edges(s_id))
                    #    print(g.out_edges(s_id))
                    node_sum -= gp.quicksum(in_nodes_vars)
                    out_nodes_vars = [path_node_vars[(plane_graph_ids[g_id], edge[1])] for edge in g.out_edges(s_id)]
                    #if s.id == 60:
                    #    print(s_id, g.in_edges(s_id))
                    #    print(s_id, g.out_edges(s_id))
                    node_sum += gp.quicksum(out_nodes_vars)
                    m.addConstr(node_sum >= 0 - M * (1 - path_node_vars[(plane_graph_ids[g_id], s_id)]))
                    m.addConstr(node_sum <= 0 + M * (1 - path_node_vars[(plane_graph_ids[g_id], s_id)]))
                    # all path_node_vars have to be well constructed
                    if not(len(strokes[s_id].anchor_intersections) > 1 and len(ellipse_fittings[s_id]) == 0):
                        continue
                    m.addConstr(path_node_vars[(plane_graph_ids[g_id], s_id)] <= a_vars[s_id])
            #if s.id == 17:
            #    exit()
            path_vars_dict[s.id] = path_node_vars

    # design constraint definitions
    #print("per_projection_constraint_stroke_ids")
    #for pc in per_projection_constraint_stroke_ids.values():
    #    print(pc)
    #print(per_projection_constraint_stroke_ids)
    #exit()
    #print("per_projection_constraint_stroke_ids[pc_id]")
    for pc_id in per_projection_constraint_stroke_ids.keys():
    #    print(per_projection_constraint_stroke_ids[pc_id])
        m.addConstr(x_vars.sum(per_projection_constraint_stroke_ids[pc_id]) >= 1 - M * (1-projection_constraint_vars.sum(pc_id)))

    # descendants variables
    descendants_sum = gp.QuadExpr()
    for s in strokes:
        if not(len(s.anchor_intersections) > 1 and len(ellipse_fittings[int(s.id)]) == 0):
            continue
        if not s.id in intersection_dag.nodes:
            continue
        for inter_i in list(intersection_dag.adj[s.id].keys()):
            for d_id in per_stroke_descendants[str(s.id)]:
                # direct descendant
                if d_id == inter_i:
                    m.addConstr(x_vars.sum(s.id) + x_vars.sum(inter_i) <= 1 + per_stroke_descendants_vars.sum(s.id, d_id))
                elif d_id in intersection_dag.adj[inter_i].keys():
                    m.addConstr(x_vars.sum(s.id) + x_vars.sum(inter_i) + per_stroke_descendants_vars.sum(inter_i, d_id) <= 2 + per_stroke_descendants_vars.sum(s.id, d_id))
    for s in strokes:
        if len(per_stroke_descendants[str(s.id)]) > 0:
            descendants_sum += (1-a_vars[s.id])*per_stroke_descendants_vars.sum(s.id, "*")

    # extrusion line well constructed
    construction_sum = []
    construction_anchored_sum = []
    construction_pos_sum = []
    construction_half_anchored_sum = []
    feature_line_sum = []
    feature_line_anchored_sum = []
    feature_line_pos_sum = []
    feature_line_half_anchored_sum = []
    extrude_anchoring = []
    fillet_anchoring = []

    #for s in strokes:
    #    if s.type == "sketch" or s.type == "feature_line":
    #        feature_line_sum.append(x_vars[s.id])
    #        feature_line_anchored_sum.append(a_vars[s.id])
    #        feature_line_pos_sum.append(pos_vars[s.id])
    #        feature_line_half_anchored_sum.append(half_a_vars[s.id])
    #    elif s.type == "extrude_line":
    #        extrude_anchoring.append(a_vars[s.id])
    #    elif s.type == "fillet_line":
    #        fillet_anchoring.append(a_vars[s.id])
    #    else:
    #        #construction_sum.append(stroke_lengths[s.id]*x_vars[s.id])
    #        construction_sum.append(x_vars[s.id])
    #        construction_pos_sum.append(x_vars[s.id] * (1 - pos_vars[s.id]))
    #        if len(s.anchor_intersections) == 1:
    #            construction_half_anchored_sum.append(x_vars[s.id]*(1-half_a_vars[s.id]))
    #        else:
    #            construction_anchored_sum.append(x_vars[s.id] * (1 - a_vars[s.id]))

#{'visibility_sum': 61.81322742671206, 'construction_sum': 1.0, 'construction_sum_half': 0.0, 'construction_sum_pos': 6.0, 'construction_sum_tan_var': 3.0, 'occlusion_sum': 2695.0, 'midpoint_sum': 0.0, 'descendants_sum': 3.0}
#{'visibility_sum': 61.81322742671206, 'construction_sum': 1.0, 'construction_sum_half': 0.0, 'construction_sum_pos': 9.0, 'construction_sum_tan_var': 2.0, 'occlusion_sum': 2695.0, 'midpoint_sum': 0.0, 'descendants_sum': 4.0}
#
#
    visibility_sum = gp.LinExpr()
    construction_sum = gp.LinExpr()
    construction_sum_half = gp.LinExpr()
    construction_sum_pos = gp.LinExpr()
    construction_sum_tan_var = gp.LinExpr()
    occlusion_sum = o_vars.sum()
    trivial_solution_sum = gp.LinExpr()

    need_construction_stroke_ids = []
    need_construction_pos_stroke_ids = []
    for s in strokes:
        visibility_sum += x_vars[s.id]*visibility_scores[s.id]
        # DEBUG
        #x_vars[s.id].ub = 1
        #x_vars[s.id].lb = 1
        construction_coeff = 1.0
        #construction_coeff = visibility_scores[s.id]

        if per_stroke_mutually_exlusive_set_vars[s.id] >= 2:
            construction_sum_pos += construction_coeff*x_vars[s.id]*(1-pos_vars[s.id])
            need_construction_pos_stroke_ids.append(s.id)

        #for i in s.anchor_intersections:
        #    if len(i) >= 2:
        #        construction_sum_pos += construction_coeff*x_vars[s.id]*(1-pos_vars[s.id])
        #        need_construction_pos_stroke_ids.append(s.id)
        #        break
        if len(s.anchor_intersections) > 1 and len(ellipse_fittings[s.id]) == 0:
            construction_sum += construction_coeff*x_vars[s.id]*(1-a_vars[s.id])
            need_construction_stroke_ids.append(s.id)
        elif len(s.anchor_intersections) > 0 and len(ellipse_fittings[s.id]) == 0:
            construction_sum_half += construction_coeff*x_vars[s.id]*(1-half_a_vars[s.id])
        if len(s.tangent_intersections) > 0:
            construction_sum_tan_var += construction_coeff*x_vars[s.id]*(1-tan_vars[s.id])

        if not s.original_feature_line:
            trivial_solution_sum += 1e-5*x_vars[s.id]

        for const_id in constraints.keys():
            const = constraints[const_id]
            if s.id in const["affected_stroke_ids"]:

                #print(const)
                #m.addConstr(a_vars[s.id] <= projection_constraint_vars.sum(int(const_id)))
                if const["type"] == "projection":
                    m.addConstr(a_vars[s.id] <= projection_constraint_vars.sum(int(const_id)))
                #    print("add projection")
                #    #construction_sum_tan_var += construction_coeff*x_vars[s.id]*(1-projection_constraint_vars.sum(int(const_id)))
                #    m.addConstr(a_vars[s.id] <= projection_constraint_vars.sum(int(const_id)))
                if const["type"] == "midpoint":
                    #const_vars[const_id]["midpoint_vars"].sum("main")
                    #print(const)
                    print(s.id)
                    m.addConstr(a_vars[s.id] <= const_vars[const_id]["midpoint_vars"].sum("main"))
                    #if s.id == 76:
                    #    m.addConstr(const_vars[const_id]["midpoint_vars"].sum("main") == 1)
                #    construction_sum_tan_var += construction_coeff*x_vars[s.id]*(1-const_vars[const_id]["midpoint_vars"].sum("main"))

    #m.addConstr(a_vars[74] == 1)
    #m.addConstr(x_vars[67] == 1)
    #m.addConstr(x_vars[68] == 1)
    #m.addConstr(x_vars[59] == 1)
    #m.addConstr(x_vars[60] == 1)
    #m.addConstr(x_vars[61] == 1)
    #m.addConstr(x_vars[69] == 1)
    #m.addConstr(x_vars[71] == 1)
    #m.addConstr(x_vars[74] == 1)
    #m.addConstr(x_vars[75] == 1)
    #m.addConstr(x_vars[76] == 1)
    #m.addConstr(x_vars[77] == 1)
    #m.addConstr(x_vars[105] == 1)
    #m.addConstr(x_vars[107] == 1)
    #{'visibility_sum': 26.737448719948706, 'construction_sum': 0.0, 'construction_sum_half': 0.0, 'construction_sum_pos': 0.0, 'construction_sum_tan_var': 0.0, 'occlusion_sum': 59.0, 'midpoint_sum': 0.0, 'descendants_sum': 0.0}
    #{'visibility_sum': 26.737448719948706, 'construction_sum': 0.0, 'construction_sum_half': 0.0, 'construction_sum_pos': 0.0, 'construction_sum_tan_var': 0.0, 'occlusion_sum': 56.0, 'midpoint_sum': 0.0, 'descendants_sum': 0.0}

    midpoint_sum = gp.quicksum([const_vars[const_id]["midpoint_vars"].sum("main") for const_id in const_vars.keys()])
    obj = lambda_0*visibility_sum \
          -lambda_1*(construction_sum + 0.5*construction_sum_half + 0.5*construction_sum_pos + construction_sum_tan_var + descendants_sum) \
          -lambda_2*occlusion_sum - trivial_solution_sum # + 100.0*midpoint_sum
    #obj = lambda_0*gp.quicksum(feature_line_anchored_sum) + 0.5*lambda_0*gp.quicksum(feature_line_half_anchored_sum) + \
    #      lambda_0*gp.quicksum(feature_line_pos_sum) - \
    #      lambda_1*gp.quicksum(construction_anchored_sum) - 0.5*lambda_1*gp.quicksum(construction_half_anchored_sum) - \
    #      lambda_1 * gp.quicksum(construction_pos_sum) - \
    #      lambda_2*gp.quicksum(construction_sum) + \
    #      lambda_3*(gp.quicksum(extrude_anchoring) + projection_constraint_vars.sum() + gp.quicksum(fillet_anchoring)) + \
    #      100*tan_vars.sum("*") + 100.0*visibility_sum
    #lambda_2*gp.quicksum(construction_sum) + lambda_3*(gp.quicksum(extrude_anchoring))
          #lambda_2*gp.quicksum(construction_sum) + lambda_3*projection_constraint_vars.sum()

    m.setObjective(obj, sense=GRB.MAXIMIZE)


    solve_time = process_time()
    m.optimize()
    if m.status == 9:
        print("timeLimit reached")
        return [], 0, 0, 0, 0, 0
    #print("solve time", process_time() - solve_time)
    #print("obj value", m.getObjective().getValue())
    solution_terms = {
        "visibility_sum": visibility_sum.getValue(),
        "construction_sum": construction_sum.getValue(),
        "construction_sum_half": construction_sum_half.getValue(),
        "construction_sum_pos": construction_sum_pos.getValue(),
        "construction_sum_tan_var": construction_sum_tan_var.getValue(),
        "occlusion_sum": occlusion_sum.getValue(),
        "midpoint_sum": midpoint_sum.getValue(),
        "descendants_sum": descendants_sum.getValue(),
    }
    #print(solution_terms)
    #solution_terms = {
    #    "feature_line_anchored_sum": gp.quicksum(feature_line_anchored_sum).getValue(),
    #    "feature_line_half_anchored_sum": gp.quicksum(feature_line_half_anchored_sum).getValue(),
    #    "feature_line_pos_sum": gp.quicksum(feature_line_pos_sum).getValue(),
    #    "tangent_sum": gp.quicksum(tan_vars).getValue(),
    #    "construction_anchored_sum": gp.quicksum(construction_anchored_sum).getValue(),
    #    "construction_half_anchored_sum": gp.quicksum(construction_half_anchored_sum).getValue(),
    #    "construction_pos_sum": gp.quicksum(construction_pos_sum).getValue(),
    #    "construction_sum": gp.quicksum(construction_sum).getValue(),
    #    "projection_constraint_sum": projection_constraint_vars.sum().getValue(),
    #    "visibility_sum": visibility_sum.getValue()
    #}
    #print(solution_terms)
    #print(const_vars)
    solution = m.getAttr('x', projection_constraint_vars)
    #print("projection_constraint_vars")
    #print(projection_constraint_indices)
    for corr in solution:
        if not np.isclose(solution[corr], 1.0):
            continue
        #print(corr, solution[corr])
    for const_id in const_vars.keys():
        solution = m.getAttr('x', const_vars[const_id]["midpoint_x_vars"])
        for corr in solution:
            if not np.isclose(solution[corr], 1.0):
                continue
        solution = m.getAttr('x', const_vars[const_id]["midpoint_triple_intersection_vars"])
        #print("midpoint_triple_intersection_vars")
        for corr in solution:
            if not np.isclose(solution[corr], 1.0):
                continue
            #print(corr, solution[corr])
    a_constructed = []
    half_a_constructed = []
    pos_constructed = []
    descendants_vars = []
    solution = m.getAttr('x', half_a_vars)
    for corr in solution:
        if not np.isclose(solution[corr], 1.0):
            continue
        half_a_constructed.append(int(corr))
    solution = m.getAttr('x', per_stroke_descendants_vars)
    for corr in solution:
        if not np.isclose(solution[corr], 1.0):
            continue
        descendants_vars.append(corr)
        #print("half_a", corr)
    #print("descendants_vars")
    #print(descendants_vars)
    #solution = m.getAttr('x', i_vars)
    #for corr in solution:
    #    if not np.isclose(solution[corr], 1.0):
    #        continue
    #    #print("i", corr)
    solution = m.getAttr('x', x_vars)
    selected_stroke_ids = []
    for corr in solution:
        if not np.isclose(solution[corr], 1.0):
            continue
        selected_stroke_ids.append(corr)
        #print("x", corr)
    solution = m.getAttr('x', a_vars)
    for corr in solution:
        if not np.isclose(solution[corr], 1.0):
            #if int(corr) in selected_stroke_ids and int(corr) in need_construction_stroke_ids:
            #    print("not well constructed stroke ", int(corr))
            #    #exit()
            continue
        a_constructed.append(int(corr))
        #print("a", corr)
    solution = m.getAttr('x', pos_vars)
    for corr in solution:
        if not np.isclose(solution[corr], 1.0):
            #if int(corr) in selected_stroke_ids and int(corr) in need_construction_pos_stroke_ids:
            #    print("not well pos constructed stroke ", int(corr))
            ##print(corr, solution[corr])
            continue
        pos_constructed.append(int(corr))
        #print("pos", corr)
    path_solutions = {}
    for ext_line_id in path_vars_dict.keys():
        path_solutions[ext_line_id] = []
        solution = m.getAttr('x', path_vars_dict[ext_line_id])
        for corr in solution:
            if not np.isclose(solution[corr], 1.0):
                continue
            path_solutions[ext_line_id].append(corr)
            #print("d", corr)
    #print(m.getObjective())
    #print(m.getVars())
    solution = m.getAttr('x', projection_constraint_vars)
    for corr in solution:
        if not np.isclose(solution[corr], 1.0):
            continue
        #print("pc", corr)

    # return selected stroke-ids
    return np.unique(selected_stroke_ids), np.unique(half_a_constructed), np.unique(a_constructed), np.unique(pos_constructed), \
           path_solutions, coplanar_graphs

if __name__ == "__main__":
    abc_id = 24
    theta = 60
    phi = 50
    data_folder = os.path.join("data", str(abc_id))
    all_edges_file_name = os.path.join(data_folder, "filtered_all_edges.json")
    with open(all_edges_file_name, "r") as f:
        all_edges = json.load(f)
    strokes_dict_file_name = os.path.join(data_folder, "strokes_dict.json")
    with open(strokes_dict_file_name, "r") as f:
        strokes_dict = json.load(f)
    strokes = [utils.Stroke(id=s["id"], intersections=s["intersections"], planes=s["planes"], type=s["type"],
                            previous_strokes=s["previous_strokes"], anchor_intersections=s["anchor_intersections"],
                            overlapping_stroke_ids=s["overlapping_stroke_ids"]) for s in strokes_dict]
    for s in strokes:
        print(s)
    prev_half_a_constructed = []
    prev_a_constructed = []
    prev_pos_constructed = []
    for lambda_0 in range(1, 10):
        runtime = process_time()
        selected_stroke_ids, half_a_constructed, a_constructed, pos_constructed = declutter(strokes, lambda_0=lambda_0)
        print("runtime: ", process_time()-runtime, "s")
        print(len(selected_stroke_ids))
        print(selected_stroke_ids)
        new_id_mapping = np.zeros(np.max(selected_stroke_ids)+1, dtype=int)
        for pos, i in enumerate(selected_stroke_ids):
            new_id_mapping[i] = pos
        half_a_constructed = new_id_mapping[half_a_constructed]
        a_constructed = new_id_mapping[a_constructed]
        pos_constructed = new_id_mapping[pos_constructed]
        new_half_a_constructed = [i for i in half_a_constructed if not i in prev_half_a_constructed]
        prev_half_a_constructed = deepcopy(half_a_constructed)
        new_a_constructed = [i for i in a_constructed if not i in prev_a_constructed]
        prev_a_constructed = deepcopy(a_constructed)
        new_pos_constructed = [i for i in pos_constructed if not i in prev_pos_constructed]
        prev_pos_constructed = deepcopy(pos_constructed)
        #print("selected_stroke_ids", selected_stroke_ids)
        #final_curves = [edge["geometry"] for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
        #construction_lines = [edge["geometry"] for edge_id, edge in enumerate(all_edges)
        #                      if edge_id in selected_stroke_ids and edge["type"] != "sketch" and edge["type"] != "feature_line"]
        #feature_lines = [edge["geometry"] for edge_id, edge in enumerate(all_edges)
        #                 if edge_id in selected_stroke_ids and ((edge["type"] == "sketch") or (edge["type"] == "feature_line"))]
        #final_edges = [edge for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
        #for new_s_id, s_id in enumerate(selected_stroke_ids):
        #    print(s_id, new_s_id)

        # prepare_decluttering(all_edges)
        final_edges = [edge for edge_id, edge in enumerate(all_edges) if edge_id in selected_stroke_ids]
        decluttered_strokes_file_name = os.path.join("data", str(abc_id), "decluttered_lambda0_"+str(lambda_0)+".json")
        final_edges_dict = {}
        for edge_id, edge in enumerate(final_edges):
            final_edges_dict[edge_id] = list(edge["geometry"])
            with open(decluttered_strokes_file_name, "w") as fp:
                json.dump(final_edges_dict, fp)
        # for edge_id, edge in enumerate(final_edges):
        #    print(edge_id, edge["type"])
        svg_file_name = os.path.join("data", str(abc_id), "decluttered_lambda0_"+str(lambda_0)+".svg")
        typed_feature_lines_to_svg(deepcopy(final_edges), svg_file_name=svg_file_name,
                                   theta=theta, phi=phi, title="Final drawing")
        svg_file_name = os.path.join("data", str(abc_id), "decluttered_lambda0_"+str(lambda_0)+"_half_a.svg")
        indexed_lines_to_svg(deepcopy(final_edges), half_a_constructed, svg_file_name=svg_file_name,
                             theta=theta, phi=phi, title="Final drawing")
        svg_file_name = os.path.join("data", str(abc_id), "decluttered_lambda0_" + str(lambda_0) + "_a.svg")
        indexed_lines_to_svg(deepcopy(final_edges), a_constructed, svg_file_name=svg_file_name,
                             theta=theta, phi=phi, title="Final drawing")
        os.system("rsvg-convert -f pdf " + svg_file_name + " > " + svg_file_name.replace("svg", "pdf"))
        svg_file_name = os.path.join("data", str(abc_id), "decluttered_lambda0_" + str(lambda_0) + "_pos.svg")
        indexed_lines_to_svg(deepcopy(final_edges), pos_constructed, svg_file_name=svg_file_name,
                             theta=theta, phi=phi, title="Final drawing")
    #typed_feature_lines_to_svg_successive(deepcopy(final_edges), svg_file_name=svg_file_name,
    #                                      theta=theta, phi=phi)
    #pdf_file_name = os.path.join("data", str(abc_id), "final_output.pdf")
    #os.system("rsvg-convert -f pdf " + svg_file_name + " > " + pdf_file_name)
