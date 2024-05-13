import os, json
from copy import deepcopy
from render_shapes import features_lines_to_svg, typed_feature_lines_to_svg, \
    typed_feature_lines_to_svg_successive, indexed_lines_to_svg
import imageio
from matplotlib.patches import Rectangle
from scipy.linalg import lstsq
from scipy.optimize import nnls, lsq_linear
from render_training_data import project_lines_opengl
from shapely.geometry import LineString
import numpy as np
import matplotlib.pyplot as plt
import utils


def optimize_opacities(edges, stylesheet, cam_pos, obj_center, up_vec, mesh, multiplicative=True, VERBOSE=False):
    geometries = []
    for edge_id in range(len(edges.keys())):
        edge = edges[str(edge_id)]
        geometries.append(edge["geometry_3d"])
    # find out "style line types"
    max_feature_id = 0
    for s_id in range(len(edges)):
        max_feature_id = max(max_feature_id, np.max([l["feature_id"] for l in edges[str(s_id)]["original_labels"]]))
    style_labels = []
    for edge_id in edges.keys():
        edge = edges[edge_id]
        tmp_max_feature_id = np.max([l["feature_id"] for l in edge["original_labels"]])
        all_line_types = [l["type"] for l in edge["original_labels"]]
        if tmp_max_feature_id == max_feature_id and "silhouette_line" in all_line_types and edge["visibility_score"] > 0.9:
        #if tmp_max_feature_id == max_feature_id and "silhouette_line" in all_line_types:
            style_labels.append("silhouette_line")
            continue
        if tmp_max_feature_id == max_feature_id and ("feature_line" in all_line_types or edge["type"] == "feature_line"):
        #if tmp_max_feature_id == max_feature_id and "feature_line" in all_line_types:
            style_labels.append("feature_line")
            continue
        style_labels.append("grid_lines")
    # sample a target opacity per edge
    # DEBUG
    #target_opacities = {
    #    "grid_lines": 0.3,
    #    "feature_line": 0.8,
    #    "silhouette_line": 1.0
    #}
    #old_opacities = [target_opacities[s_l] for s_l in style_labels]
    print("style_labels")
    for i, s in enumerate(style_labels):
        print(i, s)

    if VERBOSE:
        vis_lines_file_name = os.path.join(".", "vis_lines.svg")
        indexed_lines_to_svg(deepcopy([{"geometry": edges[edge_id]["geometry_3d"],
                                        "type": edges[edge_id]["type"],
                                        "feature_id": edges[edge_id]["feature_id"]}
                                       for edge_id in edges.keys()]),
                                       [edge_id for edge_id in edges.keys() if style_labels[int(edge_id)] == "silhouette_line"],
                             cam_pos, obj_center, up_vec,
                             svg_file_name=vis_lines_file_name,
                             title="Visible feature lines")
    old_opacities = []
    print(stylesheet["opacities_per_type"])
    for edge_id, s_l in enumerate(style_labels):
        mu = None
        sigma = None
        if s_l == "grid_lines":
            mu = stylesheet["opacities_per_type"]["scaffold"]["mu"]
            sigma = stylesheet["opacities_per_type"]["scaffold"]["sigma"]
        elif s_l == "feature_line":
            if edges[str(edge_id)]["visibility_score"] > 0.5:
                mu = stylesheet["opacities_per_type"]["vis_edges"]["mu"]
                sigma = stylesheet["opacities_per_type"]["vis_edges"]["sigma"]
            else:
                mu = stylesheet["opacities_per_type"]["occ_edges"]["mu"]
                sigma = stylesheet["opacities_per_type"]["occ_edges"]["sigma"]
        elif s_l == "silhouette_line":
            mu = stylesheet["opacities_per_type"]["silhouette"]["mu"]
            sigma = stylesheet["opacities_per_type"]["silhouette"]["sigma"]
        #sampled_op = min(1.0, max(0.0, np.random.normal(loc=mu, scale=sigma/3, size=1)[0]))
        #sampled_op = mu
        sampled_op = min(1.0, max(0.0, np.random.normal(loc=mu, scale=sigma/5, size=1)[0]))
        #if s_l == "grid_lines":
        #    sampled_op = 0.05
        old_opacities.append(sampled_op)
    return old_opacities

    projected_lines = project_lines_opengl(geometries, (1024, 1024), cam_pos, obj_center, up_vec)
    style_labels = np.array(style_labels)
    #print(style_labels)

    lines = [LineString(p_l) for p_l in projected_lines]
    polys = [l.buffer(0.1) for l in lines]
    overlap_ratios = np.zeros([len(lines), len(lines)])

    columns = []
    rows = []
    a_rows = []
    b_rows = []

    for l1_id, l1 in enumerate(polys):
        for l2_id, l2 in enumerate(polys):
            if l2_id <= l1_id:
                continue
            if l2.intersects(l1):
                l2_ratio = l2.intersection(l1).area/l2.area
                l1_ratio = l2.intersection(l1).area/l1.area
                if l2_ratio > 0.3 or l1_ratio > 0.3:
                    overlap_ratios[l1_id][l2_id] = l1_ratio
                    overlap_ratios[l2_id][l1_id] = l2_ratio

    for l1_id, l1 in enumerate(polys):
        for l2_id, l2 in enumerate(polys):
            if overlap_ratios[l1_id][l2_id] < 0.1:
                continue
            inter = l2.intersection(l1)
            # get all other stroke-ids contained here
            inter_stroke_ids = [l1_id, l2_id]
            for l3_id, l3 in enumerate(polys):
                if overlap_ratios[l1_id][l3_id] < 0.1:
                    continue
                if inter.intersects(l3):
                    inter_stroke_ids.append(l3_id)
            inter_stroke_ids = np.unique(inter_stroke_ids)
            inter_style_labels = style_labels[inter_stroke_ids]
            #inter_opacity = np.max([target_opacities[i_s_l] for i_s_l in inter_style_labels])
            inter_opacity = np.max([old_opacities[i_s_id] for i_s_id in inter_stroke_ids])
            a_rows.append(inter_stroke_ids)
            b_rows.append(inter_opacity)
        if not np.any(np.isclose(overlap_ratios[l1_id], 1.0)):
            a_rows.append([l1_id])
            #b_rows.append(target_opacities[style_labels[l1_id]])
            b_rows.append(old_opacities[l1_id])
    #print(a_rows)
    #print(b_rows)

    a_mat = np.zeros([len(a_rows), len(lines)])
    b_mat = np.zeros([len(b_rows), 1])

    for i, a_row in enumerate(a_rows):
        a_mat[i, a_row] = 1.0
    for i, b_row in enumerate(b_rows):
        #b_mat[i] = b_row
        b_mat[i] = np.log(max(np.finfo(float).eps, 1-b_row))

    #p, res, rnk, s = lstsq(a_mat, b_mat.reshape(-1))
    #print(p)
    #x = lsq_linear(a_mat, b_mat.reshape(-1), (0.1, 1.0))
    # alpha_i should be in [0.1, 1.0]
    x = lsq_linear(a_mat, b_mat.reshape(-1), bounds=(-np.inf, -0.05536), verbose=0, method="bvls", tol=1e-20)
    #print(a_mat)
    #print(b_mat)
    #print(x["x"])
    #new_opacities = np.array(x["x"])
    new_opacities = []
    for op in np.array(x["x"]):
        new_op = 1.0 - np.exp(op)
        #new_opacities.append(1.0-new_op)
        new_opacities.append(new_op)

    if VERBOSE:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                            bottom=0.0,
                            top=1.0)

        for l1_id, l1 in enumerate(polys):
            ax[0].plot(np.array(lines[l1_id].coords)[:, 0],
                       np.array(lines[l1_id].coords)[:, 1], c="black", lw=4,
                       alpha=old_opacities[l1_id])
            ax[1].plot(np.array(lines[l1_id].coords)[:, 0],
                       np.array(lines[l1_id].coords)[:, 1], c="black", lw=4,
                       alpha=new_opacities[l1_id])

        for a in ax:
            a.set_aspect("equal")
            a.axis("off")
        #plt.show()
        plt.savefig(os.path.join("test.png"))
        plt.close(fig)
        opacity_map = imageio.imread("test.png")
        plt.imshow(opacity_map)
        plt.show()
        #for i in range(len(style_labels)):
        #    if not np.isclose(old_opacities[i], new_opacities[i]):
        #        print(i, old_opacities[i], new_opacities[i])

        #for l1_id, l1 in enumerate(polys):
        #    if np.any(np.isclose(overlap_ratios[l1_id], 1.0)):
        #        plt.plot(np.array(lines[l1_id].coords)[:, 0],
        #                 np.array(lines[l1_id].coords)[:, 1], c="red", lw=2)
        ## get all intersections and check which other strokes are intersected by this
        #for l1_id, l1 in enumerate(polys):
        #    if np.any(np.isclose(overlap_ratios[l1_id], 1.0)):
        #        plt.plot(np.array(lines[l1_id].coords)[:, 0],
        #                 np.array(lines[l1_id].coords)[:, 1], c="red", lw=2)
    return new_opacities

if __name__ == "__main__":

    #data_folder = os.path.join("data/student3_waffle_iron")
    #per_view_data_folder = os.path.join(data_folder, "60_125_1.4")
    #final_edges_file_name = os.path.join(per_view_data_folder, "final_edges.json")
    #with open(final_edges_file_name, "r") as fp:
    #    edges = json.load(fp)

    #mesh = utils.load_last_mesh(data_folder)
    #cam_pos, obj_center = utils.get_cam_pos_obj_center(mesh.vertices, radius=1.0, theta=90, phi=30)
    #for x in os.listdir(data_folder):
    #    if "camparam.json" in x:
    #        with open(os.path.join(data_folder, x), "r") as fp:
    #            cam_params = json.load(fp)["restricted"]
    #            #cam_params = json.load(fp)["general"]
    #            cam_pos = np.array(cam_params["C"]) - obj_center
    #            up_vec = np.array(cam_params["up"])
    #optimize_opacities(edges, {}, cam_pos, obj_center, up_vec, mesh)
    xy = (100, 100)
    a = 100
    a_1 = 0.2
    a_2 = 0.3
    a_3 = 0.25
    a_4 = 0.1
    alphas = [a_1, a_2, a_3, a_4]
    alphas = [a_1, a_2, a_3, a_4, 0.3, 0.5]
    #alphas = [a_1, a_2, a_3]
    boxes = [Rectangle(xy, a, a, alpha=a_i, fc=(0, 0, 0)) for a_i in alphas]
    alphas = np.array(alphas)
    alphas = np.random.permutation(alphas)
    print(alphas)
    final_alpha = np.prod([1-a_i for a_i in alphas])
    #final_alpha = 1 - (a_1 + a_2*(1-a_1) + a_3*(1-a_1)*(1-a_2) + a_4*(1-a_1)*(1-a_2)*(1-a_3))
    #for i in range(len(alphas)):
    #    print(i)
    #    print([1-a_i for a_i in alphas[:i]])
    #    print(alphas[i])
    #    print(alphas[i]*np.prod([1-a_i for a_i in alphas[:i]]))
    #print(np.sum([alphas[i]*np.prod([1-a_i for a_i in alphas[:i]]) for i in range(len(alphas))]))
    #final_alpha = 1 - np.sum([alphas[i]*np.prod([1-a_i for a_i in alphas[:i]]) for i in range(len(alphas))])
    print(final_alpha)
    print(255.0*final_alpha)
    boxes = [Rectangle(xy, a, a, alpha=a_i, fc=(0, 0, 0)) for a_i in alphas]
    final_box = Rectangle((200, 200), 50, 50, alpha=1-final_alpha, fc=(0, 0, 0))
    #box_2 = Rectangle(xy, a, a, alpha=a_2, fc=(0, 0, 0))
    #box_3 = Rectangle(xy, a, a, alpha=a_3, fc=(0, 0, 0))
    #box_4 = Rectangle(xy, a, a, alpha=a_4, fc=(0, 0, 0))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                        bottom=0.0,
                        top=1.0)
    for box in boxes:
        ax.add_artist(box)
    ax.add_artist(final_box)
    #ax.add_artist(box_3)
    ax.set_aspect("equal")
    ax.set_xlim(0, 300)
    ax.set_ylim(300, 0)
    ax.axis("off")
    #plt.show()
    plt.savefig(os.path.join("test.png"))
    plt.close(fig)
    opacity_map = imageio.imread("test.png")
    plt.imshow(opacity_map)
    plt.show()


