import os, json
from pylowstroke.sketch_core import Sketch, Stroke, StrokePoint
from pylowstroke.sketch_io import SketchSerializer as skio
import matplotlib.pyplot as plt
import numpy as np

ours_feature_line_types = ["feature_line", "silhouette_line"]
ours_construction_line_types = ["extrude_line", "sketch", "grid_lines", "fillet_line", "section_lines", "circle_square_line"]
feature_line_types = [0, 1, 2, 3, 4, 5]
construction_line_types = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

silhouette = [0]
vis_edges = [1, 2]
occ_edges = [3, 4]
scaffold = construction_line_types

line_types = {
         0: 'silhouette smooth' ,
         1: 'ridges visible' ,
         2: 'vallleys visible' ,
         3: 'ridges occluded',
         4: 'valleys occluded',
         5: 'discriptive cross sections',
         6: 'axis and grids',
         7: 'scaffolds',
         8: 'scaffolds: lines to VP',
         9: 'scaffolds: square for an ellipse',
        10: 'scaffolds: tangents',
        11: 'surfacing: cross section',
        12: 'surfacing: cross section scaffold',
        13: 'surfacing: mirroring',
        14: 'surfacing: temporary plane',
        15: 'surfacing: projection lines',
        16: 'proportions div. rectangle2',
        17: 'proportions div. rectangle3',
        18: 'proportions div. ellipse',
        19: 'proportions mult. rectangle',
        20: 'hinging and rotating elements',
        21: 'ticks',
        22: 'hatching',
        23: 'text',
        24: 'outline',
        25: 'shadow construction',
        26: 'background'
}

def count_labels_fast(labels_opensketch, labels_ours):
    ref_counter = {
        "vis_edges": 0,
        "occ_edges": 0,
        "scaffold": 0,
        "silhouette": 0,
    }
    ours_counter = {
        "vis_edges": 0,
        "occ_edges": 0,
        "scaffold": 0,
        "silhouette": 0,
    }
    for l in labels_opensketch:
        if l in vis_edges:
            ref_counter["vis_edges"] += 1
        elif l in occ_edges:
            ref_counter["occ_edges"] += 1
        elif l in scaffold:
            ref_counter["scaffold"] += 1
        elif l in silhouette:
            ref_counter["silhouette"] += 1

    for l in labels_ours:
        if l in ["vis_edge"]:
            ours_counter["vis_edges"] += 1
        elif l in ["occ_edge"]:
            ours_counter["occ_edges"] += 1
        elif l in ["scaffold"]:
            ours_counter["scaffold"] += 1
        elif l in ["silhouette"]:
            ours_counter["silhouette"] += 1

    #print(labels_opensketch)
    #print(labels_ours)
    num_ref = ref_counter["vis_edges"]+ref_counter["occ_edges"]+ref_counter["scaffold"]+ref_counter["silhouette"]
    if num_ref == 0:
        num_ref = 1
    num_ours = ours_counter["vis_edges"]+ours_counter["occ_edges"]+ours_counter["scaffold"]+ours_counter["silhouette"]
    if num_ours == 0:
        num_ours = 1

    ref_ratios = np.array([ref_counter["vis_edges"]/num_ref, ref_counter["occ_edges"]/num_ref,
                           ref_counter["scaffold"]/num_ref, ref_counter["silhouette"]/num_ref])
    ours_ratios = np.array([ours_counter["vis_edges"]/num_ours, ours_counter["occ_edges"]/num_ours,
                            ours_counter["scaffold"]/num_ours, ours_counter["silhouette"]/num_ours])

    #print(ref_ratios)
    #print(ours_ratios)
    ratio_dist = np.linalg.norm(ref_ratios-ours_ratios)
    return ratio_dist, ref_ratios, ours_ratios

def count_labels(labels_file, decluttered_file):
    ref_feature_line_counter = 0
    ref_construction_line_counter = 0
    with open(labels_file, "r") as fp:
        labels = json.load(fp)["strokes_line_types"]
    for l in labels:
        if l in feature_line_types:
            ref_feature_line_counter += 1
        if l in construction_line_types:
            ref_construction_line_counter += 1

    ours_feature_line_counter = 0
    ours_construction_line_counter = 0
    with open(decluttered_file, "r") as fp:
        ours_strokes = json.load(fp)
    ours_labels = [s["line_type"] for s in ours_strokes.values()]
    for l in ours_labels:
        if l in ours_feature_line_types:
            ours_feature_line_counter += 1
        if l in ours_construction_line_types:
            ours_construction_line_counter += 1

    print(labels)
    print(ours_labels)

    ref_construction_ratio = ref_construction_line_counter/(ref_feature_line_counter+ref_construction_line_counter)
    ref_feature_ratio = ref_feature_line_counter/(ref_feature_line_counter+ref_construction_line_counter)

    ours_construction_ratio = ours_construction_line_counter/(ours_feature_line_counter+ours_construction_line_counter)
    ours_feature_ratio = ours_feature_line_counter/(ours_feature_line_counter+ours_construction_line_counter)
    print(ref_feature_line_counter, ref_construction_line_counter, ref_feature_ratio, ref_construction_ratio)
    print(ours_feature_line_counter, ours_construction_line_counter, ours_feature_ratio, ours_construction_ratio)

if __name__ == "__main__":
    designer_name = "Professional2"
    designer_name = "student9"
    designer_name = "student8"

    object_name = "vacuum_cleaner"
    object_name = "house"

    decluttered_file = os.path.join("data", designer_name+"_"+object_name, "60_125_1.4/decluttered_lambda0_1.0.json")
    decluttered_file = os.path.join("data", designer_name+"_"+object_name, "60_125_1.4/final_edges.json")
    edges_file = os.path.join("data", designer_name+"_"+object_name, "60_125_1.4/unique_edges.json")

    ours_sketch_file_name = os.path.join("data", designer_name+"_"+object_name, "60_125_1.4/decluttered_lambda0_1.0.svg")

    # load concept sketch
    labels_file = os.path.join("../sketches_labeling_first_viewpoint", designer_name, object_name, "strokes_lines_types_view1_concept.json")
    sketch_file = os.path.join("../sketches_json_first_viewpoint", designer_name, object_name, "view1_concept.json")
    with open(labels_file, "r") as fp:
        labels = json.load(fp)["strokes_line_types"]
    with open(sketch_file, "r") as fp:
        sketch = json.load(fp)
    print(sketch.keys())
    print(len(sketch["strokes"]))
    print(sketch["strokes"][0].keys())
    strokes = [s for s in sketch["strokes"] if not s["is_removed"]]
    sketch = Sketch()
    for s_id, s in enumerate(strokes):
        if not type(s["points"]) == dict and len(s["points"]) > 1:
            new_s = Stroke([])
            new_s.label = labels[s_id]
            points_list = []
            for p in s["points"]:
                new_p = StrokePoint(p["x"], p["y"])
                print(p["p"])
                new_p.add_data("pressure", p["p"])
                points_list.append(new_p)
            #new_s.from_array([[p["x"], p["y"]] for p in s["points"]])
            #sketch.strokes.append(new_s)
            new_s = Stroke(points_list)
            new_s.label = labels[s_id]
            new_s.add_avail_data("pressure")
            sketch.strokes.append(new_s)

    # load presentation sketch
    pres_labels_file = os.path.join("../sketches_labeling_first_viewpoint", designer_name, object_name, "strokes_lines_types_view1_presentation.json")
    pres_sketch_file = os.path.join("../sketches_json_first_viewpoint", designer_name, object_name, "view1_presentation.json")
    with open(pres_labels_file, "r") as fp:
        pres_labels = json.load(fp)["strokes_line_types"]
    with open(pres_sketch_file, "r") as fp:
        pres_sketch = json.load(fp)
    pres_strokes = [s for s in pres_sketch["strokes"] if not s["is_removed"]]
    pres_sketch = Sketch()
    for s_id, s in enumerate(pres_strokes):
        if pres_labels[s_id] > 20:
            continue
        if not type(s["points"]) == dict and len(s["points"]) > 1:
            new_s = Stroke([])
            new_s.label = pres_labels[s_id]
            new_s.from_array([[p["x"], p["y"]] for p in s["points"]])
            pres_sketch.strokes.append(new_s)
    print(len(pres_strokes))
    print(len(pres_labels))

    ours_sketch = skio.load(ours_sketch_file_name)

    with open(decluttered_file, "r") as fp:
        ours_sketch_data = json.load(fp)
    with open(edges_file, "r") as fp:
        unique_edges_data = json.load(fp)

#    # find out which stroke belongs to the final shape
#    max_feature_id = 0
#    for s_id in range(len(ours_sketch.strokes)):
#        max_feature_id = max(max_feature_id, np.max([l["feature_id"] for l in unique_edges_data[s_id]["original_labels"]]))
#
#    for s_id in range(len(ours_sketch.strokes)):
#        #original_s_id = ours_sketch_data[str(s_id)]["id"]
#        ours_sketch.strokes[s_id].label = ours_sketch_data[str(s_id)]["line_type"]
#        ours_sketch.strokes[s_id].feature_id = ours_sketch_data[str(s_id)]["feature_id"]
#        ours_sketch.strokes[s_id].max_feature_id = np.max([l["feature_id"] for l in ours_sketch_data[str(s_id)]["original_labels"]])
#        #max_feature_id = max(max_feature_id, unique_edges_data[s_id]["feature_id"])

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                        bottom=0.0,
                        top=1.0)
    sketch.display_strokes_2(fig, ax[0],
                             linewidth_data=lambda p: p.get_data("pressure")+0.5,
                             color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
                                                      for p in s.points_list] if not s.label in silhouette else "blue")
    ax[0].set_xlim(0, sketch.width)
    ax[0].set_ylim(sketch.height, 0)
    ax[0].set_aspect("equal")
    ax[0].axis("off")

    #pres_sketch.display_strokes_2(fig, ax[1],
    #                         color_process=lambda s: "black")
    #ax[1].set_xlim(0, pres_sketch.width)
    #ax[1].set_ylim(pres_sketch.height, 0)
    #ax[1].set_aspect("equal")
    #ax[1].axis("off")

    ours_sketch.display_strokes_2(fig, ax[1],
                                  color_process=lambda s: "red")
                                  #if s.max_feature_id == max_feature_id else "blue")
    ax[1].set_xlim(0, ours_sketch.width)
    ax[1].set_ylim(ours_sketch.height, 0)
    ax[1].set_aspect("equal")
    ax[1].axis("off")

    #plt.gca().invert_yaxis()
    plt.show()
    count_labels(labels_file, decluttered_file)
