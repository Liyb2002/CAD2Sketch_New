import json, os, sys
import numpy as np
from utils import spherical_to_cartesian_coords

per_corner_angle_dict = {
    (1, 1): {"theta": 45, "phi": 45},
    (-1, 1): {"theta": 45, "phi": 135},
    (-1, -1): {"theta": 45, "phi": -135},
    (1, -1): {"theta": 45, "phi": -45}}

# search for best 3/4 viewpoint, given the CAD operations of the sequence
def get_best_viewpoint(data_folder):
    with open(os.path.join(data_folder, "parsed_features.json")) as fp:
        parsed_features = json.load(fp)
    bd_pool = []
    bd_i = 0
    bd_name = os.path.join(data_folder, "bodydetails"+str(bd_i)+".json")
    while os.path.exists(bd_name):
        with open(bd_name, "r") as fp:
            bd = json.load(fp)
        if len(bd["bodies"]) > 0:
            bd_pool.append(bd["bodies"])

        bd_i += 1
        bd_name = os.path.join(data_folder, "bodydetails"+str(bd_i)+".json")

    # fetch all planes and edges modified by operations
    entities = []
    for op_id, op in enumerate(parsed_features["sequence"]):
        if not op["type"] in ["extrude", "fillet"]:
            continue
        entities.append({"plane_id": parsed_features["entities"][op["entity"]]["entities"],
                         "feat_id": op_id})

    #print(entities)
    normals = []
    # look in sketches
    for ent in parsed_features["entities"].values():
        if ent["type"] != "Sketch":
            continue
        for face in ent["profiles"]["faces"]:
            for tmp_ent in entities:
                if face["id"] in tmp_ent["plane_id"]:
                    if face["param"]["type"] == "Plane":
                        if tmp_ent["feat_id"] < 2:
                            normals.append(-np.array(face["param"]["normal"]))
                        else:
                            normals.append(np.array(face["param"]["normal"]))
    # look in bodydetails
    # for edge_ids, we'll have to look later for two adjacent faces
    edge_ids = {}

    for bds in reversed(bd_pool):
        for bd in bds:
            #print(bd)
            for face in bd["faces"]:
                for tmp_ent in entities:
                    if face["id"] in tmp_ent["plane_id"]:
                        #print(face)
                        if face["surface"]["type"] == "plane":
                            normals.append(face["surface"]["normal"])
            for edge in bd["edges"]:
                for tmp_ent in entities:
                    if edge["id"] in tmp_ent["plane_id"]:
                        edge_ids[edge["id"]] = {"face_ids": [], "normals": []}
    for bds in reversed(bd_pool):
        for bd in bds:
            for face in bd["faces"]:
                for loop in face["loops"]:
                    for coedge in loop["coedges"]:
                        if coedge["edgeId"] in edge_ids.keys():
                            if face["surface"]["type"] == "plane" and len(edge_ids[coedge["edgeId"]]["face_ids"]) < 2:
                                edge_ids[coedge["edgeId"]]["face_ids"].append(face["id"])
                                edge_ids[coedge["edgeId"]]["normals"].append(face["surface"]["normal"])

    # average normals
    for edge in edge_ids.values():
        normals.append(np.mean(edge["normals"], axis=0))
    averaged_normal = np.mean(normals, axis=0)[:2]
    averaged_normal /= np.linalg.norm(averaged_normal)

    corner_points = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    best_corner_point = corner_points[np.argmin(np.linalg.norm(corner_points-averaged_normal, axis=-1))]
    #print(best_corner_point)
    #print(spherical_to_cartesian_coords(2.0, np.deg2rad(45), np.deg2rad(45)))
    #print(spherical_to_cartesian_coords(2.0, np.deg2rad(45), np.deg2rad(135)))
    #print(spherical_to_cartesian_coords(2.0, np.deg2rad(45), np.deg2rad(-135)))
    #print(spherical_to_cartesian_coords(2.0, np.deg2rad(45), np.deg2rad(-45)))
    angles = per_corner_angle_dict[(int(best_corner_point[0]), int(best_corner_point[1]))]
    return angles["phi"]


if __name__ == "__main__":
    data_folder = os.path.join("data/884")
    phi = get_best_viewpoint(data_folder)
    print(phi)