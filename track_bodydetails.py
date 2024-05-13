import collections
import json

# bd_0: bodydetails before feature
# bd_1: bodydetails after feature
import numpy as np
import polyscope as ps

# use for knowing which lines have been newly created
def get_new_feature_lines(bd_0, bd_1):
    new_edge_ids = []
    for body_1 in bd_1:
        for new_edge in body_1["edges"]:
            found_edge = False
            # if bd_0 is empty
            #if not "edges" in bd_0.keys():
            if len(bd_0) == 0:
                new_edge_ids.append(new_edge["id"])
                continue
            for body_0 in bd_0:
                for old_edge in body_0["edges"]:
                    if new_edge["id"] == old_edge["id"]:
                        found_edge = True
                        break
                if found_edge:
                    break
            if not found_edge:
                new_edge_ids.append(new_edge["id"])
    return new_edge_ids

def get_curves_brep(bodydetails_pool, new_edge_ids):
    curves_brep = {}
    for body in bodydetails_pool:
        for bd in body:
            for new_edge in bd["edges"]:
                if new_edge["id"] in new_edge_ids:
                    curves_brep[new_edge["id"]] = new_edge
    return curves_brep

# use for drawing current step
def get_new_and_modified_feature_lines(bd_0, bd_1):
    new_edge_ids = []
    new_edge_body_ids = []
    for body_1 in bd_1:
        for new_edge in body_1["edges"]:
        #for new_edge in bd_1["edges"]:
            found_edge = False
            # if bd_0 is empty
            if len(bd_0) == 0:
            #if not "edges" in bd_0.keys():
                new_edge_ids.append(new_edge["id"])
                new_edge_body_ids.append(body_1["id"])
                continue
            for body_0 in bd_0:
                for old_edge in body_0["edges"]:
                #for old_edge in bd_0["edges"]:
                    if new_edge["curve"]["type"] == old_edge["curve"]["type"]:
                    #if new_edge["id"] == old_edge["id"]:
                        # TODO: check if geometry has been modified
                        if new_edge["curve"]["type"] == "line" and \
                                np.all(np.isclose(new_edge["geometry"]["startPoint"], old_edge["geometry"]["startPoint"])) and \
                                np.all(np.isclose(new_edge["geometry"]["endPoint"], old_edge["geometry"]["endPoint"])):
                            found_edge = True
                            break
                        elif new_edge["curve"]["type"] == "circle" \
                            and np.all(np.isclose(new_edge["curve"]["origin"], old_edge["curve"]["origin"])) and \
                                np.isclose(new_edge["curve"]["radius"], old_edge["curve"]["radius"]) and \
                                np.isclose(new_edge["geometry"]["arcSweep"], old_edge["geometry"]["arcSweep"]):
                            found_edge = True
                            break
                        elif new_edge["curve"]["type"] == "ellipse" \
                                and np.all(np.isclose(new_edge["curve"]["origin"], old_edge["curve"]["origin"])) and \
                                np.isclose(new_edge["curve"]["majorRadius"], old_edge["curve"]["majorRadius"]) and \
                                np.isclose(new_edge["curve"]["minorRadius"], old_edge["curve"]["minorRadius"]) and \
                                np.all(np.isclose(new_edge["curve"]["majorAxis"], old_edge["curve"]["majorAxis"])) and \
                                np.isclose(new_edge["geometry"]["arcSweep"], old_edge["geometry"]["arcSweep"]):
                            found_edge = True
                            break
                if found_edge:
                    break
            if not found_edge:
                new_edge_ids.append(new_edge["id"])
                new_edge_body_ids.append(body_1["id"])
    return new_edge_ids, new_edge_body_ids

if __name__ == "__main__":
    ps.init()
    for i in range(1, 10):
        with open("bodydetails"+str(i+1)+".json", "r") as f:
            bd_0 = json.load(f)["bodies"]
        with open("bodydetails"+str(i+2)+".json", "r") as f:
            bd_1 = json.load(f)["bodies"]
        with open("data/1/feature_lines_"+str(i)+".json", "r") as f:
            geom_0 = json.load(f)
        with open("data/1/feature_lines_"+str(i+1)+".json", "r") as f:
            geom_1 = json.load(f)
        new_edge_ids = get_new_and_modified_feature_lines(bd_0, bd_1)

        for body_1 in bd_1:
            for edge in body_1["edges"]:
                if edge["id"] in new_edge_ids:
                    continue
                curve_geom = geom_1[edge["id"]]
                if len(curve_geom) == 1:
                    edges_array = np.array([[0, 0]])
                else:
                    edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
                ps.register_curve_network(edge["id"], nodes=np.array(curve_geom),
                                          edges=edges_array, color=(0, 0, 1))
        for edge_id in new_edge_ids:
            curve_geom = geom_1[edge_id]
            if len(curve_geom) == 1:
                edges_array = np.array([[0, 0]])
            else:
                edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
            ps.register_curve_network(edge_id, nodes=np.array(curve_geom),
                                      edges=edges_array, color=(1, 0, 0))
        ps.show()
