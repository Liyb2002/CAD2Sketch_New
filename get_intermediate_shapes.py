# TODO: loop through different CAD models
# TODO: output 3D sketches
# TODO: loop through different parts

import os.path
from trimesh import Trimesh
from trimesh.exchange.obj import export_obj
from trimesh.exchange.stl import load_stl
from collections import OrderedDict
import onshape.call as call
from onshape.client import Client
import json
import yaml
import polyscope as ps
import numpy as np
import seaborn as sns
import sys
#try:
#    import pymeshlab
#except:
#    print(sys.exc_info())
import igl
from parser import FeatureListParser

def copy_workspace(abc_url, new_name="new_doc"):

    did, wid, eid = call._parse_url(abc_url)
    # i) get element name
    elements = call.get_elements(url=abc_url,
                                 logging=False)
    ele_name = ""
    for ele in elements:
        if ele["id"] == eid:
            ele_name = ele["name"]
    if ele_name == "":
        print("element id not found")
        exit()

    resp = call.copy_workspace(url=abc_url,
                               new_name=new_name,
                               logging=False)
    new_did = resp["newDocumentId"]
    new_wid = resp["newWorkspaceId"]
    new_eid = -1
    # iii) get new element id
    new_url = "https://cad.onshape.com/documents/"+new_did+"/w/"+new_wid+"/e/0"
    elements = call.get_elements(url=new_url,
                                 logging=False)
    for ele in elements:
        if ele["name"] == ele_name:
            new_eid = ele["id"]
    if new_eid == -1:
        print("Workspace copy problem")
        exit()
    new_url = "https://cad.onshape.com/documents/"+new_did+"/w/"+new_wid+"/e/"+new_eid
    return new_url

def get_intermediate_shapes(abc_url, data_folder, VERBOSE=False):
    #ms = pymeshlab.MeshSet()

    if VERBOSE:
        ps.init()

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    #bbox = call.get_bbox(abc_url)
    #print(bbox)
    #entity = call.get_entity_by_id(abc_url, "JGB", "FACE")
    #print(entity)
    #exit()
    ##entity = call.get_entity_by_id(abc_url, "JGC", "FACE")

    #print(entity)
    #exit()
    #tess_sketch = call.get_tesselated_sketch(abc_url, "FY2psPHhtCzM1zr")
    #print(tess_sketch)
    #exit()
    #sketch_topo = call.get_sketch_topology(abc_url)
    #print(sketch_topo)
    #exit()
    #features = call.get_features(url=abc_url, logging=False)
    #with open("tmp.json", "w") as f:
    #    json.dump(features, f, indent=4)
    #for rollback in range(0, 10):
    #    bodydetails = call.get_bodydetails(abc_url, rollbackbarindex=rollback)
    #    with open("bodydetails"+str(rollback)+".json", "w") as f:
    #        json.dump(bodydetails, f, indent=4)
    #exit()
    #if VERBOSE:
    #    print("len(features)", len(features))

    sketch_info = call.get_info(url=abc_url, logging=False, output_3d=True)
    with open(os.path.join(data_folder, "sketch_info.json"), "w") as f:
        json.dump(sketch_info, f, indent=4)
    #exit()

    sketches_dict = {}
    for sketch in sketch_info["sketches"]:
        tess_sketch = call.get_tesselated_sketch(abc_url, sketch["featureId"])
        curves_3d = []
        if "sketchEntities" in tess_sketch.keys():
            for ent in tess_sketch["sketchEntities"]:
                #curves_3d.append(np.array(ent["tessellationPoints"]))
                #print(ent)
                curves_3d.append(ent["tessellationPoints"])
        sketches_dict[sketch["featureId"]] = curves_3d
        with open(os.path.join(data_folder, "sketch_"+sketch["featureId"]+".json"), "w") as f:
            json.dump({"curves": curves_3d}, f, indent=4)

    # get intermediate STLs
    #did, wid, eid = call._parse_url(abc_url)
    ## i) get element name
    #elements = call.get_elements(url=abc_url,
    #                           logging=False)
    #ele_name = ""
    #for ele in elements:
    #    if ele["id"] == eid:
    #        ele_name = ele["name"]
    #if ele_name == "":
    #    print("element id not found")
    #    exit()

    ## ii) copy workspace
    new_url = copy_workspace(abc_url)
    #new_url = abc_url
    v_list = new_url.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
    #if VERBOSE:
    #    print("new_url:", new_url)

    # iv) get features
    features = call.get_features(url=new_url, logging=False)
    with open(os.path.join(data_folder, "features.json"), "w") as fp:
        json.dump(features, fp, indent=4)

    c = Client(logging=False)
    parser = FeatureListParser(c, did, wid, eid, sketch_info, data_id=None)
    parsed_features = {"entities": OrderedDict(), "properties": {}, "sequence": []}
    try:
        bbox = parser._parse_boundingBox()
    except Exception as e:
        print(parser.data_id, "bounding box failed:", e)
        return parsed_features
    parsed_features["properties"].update({"bounding_box": bbox})

    if VERBOSE:
        print("len(features)", len(features))

    for feat_id, feat in enumerate(features["features"]):
        #print(feat["message"]["featureType"])
        # v) set rollback id
        print(feat_id)
        if VERBOSE:
            print("feat_id:", feat_id)
            print(feat)
        resp = call.feature_rollback(url=new_url,
                                     query={"rollbackIndex": feat_id+1,
                                            "serializationVersion": features["serializationVersion"],
                                            "sourceMicroversion": features["sourceMicroversion"]},
                                     logging=False)

        feat_data = feat['message']
        feat_type = feat_data['featureType']
        feat_Id = feat_data['featureId']
        #print(feat_type)
        #try:
        if feat_type == 'newSketch':
            #print(feat_data)
            # get corresponding sketch_info
            sketch_name = feat["message"]["name"]
            tmp_sketch_info = [s_i for s_i in sketch_info["sketches"] if s_i["sketch"] == sketch_name][0]
            feat_dict = parser._parse_sketch(feat_data)
            print(feat_data["name"])
            for k in feat_dict['profiles'].keys():
                parser.profile2sketch.update({k: feat_Id})
        elif feat_type == 'extrude':
            # feat_dict = self._parse_extrude(feat_data)
            feat_dict = parser._parse_extrude_simple(feat_data)
            # self.c.eval_extrude(self.did, self.wid, self.eid, feat_Id)
            # topo = self.c.eval_sketch_topology_by_adjacency(self.did, self.wid, self.eid, feat_Id)
        elif feat_type == 'fillet':
            feat_dict = parser._parse_fillet(feat_data)
        else:
            feat_dict = {"type": feat_type}
            # continue
            # raise NotImplementedError(self.data_id, "unsupported feature type: {}".format(feat_type))
        #except Exception as e:
        #    print(parser.data_id, "parse feature failed:", e)
        #    feat_dict = {"type": feat_type,
        #                 }
            #break
        parsed_features["entities"].update({feat_Id: feat_dict})
        parsed_features["sequence"].append({"index": feat_id, "type": feat_dict['type'], "entity": feat_Id})

        bodydetails = call.get_bodydetails(abc_url, rollbackbarindex=feat_id+1)
        with open(os.path.join(data_folder, "bodydetails"+str(feat_id)+".json"), "w") as f:
            json.dump(bodydetails, f, indent=4)

        if VERBOSE:
            ps.remove_all_structures()
            for tmp_feat in features["features"][:feat_id+1]:
                if tmp_feat["typeName"] == "BTMSketch":
                    sketch_name = tmp_feat["message"]["name"]
                    #print(sketch_name)
                    #print(tmp_feat)
                    for curve_id, curve_geom in enumerate(sketches_dict[tmp_feat["message"]["featureId"]]):
                        if len(curve_geom) == 1:
                            edges_array = np.array([[0, 0]])
                        else:
                            edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
                        ps.register_curve_network(sketch_name + "_" + str(curve_id), nodes=np.array(curve_geom),
                                                  edges=edges_array, color=(1, 0, 0))
        # vi) export intermediate shape
        edges = call.get_tesselated_edges(new_url)
        edges_dict = {}
        faces = call.get_tesselated_faces(new_url)
        faces_dict = {}
        feature_lines_file = os.path.join(data_folder, "feature_lines_"+str(feat_id)+".json")
        feature_faces_file = os.path.join(data_folder, "feature_faces_"+str(feat_id)+".json")
        if len(edges) == 0:
            if VERBOSE:
                ps.show()
            with open(feature_lines_file, "w") as f:
                json.dump(edges_dict, f, indent=4)
            with open(feature_faces_file, "w") as f:
                json.dump(faces_dict, f, indent=4)
            continue
        stl = call.export_stl(new_url, logging=False)
        stl_mesh_file = os.path.abspath(os.path.join(data_folder, "shape_"+str(feat_id)+".stl"))
        with open(stl_mesh_file, 'w') as f:  # Write STL to file
            f.write(stl.text)
        #ms.load_new_mesh(stl_mesh_file)
        with open(stl_mesh_file, "r") as fp:
            stl_dict = load_stl(fp)
            stl_mesh = Trimesh(vertices=stl_dict["vertices"],
                               faces=stl_dict["faces"],
                               face_normals=stl_dict["face_normals"])
        obj_mesh_file = os.path.join(data_folder, "shape_"+str(feat_id)+".obj")
        obj_str = export_obj(stl_mesh)
        with open(obj_mesh_file, "w") as fp:
            fp.write(obj_str)

        #ms.save_current_mesh(obj_mesh_file)

        edges_3d = []
        for edge in edges:
            for subedge in edge["edges"]:
                #edges_3d.append(np.array(subedge["vertices"]))
                edges_3d.append(subedge["vertices"])
                edges_dict[subedge["id"]] = subedge["vertices"]
                #print(subedge["id"])
        with open(feature_lines_file, "w") as f:
            json.dump(edges_dict, f, indent=4)
            #json.dump(edges_3d, f, indent=4)

        #faces_3d = []
        for part in faces:
            for face in part["faces"]:
                facets = []
                #print(face)
                #edges_3d.append(np.array(subedge["vertices"]))
                for facet in face["facets"]:
                    #facets.append(np.array(facet["vertices"]))
                    facets.append(facet["vertices"])
                    #print(facet)
                    #exit()
                    #edges_3d.append(subedge["vertices"])
                    #edges_dict[subedge["id"]] = subedge["vertices"]
                    #print(subedge["id"])
                faces_dict[face["id"]] = facets

        with open(feature_faces_file, "w") as f:
            json.dump(faces_dict, f, indent=4)
            #json.dump(edges_3d, f, indent=4)

        # DEBUG
        if VERBOSE:
            v, f = igl.read_triangle_mesh(obj_mesh_file)
            mesh_color = (139.0/255.0, 139.0/255.0, 230.0/255.0)
            ps_mesh = ps.register_surface_mesh("surface", vertices=v, faces=f, color=mesh_color)
            for curve_id, curve_geom in enumerate(edges_3d):
                edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
                ps.register_curve_network(str(curve_id), nodes=np.array(curve_geom),
                                          edges=edges_array, color=(0, 0, 0))
            ps.show()

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with open(os.path.join(data_folder, "parsed_features.json"), "w") as f:
        json.dump(parsed_features, f, indent=4)
    # vii) delete new document
    #call.delete_document(new_url)

if __name__ == "__main__":
    house_url = "https://cad.onshape.com/documents/adab1210c5a4bedaf968383b/w/1780773ba81784eff394e0fa/e/efbe2a7ee277f245a49509c7"
    abc_url = "https://cad.onshape.com/documents/290a9120f9f249a7a05cfe9c/w/f3d6fe4cfa4f4fd5a956c1f5/e/f83841055a93404a97c5ae79"
    abc_url = "https://cad.onshape.com/documents/92d1f080c60c0a4e3468fdd9/w/8cc36f617f0878b8ec3ce66b/e/cf697f79e1a98eafb51062b5"
    abc_url = "https://cad.onshape.com/documents/9b3d6a97e8de4aa193b81000/w/efbdb20ff72149beb24b2ae1/e/310d6b0aaf944863a63c9f20"
    abc_url = "https://cad.onshape.com/documents/e939ac566055e911385dcbf1/w/5b2596a854bc6f3a645a9c07/e/ef58e4dbaccaebd4f52cbabb"
    abc_url = "https://cad.onshape.com/documents/adab1210c5a4bedaf968383b/w/1780773ba81784eff394e0fa/e/efbe2a7ee277f245a49509c7"
    abc_url = "https://cad.onshape.com/documents/9b3d6a97e8de4aa193b81000/w/efbdb20ff72149beb24b2ae1/e/310d6b0aaf944863a63c9f20"
    abc_url = "https://cad.onshape.com/documents/31b4cd94eaeaee2bafddfce3/w/d3682bb24313b01aca4b6c7b/e/97fd87a958ea459e2ff40fa0"
    abc_id = 1
    #get_intermediate_shapes(abc_url, abc_id, VERBOSE=True)
    get_intermediate_shapes(abc_url, os.path.join("data", "1000"))

    exit()

    #    all_urls = {}
#    for subdir, dirs, files in os.walk("abc_dataset"):
#        for file in files:
#            # print os.path.join(subdir, file)
#            filepath = subdir + os.sep + file
#
#            if filepath.endswith(".yml"):
#                print(filepath)
#                with open(filepath, "r") as f:
#                    urls = yaml.load(f, Loader=yaml.FullLoader)
#                    all_urls = {**all_urls, **urls}
#                    break
#                print(type(urls))

    all_urls_file = os.path.join("abc_dataset", "all_urls.yml")
    #with open(all_urls_file, "w") as f:
    #    yaml.dump(all_urls, f)
    #print(all_urls)

    with open(all_urls_file, "r") as f:
        urls = yaml.load(f, Loader=yaml.FullLoader)
    for abc_id in urls.keys():
        print(abc_id, urls[abc_id])
        get_intermediate_shapes(urls[abc_id], abc_id)
