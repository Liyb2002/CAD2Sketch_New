import math
from trimesh import Trimesh
from trimesh.proximity import closest_point
from sklearn.neighbors import NearestNeighbors
import os
from get_intermediate_shapes import copy_workspace
import onshape_data._entity as entity
import matplotlib.pyplot as plt
import polyscope as ps
from shapely.geometry import MultiPoint
from skspatial.objects import Plane, Line
from trimesh.sample import sample_surface_even
import onshape.call as call
import trimesh
import utils
from random import choice, sample, uniform
from onshape_data.sketch import Sketch
from get_intermediate_shapes import get_intermediate_shapes
import json
import numpy as np

TEMPLATE_PATH = "/".join(__file__.split("/")[:-1])+'/onshape/feature_template_extrude.json'
TEMPLATE_PATH = os.path.join("onshape", "feature_template_extrude.json")

UP_TO_TEMPLATE_PATH = "/".join(__file__.split("/")[:-1])+'/onshape/feature_template_extrude_up_to.json'
UP_TO_TEMPLATE_PATH = os.path.join("onshape", "feature_template_extrude_up_to.json")

FILLET_TEMPLATE_PATH = "/".join(__file__.split("/")[:-1])+'/onshape/feature_template_fillet.json'
FILLET_TEMPLATE_PATH = os.path.join("onshape", "feature_template_fillet.json")

SKETCH_TEMPLATE_PATH = "/".join(__file__.split("/")[:-1])+'/onshape/feature_template.json'
SKETCH_TEMPLATE_PATH = os.path.join("onshape", "feature_template.json")

def get_fillet_dict(edge_ids, radius=1.0):
    with open(FILLET_TEMPLATE_PATH, 'r') as fh:
        template = json.load(fh)
    for edge_id in edge_ids:
        template["feature"]["message"]["parameters"][1]["message"]["queries"].append(
            {
                "type": 138,
                "typeName": "BTMIndividualQuery",
                "message": {
                    "geometryIds": [edge_id],
                    "hasUserCode": False
                }
            }
        )
    template["feature"]["message"]["parameters"][7]["message"]["expression"] = str(radius) + " cm"
    return template

def get_extrude_dict_up_to(face_feature_id, depth=1.0, body_type="REMOVE", next_face_id=None):
    with open(UP_TO_TEMPLATE_PATH, 'r') as fh:
        template = json.load(fh)
    #template["feature"]["message"]["parameters"][1]["message"]["value"] = body_type
    #template["feature"]["message"]["parameters"][1]["message"]["value"] = "REMOVE"
    #template["feature"]["message"]["parameters"][3]["message"]["value"] = "UP_TO_SURFACE"
    #template["feature"]["message"]["parameters"][6]["message"]["value"] = True
    template["feature"]["message"]["parameters"][5]["message"]["queries"][0]["message"]["featureId"] = str(face_feature_id)
    template["feature"]["message"]["parameters"][10]["message"]["queries"][0]["message"]["geometryIds"] = [str(next_face_id)]
    #template["feature"]["message"]["parameters"][10]["message"]["queries"] = [{"message": {"geometryIds": [next_face_id]}}]
    #template["feature"]["message"]["parameters"][4]["message"]["expression"] = str(depth)+"*cm"
    #print(template)
    #for param in template["feature"]["message"]["parameters"]:
    #    print(param)
    return template

def get_extrude_dict(face_feature_id, depth=1.0, body_type="NEW", next_face_id=None, is_face=False):
    with open(TEMPLATE_PATH, 'r') as fh:
        template = json.load(fh)
    template["feature"]["message"]["parameters"][1]["message"]["value"] = body_type
    if body_type == "REMOVE":
        template["feature"]["message"]["parameters"][3]["message"]["value"] = "UP_TO_NEXT"
        template["feature"]["message"]["parameters"][6]["message"]["value"] = True
    if body_type == "UP_TO_FACE":
        template["feature"]["message"]["parameters"][1]["message"]["value"] = "REMOVE"
        template["feature"]["message"]["parameters"][3]["message"]["value"] = "UP_TO_SURFACE"
        template["feature"]["message"]["parameters"][6]["message"]["value"] = True
        template["feature"]["message"]["parameters"][7]["message"]["queries"] = [{"message": {"geometryIds": [next_face_id]}}]
    template["feature"]["message"]["parameters"][2]["message"]["queries"][0]["message"]["featureId"] = str(face_feature_id)
    if is_face:
        template["feature"]["message"]["parameters"][2]["message"]["queries"][0] = {
            "type": 138,
            "typeName": "BTMIndividualQuery",
            "message": {
                "geometryIds": [str(face_feature_id)]
            }
        }

    template["feature"]["message"]["parameters"][4]["message"]["expression"] = str(depth)+"*cm"
    #print(template)
    #for param in template["feature"]["message"]["parameters"]:
    #    print(param)
    return template

def extrude_script(face_feature_id, depth):

    extrude_script = """
    {
      "feature" : {
        "type": 134,
        "typeName": "BTMFeature",
        "message": {
          "featureType": "extrude",
          "name": "Extrude 1",
          "parameters": [
            {
              "type": 145,
              "typeName": "BTMParameterEnum",
              "message": {
                "enumName": "ToolBodyType",
                "value": "SOLID",
                "parameterId": "bodyType"
              }
            },
            {
              "type": 145,
              "typeName": "BTMParameterEnum",
              "message": {
                "enumName": "NewBodyOperationType",
                "value": "NEW",
                "parameterId": "operationType"
              }
            },
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "queries": [
                  {
                    "type": 140,
                    "typeName": "BTMIndividualSketchRegionQuery",
                    "message": {
                      "featureId": """+face_feature_id+"""
                    }
                  }
                ],
                "parameterId": "entities"
              }
            },
            {
              "type": 145,
              "typeName": "BTMParameterEnum",
              "message": {
                "enumName": "BoundingType",
                "value": "BLIND",
                "parameterId": "endBound"
              }
            },
            {
              "type": 147,
              "typeName": "BTMParameterQuantity",
              "message": {
                "expression": \""""+str(depth)+"""*in",
                "parameterId": "depth"
              }
            },
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "parameterId": "surfaceEntities"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "parameterId": "oppositeDirection"
              }
            },
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "parameterId": "endBoundEntityFace"
              }
            },
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "parameterId": "endBoundEntityBody"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "parameterId": "hasDraft"
              }
            },
            {
              "type": 147,
              "typeName": "BTMParameterQuantity",
              "message": {
                "expression": "3.0*deg",
                "parameterId": "draftAngle"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "parameterId": "draftPullDirection"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "value": false,
                "parameterId": "hasSecondDirection"
              }
            },
            {
              "type": 145,
              "typeName": "BTMParameterEnum",
              "message": {
                "enumName": "BoundingType",
                "value": "BLIND",
                "parameterId": "secondDirectionBound"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "value": true,
                "parameterId": "secondDirectionOppositeDirection"
              }
            },
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "parameterId": "secondDirectionBoundEntityFace"
              }
            },
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "parameterId": "secondDirectionBoundEntityBody"
              }
            },
            {
              "type": 147,
              "typeName": "BTMParameterQuantity",
              "message": {
                "expression": "1.0*in",
                "parameterId": "secondDirectionDepth"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "parameterId": "hasSecondDirectionDraft"
              }
            },
            {
              "type": 147,
              "typeName": "BTMParameterQuantity",
              "message": {
                "expression": "3.0*deg",
                "parameterId": "secondDirectionDraftAngle"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "parameterId": "secondDirectionDraftPullDirection"
              }
            },
            {
              "type": 144,
              "typeName": "BTMParameterBoolean",
              "message": {
                "parameterId": "defaultScope"
              }
            },
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "parameterId": "booleanScope"
              }
            }
          ]
        }
      }
    }
    """
    return extrude_script

def sketch_script(sketch_plane):
    sketch_script = """
    {
      "feature" : {
        "type": 151,
        "typeName": "BTMSketch",
        "message": {
          "entities": [
            {
              "type": 4,
              "typeName": "BTMSketchCurve",
              "message": {
                "geometry": {
                  "type": 115,
                  "typeName": "BTCurveGeometryCircle",
                  "message": {
                    "radius": 0.025400000000000002,
                    "xDir": 1,
                    "yDir": 0,
                    "xCenter": 0.0,
                    "yCenter": 0.0
                  }
                }
              }
            },
            {
                "type": 155,
                "typeName": "BTMSketchCurveSegment",
                "message": {
                    "startParam": -0.04096709564328194,
                    "endParam": 0.04096709564328194,
                    "geometry": {
                        "type": 117,
                        "typeName": "BTCurveGeometryLine",
                        "message": {
                            "pntX": 0.0045100972056388855,
                            "pntY": -0.0320175401866436,
                            "dirX": 1.0,
                            "dirY": 0.0
                        }
                    }
                }
            }
          ],
          "constraints": [],
          "featureType": "newSketch",
          "name": "Sketch 1",
          "parameters": [
            {
              "type": 148,
              "typeName": "BTMParameterQueryList",
              "message": {
                "queries": [
                  {
                    "type": 138,
                    "typeName": "BTMIndividualQuery",
                    "message": {
                      "geometryIds": [
                        \""""+sketch_plane+"""\"
                      ]
                    }
                  }
                ],
                "parameterId": "sketchPlane"
              }
            }
          ]
        }
      }
    }
    """
    return sketch_script

def plane_coord_script():
    return """
    function(context is Context, queries)
    {
    return evPlane(context, {face: queries.id});
    }
    """

def extract_coord_system(msg):
    #print("extract_coord_system")
    #print(msg)
    coord_dict = {"normal": np.zeros(3),
                  "origin": np.zeros(3),
                  "x": np.zeros(3)}
    for i in range(3):
        key = msg["result"]["message"]["value"][i]["message"]["key"]["message"]["value"]
        #print(key)
        for j in range(3):
            coord_dict[key][j] = msg["result"]["message"]["value"][i]["message"]["value"]["message"]["value"][j]["message"]["value"]
    return coord_dict

def project_points_on_plane(pts, coord_dict):
    #print("pts", pts)
    x_vec = coord_dict["x"]
    y_vec = np.cross(coord_dict["normal"], x_vec)
    #print(x_vec, y_vec)
    #y_vec = np.cross(x_vec, coord_dict["normal"])
    plane_coords_0 = np.dot(x_vec, pts.T)
    plane_coords_1 = np.dot(y_vec, pts.T)
    pts_2d = np.array([[plane_coords_0[i], plane_coords_1[i]] for i in range(len(pts))])
    #print("pts_2d", pts_2d)
    return pts_2d

def generate_circular_sketch(origin, radius, nb_points, coord_dict, align_vec=None):

    origin = np.array(origin)
    angle = 360.0/nb_points
    curr_angle = 0.0
    sketch_points = []
    #print(coord_dict)
    if align_vec is not None:
        #print("align_vec", align_vec)
        angles = np.linspace(0, 360, num=360)
        #print(angles)
        lines = [np.array([[np.cos(np.deg2rad(a+angle)), np.sin(np.deg2rad(a+angle))], [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]])
                 for a in angles]
        lines_vec = [(l[1]-l[0])/np.linalg.norm(l[1]-l[0]) for l in lines]
        #print("lines_vec", lines_vec)
        #print(np.array(lines_vec).shape)
        min_angle = angles[np.argmax(np.abs(np.dot(align_vec, np.array(lines_vec).T)))]
        #print("max_alignment", np.max(np.abs(np.dot(align_vec, np.array(lines_vec).T))))
        curr_angle = min_angle
        #print("curr_angle", curr_angle)
        #exit()
    for i in range(nb_points):
        #dev_radius = (uniform(-1.0, 1.0)*radius/10+radius)
        dev_radius = radius
        #dev_angle = uniform(-1.0, 1.0)*10+curr_angle
        dev_angle = curr_angle
        pt = [dev_radius*np.cos(np.deg2rad(dev_angle)), dev_radius*np.sin(np.deg2rad(dev_angle))]
        sketch_points.append(pt)
        curr_angle += angle
    return np.array(sketch_points)

def add_new_sketch(client, new_url, features, start_model=True, data_folder=None):
    docid, wid, eid = call._parse_url(new_url)
    new_face_id = -1
    if start_model:
        new_face_id = "JDC"
        resp = client.eval_featurescript(docid, wid, eid, plane_coord_script(),
                                         [{"key": "id", "value": [new_face_id]}])
        coord_dict = extract_coord_system(resp)
        c0 = entity.Circle("c0", radius=0.01)
        s = Sketch(entities={"c0": c0})
        new_sketch = generate_circular_sketch(np.zeros(3), 0.01, choice([3, 4, 5, 6]), coord_dict)
        #new_sketch = generate_circular_sketch(np.zeros(3), 0.01, choice([4]), coord_dict)
        #print(coord_dict)
        #new_sketch = generate_circular_sketch(np.zeros(3), 0.01, choice([4]), coord_dict,
        #                                      align_vec=np.array([1, 0]))
    else:
        bd = call.get_bodydetails(new_url)
        face_ids = [bd["bodies"][0]["faces"][i]["id"] for i in range(len(bd["bodies"][0]["faces"]))
                    if bd["bodies"][0]["faces"][i]["surface"]["type"] == "plane"]
        plane_areas = {}
        for f_id, f in enumerate(bd["bodies"][0]["faces"]):
            if f["surface"]["type"] == "plane" and len(f["loops"]) == 1:
                #print(f)
                plane_areas[f_id] = f["area"]
        median_area = np.median(list(plane_areas.values()))
        median_area = np.mean(list(plane_areas.values()))
        pop_ids = []
        for f_id in plane_areas.keys():
            if plane_areas[f_id] < 3/4*median_area:
                pop_ids.append(f_id)
        for pop_id in pop_ids:
            plane_areas.pop(pop_id)
        new_f_id = choice(list(plane_areas.keys()))
        #new_f_id = list(plane_areas.keys())[np.argmax(list(plane_areas.values()))]
        new_face_id = bd["bodies"][0]["faces"][new_f_id]["id"]
        plane_origin = (np.array(bd["bodies"][0]["faces"][new_f_id]["box"]["minCorner"]) +
                        np.array(bd["bodies"][0]["faces"][new_f_id]["box"]["maxCorner"]))/2
        plane_normal = np.array(bd["bodies"][0]["faces"][new_f_id]["surface"]["normal"])
        #print("new_face_id", new_face_id)
        resp = client.eval_featurescript(docid, wid, eid, plane_coord_script(),
                                         [{"key": "id", "value": [new_face_id]}])
        coord_dict = extract_coord_system(resp)
        #print(coord_dict)

        new_face = client.get_partstudio_tessellatedface(docid, wid, eid, new_face_id)
        #print(new_face)
        new_face_vertices = np.array([facet["vertices"] for facet in new_face[0]["faces"][0]["facets"]])
        new_face_faces = [[i * 3, i * 3 + 1, i * 3 + 2] for i in range(len(new_face_vertices))]
        new_face_vertices = new_face_vertices.reshape(-1, 3)
        #ps.init()
        #ps.register_surface_mesh("face", vertices=new_face_vertices, faces=new_face_faces)
        mesh = trimesh.Trimesh(vertices=new_face_vertices, faces=new_face_faces)
        #unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        #print(unique_edges)
        projected_vertices = project_points_on_plane(mesh.vertices, coord_dict)
        bbox = MultiPoint(projected_vertices).convex_hull.bounds
        #print(bbox)
        bbox_diag = np.linalg.norm(np.array([bbox[0], bbox[1]]) - np.array([bbox[2], bbox[3]]))
        #bbox_diag = np.linalg.norm(np.array([bbox[0], bbox[1]]) - np.array([bbox[2], bbox[3]]))
        bbox_diag = np.minimum(np.abs(bbox[0] - bbox[2]), np.abs(bbox[1] - bbox[3]))
        #unique_vertices = np.array([[new_face_vertices[e[0]], new_face_vertices[e[1]]] for e in unique_edges]).reshape(-1, 3)
        samples, _ = sample_surface_even(mesh, count=5)
        new_sketch_origin = np.array(choice(samples))
        new_sketch_origin = plane_origin
        #subsamples = np.array(sample(list(samples), 5))
        #ps.register_point_cloud("samples", samples)
        #ps.register_point_cloud("subsamples", subsamples)
        #ps.register_curve_network("unique_edges", nodes=mesh.vertices, edges=np.array(unique_edges))
        #ps.show()

        #projected_unique_vertices = project_points_on_plane(new_face_vertices, coord_dict)
        #projected_samples = project_points_on_plane(subsamples, coord_dict)
        #cvx_samples = np.array(MultiPoint(projected_samples).convex_hull.exterior.coords)
        #plt.scatter(projected_samples[:, 0], projected_samples[:, 1])
        #plt.scatter(projected_unique_vertices[:, 0], projected_unique_vertices[:, 1])

        # align with parent face (if possible)
        # get longest straight edge
        edge_lengths = {}
        #print(bd["bodies"][0]["edges"])
        coedge_ids = [coedge["edgeId"] for coedge in bd["bodies"][0]["faces"][new_f_id]["loops"][0]["coedges"]]
        #for coedge in bd["bodies"][0]["faces"][new_f_id]["loops"][0]["coedges"]:
        for vec_id, edge in enumerate(bd["bodies"][0]["edges"]):
            if edge["id"] in coedge_ids and edge["curve"]["type"] == "line":
                    edge_lengths[vec_id] = edge["geometry"]["length"]
        #print("edge_lengths", edge_lengths, len(list(edge_lengths.keys())))

        projected_center = project_points_on_plane(np.array([new_sketch_origin]), coord_dict)[0]
        projected_origin = project_points_on_plane(np.array([coord_dict["origin"]]), coord_dict)[0]
        #print("new_sketch_origin", new_sketch_origin)
        #print("projected")
        #print(projected_center)
        #print(projected_origin)
        theta = 0.0
        line_vec = None
        if len(list(edge_lengths.keys())) > 0:
            # redefine coord_dict: x
            max_edge_id = list(edge_lengths.keys())[np.argmax(list(edge_lengths.values()))]
            endpoints = np.array([bd["bodies"][0]["edges"][max_edge_id]["geometry"]["endPoint"],
                                 bd["bodies"][0]["edges"][max_edge_id]["geometry"]["startPoint"]])
            projected_endpoints = project_points_on_plane(endpoints, coord_dict)
            line_vec = projected_endpoints[0] - projected_endpoints[1]
            line_vec /= np.linalg.norm(line_vec)
            #theta = -math.acos(np.dot(coord_dict["x"], line_vec))
            #c, s = np.cos(theta), np.sin(theta)
            #rot_mat = np.array(((c, -s), (s, c)))
            #new_sketch = np.matmul(rot_mat, new_sketch.T)
            #print(new_sketch)

            #coord_dict["x"] = line_vec
            #print("redefined")

        # look at intersection of extrusion with other planes
        planes = []
        planes_bounds = []
        for f_id, f in enumerate(bd["bodies"][0]["faces"]):
            if f_id == new_f_id or f["surface"]["type"] != "plane":
                continue
            planes.append(Plane(point=np.array(f["surface"]["origin"]),
                                normal=np.array(f["surface"]["normal"])))
            planes_bounds.append(np.array([f["box"]["minCorner"], f["box"]["maxCorner"]]))
        ext_line = Line(point=plane_origin, direction=plane_normal)
        min_bounds_diag = 1000.0
        # TODO: fix bbox
        for p_id, p in enumerate(planes):
            if not ext_line.direction.is_perpendicular(p.normal):
                inter_p = p.intersect_line(ext_line)
                if np.all(inter_p <= planes_bounds[p_id][1]) and np.all(inter_p >= planes_bounds[p_id][0]):
                    expanded_bbox = np.array([
                        [planes_bounds[p_id][0][0], planes_bounds[p_id][0][1], planes_bounds[p_id][0][2]],
                        [planes_bounds[p_id][0][0], planes_bounds[p_id][1][1], planes_bounds[p_id][0][2]],
                        [planes_bounds[p_id][0][0], planes_bounds[p_id][1][1], planes_bounds[p_id][1][2]],
                        [planes_bounds[p_id][0][0], planes_bounds[p_id][0][1], planes_bounds[p_id][1][2]],
                        [planes_bounds[p_id][1][0], planes_bounds[p_id][0][1], planes_bounds[p_id][0][2]],
                        [planes_bounds[p_id][1][0], planes_bounds[p_id][1][1], planes_bounds[p_id][0][2]],
                        [planes_bounds[p_id][1][0], planes_bounds[p_id][1][1], planes_bounds[p_id][1][2]],
                        [planes_bounds[p_id][1][0], planes_bounds[p_id][0][1], planes_bounds[p_id][1][2]],
                                             ])
                    #projected_vertices = project_points_on_plane(planes_bounds[p_id], coord_dict)
                    projected_vertices = project_points_on_plane(expanded_bbox, coord_dict)
                    #print("projected_vertices", projected_vertices)
                    #ps.init()
                    #ps.register_point_cloud("plane_bounds", expanded_bbox)
                    #ps.register_point_cloud("new_face", mesh.vertices)
                    #ps.show()
                    bbox = MultiPoint(projected_vertices).convex_hull.bounds
                    #print("bbox", bbox)
                    tmp_bbox_diag = np.minimum(np.abs(bbox[0] - bbox[2]), np.abs(bbox[1] - bbox[3]))
                    min_bounds_diag = np.minimum(min_bounds_diag, tmp_bbox_diag)
        #print(planes_bounds)
        #exit()
        min_bounds_diag = np.minimum(min_bounds_diag, bbox_diag)
        #print("min_bounds_diag", min_bounds_diag)
        new_sketch = generate_circular_sketch(np.zeros(3), min_bounds_diag/3, choice([3, 4, 5, 6]),
        #new_sketch = generate_circular_sketch(np.zeros(3), min_bounds_diag/3, choice([4]),
                                              coord_dict,
                                              align_vec=line_vec)
        #align_vec=line_vec)
        #new_sketch = [np.zeros(2), np.ones(2)]
        #print("offset", projected_center - projected_origin)
        new_sketch += projected_center # - projected_origin)
    cvx_samples = new_sketch
    #plt.show()
    #ps.init()

    new_lines_dict = {}
    for i in range(len(cvx_samples)):
        p0 = entity.Point("", x=cvx_samples[i][0], y=cvx_samples[i][1])
        p1 = entity.Point("", x=cvx_samples[(i + 1) % len(cvx_samples)][0],
                          y=cvx_samples[(i + 1) % len(cvx_samples)][1])
        #generate_line = choice([True, False], p=[0.7, 0.3])
        generate_line = np.random.choice([True, False], p=[0.7, 0.3])
        generate_line = True
        if i == 0:
            generate_line = True
        if generate_line:
            l0 = entity.Line.from_points(p0, p1)
        else:
            center = (cvx_samples[i]+cvx_samples[(i+1)%len(cvx_samples)])/2
            radius = np.linalg.norm(cvx_samples[i] - cvx_samples[(i+1)%len(cvx_samples)])/2
            line_vec = cvx_samples[(i+1)%len(cvx_samples)] - cvx_samples[i]
            line_vec /= np.linalg.norm(line_vec)
            ctrl_mid_point = center + radius/3*np.array([line_vec[-1], line_vec[0]])
            l0 = entity.Arc.from_info({"id": "a0",
                                       "radius": np.linalg.norm(cvx_samples[i] - cvx_samples[(i+1)%len(cvx_samples)])/2,
                                       "clockwise": False,
                                       "center": (cvx_samples[i]+cvx_samples[(i+1)%len(cvx_samples)])/2,
                                       "startPoint": cvx_samples[i],
                                       "endPoint": cvx_samples[(i+1)%len(cvx_samples)],
                                       })
            #print([cvx_samples[i], ctrl_mid_point, cvx_samples[(i+1)%len(cvx_samples)]])
            #ctrl_pts = [cvx_samples[i], ctrl_mid_point, cvx_samples[(i+1)%len(cvx_samples)]]
            #curve = geomdl.fitting.interpolate_curve(ctrl_pts, 2)
            #l0 = entity.Spline(entityId="s0", degree=curve.degree,
            #                   controlPoints=np.array(ctrl_pts), isConstruction=False, isPeriodic=False,
            #                   isRational=False, knots=np.array(curve.knotvector),
            #                   startParam=0.0, endParam=1.0)

        l0.entityId = "l" + str(i)
        new_lines_dict[str(i)] = l0
    s = Sketch(entities=new_lines_dict)

    sketch_dict = s.to_dict()
    with open(SKETCH_TEMPLATE_PATH, 'r') as fh:
        template = json.load(fh)
    for version_key in ['serializationVersion', 'sourceMicroversion', 'libraryVersion']:
        template[version_key] = features[version_key]
    template['feature']['message']['entities'] = sketch_dict['entities']
    template['feature']['message']['constraints'] = sketch_dict['constraints']
    template["feature"]["message"]["featureId"] = "sketch1"
    template["feature"]["message"]["parameters"][0]["message"]["queries"][0]["message"]["geometryIds"] = [new_face_id]
    #print(template)
    resp = client.add_feature(docid, wid, eid, payload=template)
    if data_folder is not None:
        feat_i = 0
        while os.path.exists(os.path.join(data_folder, "feature_"+str(feat_i)+".json")):
            feat_i += 1
        with open(os.path.join(data_folder, "feature_"+str(feat_i)+".json"), "w") as fp:
            json.dump(template, fp)
    #print(resp)
    sketch_feat_id = resp["feature"]["message"]["featureId"]
    return sketch_feat_id

def synthetize_new_object(sequence_length, data_folder=None):
    template_url = "https://cad.onshape.com/documents/acd08328e8b5aa0fcf4a0b3a/w/5d4586ae42cdd2a81b2880d1/e/bdb451d8aa05bdde64341237"
    new_url = copy_workspace(template_url)
    features = call.get_features(url=new_url, logging=False)
    #print(features)
    docid, wid, eid = call._parse_url(new_url)
    client = call._create_client(False)
    EXTRUDE_BODY_TYPES = ["ADD", "REMOVE", "INTERSECT"]
    EXTRUDE_BODY_TYPES = ["ADD"]
    EXTRUDE_BODY_TYPES = ["ADD", "REMOVE"]
    EXTRUDE_SIGN = [-1, +1]

    sketch_feat_id = add_new_sketch(client, new_url, features, start_model=True, data_folder=data_folder)
    extrude_dict = get_extrude_dict(sketch_feat_id, 1, body_type="NEW")
    client.add_feature(docid, wid, eid, payload=extrude_dict)
    if data_folder is not None:
        feat_i = 0
        while os.path.exists(os.path.join(data_folder, "feature_"+str(feat_i)+".json")):
            feat_i += 1
        with open(os.path.join(data_folder, "feature_"+str(feat_i)+".json"), "w") as fp:
            json.dump(extrude_dict, fp)

    for i in range(sequence_length-1):
        features = call.get_features(url=new_url, logging=False)
        # new Sketch
        sketch_feat_id = add_new_sketch(client, new_url, features, start_model=False, data_folder=data_folder)
        # new extrude
        sign = +1
        body_type = np.random.choice(EXTRUDE_BODY_TYPES, p=[0.7, 0.3])
        if body_type == "REMOVE" or body_type == "INTERSECT":
            sign = -1
        extrude_dict = get_extrude_dict(sketch_feat_id, depth=sign * uniform(0.0, 1.0),
                                                        body_type=body_type)
        #print(extrude_dict)
        resp = client.add_feature(docid, wid, eid, payload=extrude_dict)
        if data_folder is not None:
            feat_i = 0
            while os.path.exists(os.path.join(data_folder, "feature_"+str(feat_i)+".json")):
                feat_i += 1
            with open(os.path.join(data_folder, "feature_"+str(feat_i)+".json"), "w") as fp:
                json.dump(extrude_dict, fp)
    return new_url

def get_plane(client, feat_id, new_url, bd, orientation="VERTICAL"):

    docid, wid, eid = call._parse_url(new_url)
    if feat_id == 0:
        new_face_id = "JDC"
        if orientation == "HORIZONTAL":
            new_face_id = "JCC"
        resp = client.eval_featurescript(docid, wid, eid, plane_coord_script(),
                                         [{"key": "id", "value": [new_face_id]}])
        coord_dict = extract_coord_system(resp)
        coord_dict["bbox"] = np.array([[-0.01, -0.01, 0.0],
                                       [0.01, 0.01, 0.0]])
    else:
        plane_areas = {}
        for f_id, f in enumerate(bd["bodies"][0]["faces"]):
            if f["surface"]["type"] == "plane":
                f_normal = np.array(f["surface"]["normal"])
                if not f["orientation"]:
                    f_normal *= -1
            if f["surface"]["type"] == "plane" and len(f["loops"]) == 1 and \
                    np.dot(f_normal, np.array([0, 0, 1])) >= 0:
                plane_areas[f_id] = f["area"]
        median_area = np.median(list(plane_areas.values()))
        median_area = np.mean(list(plane_areas.values()))
        pop_ids = []
        if len(plane_areas.keys()) == 0:
            return {}
        for f_id in plane_areas.keys():
            if plane_areas[f_id] < 3/4*median_area:
                pop_ids.append(f_id)
        for pop_id in pop_ids:
            plane_areas.pop(pop_id)
        new_f_id = choice(list(plane_areas.keys()))
        new_face_id = bd["bodies"][0]["faces"][new_f_id]["id"]
        plane_origin = (np.array(bd["bodies"][0]["faces"][new_f_id]["box"]["minCorner"]) +
                        np.array(bd["bodies"][0]["faces"][new_f_id]["box"]["maxCorner"]))/2
        plane_normal = np.array(bd["bodies"][0]["faces"][new_f_id]["surface"]["normal"])
        #print("new_face_id", new_face_id)
        resp = client.eval_featurescript(docid, wid, eid, plane_coord_script(),
                                         [{"key": "id", "value": [new_face_id]}])
        coord_dict = extract_coord_system(resp)
        bbox = np.array([np.array(bd["bodies"][0]["faces"][new_f_id]["box"]["minCorner"]),
                         np.array(bd["bodies"][0]["faces"][new_f_id]["box"]["maxCorner"])])
        coord_dict["bbox"] = bbox
    coord_dict["plane_id"] = new_face_id
    return coord_dict

def get_sketch(plane_dict, sketch, feat_id=-1, face_mesh=[], body_type="ADD"):
    sketch_lines_dict = {}

    bbox = plane_dict["bbox"]
    #print("sketch_bbox", bbox)
    bbox_diag = np.linalg.norm(bbox[1] - bbox[0])/np.sqrt(2)
    bbox_diag_vec = bbox[1] - bbox[0]
    middle_bbox_center = (bbox[0]+bbox[1])/2
    plane_x = np.array([1, 0, 0])
    if np.isclose(1.0, np.abs(np.dot(plane_x, plane_dict["normal"]))):
        plane_x = np.array([0, 1, 0])
    plane_y = np.cross(plane_x, plane_dict["normal"])
    plane_y /= np.linalg.norm(plane_y)
    # randomly sample bbox_center
    # determine valid box centers. That is, centers, which squares are inside the current face
    all_box_centers = [np.array([bbox[0][0] + np.random.choice([i])*(bbox[1][0]-bbox[0][0]),
                                 bbox[0][1] + np.random.choice([j])*(bbox[1][1]-bbox[0][1]),
                                 bbox[0][2] + np.random.choice([k])*(bbox[1][2]-bbox[0][2])])
                       for i in [0.25, 0.5, 0.75] for j in [0.25, 0.5, 0.75] for k in [0.25, 0.5, 0.75]]
    valid_box_centers = []
    if feat_id > 0:
        for center_id, bbox_center in enumerate(all_box_centers):
            bbox_dists = [min(abs(bbox_center[0]-bbox[0][0]), abs(bbox_center[0]-bbox[1][0])),
                          min(abs(bbox_center[1]-bbox[0][1]), abs(bbox_center[1]-bbox[1][1])),
                          min(abs(bbox_center[2]-bbox[0][2]), abs(bbox_center[2]-bbox[1][2]))]
            min_dist = min(np.sort(bbox_dists)[1:])
            min_dist_ratio = 0.5
            if not np.isclose(np.linalg.norm(middle_bbox_center - bbox_center), 0.0):
                min_dist_ratio = 1.0
            square_points = np.array([bbox_center-min_dist_ratio*min_dist*plane_x-min_dist_ratio*min_dist*plane_y,
                                      bbox_center-min_dist_ratio*min_dist*plane_x+min_dist_ratio*min_dist*plane_y,
                                      bbox_center+min_dist_ratio*min_dist*plane_x+min_dist_ratio*min_dist*plane_y,
                                      bbox_center+min_dist_ratio*min_dist*plane_x-min_dist_ratio*min_dist*plane_y,
                                      bbox_center])

            _, dists, _ = closest_point(face_mesh, square_points)
            if np.all(np.isclose(dists, 0.0)):
                valid_box_centers.append(bbox_center)
        if len(valid_box_centers) == 0:
            return -1, -1
        bbox_center = valid_box_centers[np.random.choice(np.arange(len(valid_box_centers)))]
    else:
        bbox_center = middle_bbox_center
    if sketch == "CIRCLE" and body_type == "REMOVE":
        bbox_center = middle_bbox_center

    #bbox_center = np.array([bbox[0][0] + np.random.choice([0.25, 0.5, 0.75])*(bbox[1][0]-bbox[0][0]),
    #                        bbox[0][1] + np.random.choice([0.25, 0.5, 0.75])*(bbox[1][1]-bbox[0][1]),
    #                        bbox[0][2] + np.random.choice([0.25, 0.5, 0.75])*(bbox[1][2]-bbox[0][2])])
    new_bbox = np.array([bbox_center-0.5*bbox_diag_vec,
                         bbox_center+0.5*bbox_diag_vec])
    #bbox_dists = [np.abs(bbox[1][2] - bbox_center[2]),
    #              np.abs(bbox[1][0] - bbox_center[0]),
    #              np.abs(bbox[1][1] - bbox_center[1])]
    #min_dist = min(np.sort(bbox_dists)[1:])
    #print("bbox_center", bbox_center)
    bbox_dists = [min(abs(bbox_center[0]-bbox[0][0]), abs(bbox_center[0]-bbox[1][0])),
                  min(abs(bbox_center[1]-bbox[0][1]), abs(bbox_center[1]-bbox[1][1])),
                  min(abs(bbox_center[2]-bbox[0][2]), abs(bbox_center[2]-bbox[1][2]))]
    #print(bbox_center)
    #print(bbox_dists)
    #min_dist = np.min(bbox_dists)
    # since it's a plane, the minimum dist will be 0
    #print("bbox_dists")
    #print(bbox_dists)
    min_dist = min(np.sort(bbox_dists)[1:])
    #print("min_dist", min_dist)

    if sketch == "SQUARE":
        #center = np.array(plane_dict["origin"])
        center = np.array([0, 0])
        min_dist_ratio = 0.5
        if not np.isclose(np.linalg.norm(middle_bbox_center - bbox_center), 0.0):
            min_dist_ratio = 1.0
        square_points = np.array([bbox_center-min_dist_ratio*min_dist*plane_x-min_dist_ratio*min_dist*plane_y,
                                  bbox_center-min_dist_ratio*min_dist*plane_x+min_dist_ratio*min_dist*plane_y,
                                  bbox_center+min_dist_ratio*min_dist*plane_x+min_dist_ratio*min_dist*plane_y,
                                  bbox_center+min_dist_ratio*min_dist*plane_x-min_dist_ratio*min_dist*plane_y])
        projected_points = project_points_on_plane(square_points, plane_dict)
        sketch_lines = [entity.Line.from_points(
            entity.Point("", x=projected_points[i][0], y=projected_points[i][1]),
            entity.Point("", x=projected_points[(i+1)%4][0], y=projected_points[(i+1)%4][1]))
            for i in range(4)]

        for i in range(4):
            sketch_lines[i].entityId = "l" + str(i)
            sketch_lines_dict[str(i)] = sketch_lines[i]

    elif sketch == "CIRCLE":
        circle_center = bbox_center
        if body_type == "REMOVE":
            circle_center = middle_bbox_center
        projected_center = project_points_on_plane(np.array([circle_center]), plane_dict)[0]
        #print("min_dist", min_dist)
        c0 = entity.Circle("c0", xCenter=projected_center[0], yCenter=projected_center[1], radius=0.5*min_dist)
        sketch_lines_dict[str(0)] = c0

    elif sketch == "CHAMFER":
        # we cut out a triangle
        plane_y_sign = +1
        if np.isclose(np.dot(np.array([0, 0, -1]), plane_y), 1.0):
            plane_y_sign = -1
        up_dist = np.abs(bbox[1][2] - bbox_center[2])
        x_side_dist = np.abs(bbox[1][0] - bbox_center[0])
        y_side_dist = np.abs(bbox[1][1] - bbox_center[1])
        if np.isclose(x_side_dist, 0.0):
            side_dist = y_side_dist
        else:
            side_dist = x_side_dist
        triangle_points = np.array([bbox_center + plane_y_sign*up_dist*plane_y,
                                    bbox_center + plane_y_sign*up_dist*plane_y + side_dist*plane_x,
                                    bbox_center + side_dist*plane_x])
        projected_points = project_points_on_plane(triangle_points, plane_dict)
        sketch_lines = [entity.Line.from_points(
            entity.Point("", x=projected_points[i][0], y=projected_points[i][1]),
            entity.Point("", x=projected_points[(i+1)%3][0], y=projected_points[(i+1)%3][1]))
            for i in range(3)]
        for i in range(3):
            sketch_lines[i].entityId = "l" + str(i)
            sketch_lines_dict[str(i)] = sketch_lines[i]


    s = Sketch(entities=sketch_lines_dict)
    return s.to_dict(), 100.0*min_dist

def upload_op(op_type, op_dict, features, client, new_url, plane_id):

    if op_type in ["SQUARE", "CIRCLE", "CHAMFER"]:
        with open(SKETCH_TEMPLATE_PATH, 'r') as fh:
            template = json.load(fh)
    for version_key in ['serializationVersion', 'sourceMicroversion', 'libraryVersion']:
        template[version_key] = features[version_key]
    template['feature']['message']['entities'] = op_dict['entities']
    template['feature']['message']['constraints'] = op_dict['constraints']
    template["feature"]["message"]["featureId"] = "sketch1"
    template["feature"]["message"]["parameters"][0]["message"]["queries"][0]["message"]["geometryIds"] = [plane_id]
    #print(template)
    docid, wid, eid = call._parse_url(new_url)
    resp = client.add_feature(docid, wid, eid, payload=template)
    return resp["feature"]["message"]["featureId"]

def sample_symmetric_edges(bd, axis_id=0):
    edges = bd["bodies"][0]["edges"]
    if len(edges) == 1:
        return edges[0]["id"]
    # for now, fillet only straight lines
    faces_per_edges = {}
    for f in bd["bodies"][0]["faces"]:
        for loop in f["loops"]:
            for edge in loop["coedges"]:
                if not edge["edgeId"] in faces_per_edges.keys():
                    faces_per_edges[edge["edgeId"]] = [f["surface"]["type"]]
                else:
                    faces_per_edges[edge["edgeId"]].append(f["surface"]["type"])
    lines = {}
    for e_id, e in enumerate(edges):
        if e["curve"]["type"] == "line" and \
                len(faces_per_edges[e["id"]]) == 2 and \
                np.sum([np.array(faces_per_edges[e["id"]]) == "plane"]) == 2:
            lines[e["id"]] = np.array([e["geometry"]["startPoint"], e["geometry"]["endPoint"]])
    if len(list(lines.keys())) == 0:
        return []
    sym_correspondences = []
    for l_id in lines.keys():
        l = lines[l_id]
        vec_l = np.array(l[1]) - np.array(l[0])
        vec_l /= np.linalg.norm(vec_l)
        l[0][axis_id] *= -1
        l[1][axis_id] *= -1
        for other_l_id in lines.keys():
            if l_id == other_l_id:
                continue
            other_l = lines[other_l_id]
            other_vec_l = np.array(other_l[1]) - np.array(other_l[0])
            other_vec_l /= np.linalg.norm(other_vec_l)
            #print(l, other_l)
            if Line(l[0], vec_l).is_coplanar(Line(other_l[0], other_vec_l)) and \
                np.isclose(np.abs(np.dot(vec_l, other_vec_l)), 1.0):
                sym_correspondences.append([l_id, other_l_id])
                break
            #if (np.isclose(np.linalg.norm(other_l[0] - l[0]), 0.0, atol=1e-5) and np.isclose(np.linalg.norm(other_l[1] - l[1]), 0.0, atol=1e-5)) or \
            #    (np.isclose(np.linalg.norm(other_l[1] - l[0]), 0.0, atol=1e-5) and np.isclose(np.linalg.norm(other_l[0] - l[1]), 0.0, atol=1e-5)):
            #    sym_correspondences.append([l_id, other_l_id])
            #    break
    #print("len(sym_correspondences)")
    #print(len(sym_correspondences))
    if len(sym_correspondences) == 0:
        return [np.random.choice(list(lines.keys()))]
    #chosen_correspondences = np.random.choice(np.arange(len(sym_correspondences)),
    #                                          np.random.choice(np.arange(1, min(3, len(sym_correspondences)+1))))
    chosen_correspondences = np.random.choice(np.arange(len(sym_correspondences)), 1)
    final_correspondences = []
    for corr in chosen_correspondences:
        final_correspondences.append(sym_correspondences[corr][0])
        final_correspondences.append(sym_correspondences[corr][1])
    return final_correspondences


def get_newly_created_faces_bbox(prev_bd, bd):
    old_faces_ids = [f["id"] for f in prev_bd["bodies"][0]["faces"]]
    points = np.array([np.array([f["box"]["minCorner"], f["box"]["maxCorner"]])
                       for f in bd["bodies"][0]["faces"]
                       if not f["id"] in old_faces_ids and len(f["loops"]) == 1]).reshape(-1, 3)
    max = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
    min = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
    bbox = np.array([min, max])
    return bbox

def get_newly_created_faces_plane(prev_bd, bd, client, url):
    old_planes = []
    old_faces_ids = []
    if len(prev_bd["bodies"]) > 0:
        old_faces_ids = [f["id"] for f in prev_bd["bodies"][0]["faces"]]
        for f in prev_bd["bodies"][0]["faces"]:
            if f["surface"]["type"] == "plane":
                old_planes.append(Plane(f["surface"]["origin"], f["surface"]["normal"]))
    new_faces_ids = [f["id"]
                     for f in bd["bodies"][0]["faces"]
                     if not f["id"] in old_faces_ids and len(f["loops"]) == 1
                     and f["surface"]["type"] == "plane"
                     and np.isclose(np.abs(np.dot(np.array([0, 0, 1]), np.array(f["surface"]["normal"]))), 0.0)
                     and not np.any([Plane(f["surface"]["origin"], f["surface"]["normal"]).is_close(old_p) for old_p in old_planes])]
    if len(new_faces_ids) == 0:
        return None
    new_face_id = np.random.choice(new_faces_ids)
    new_face_normal = [np.array(f["surface"]["normal"]) for f in bd["bodies"][0]["faces"]
                       if f["id"] == new_face_id]
    opposite_face_id = -1
    for f in bd["bodies"][0]["faces"]:
        if f["id"] != new_face_id and not f["id"] in old_faces_ids and f["surface"]["type"] == "plane" and \
                np.isclose(np.abs(np.dot(new_face_normal, np.array(f["surface"]["normal"]))), 1.0):
            opposite_face_id = f["id"]

    docid, wid, eid = call._parse_url(url)
    resp = client.eval_featurescript(docid, wid, eid, plane_coord_script(),
                                     [{"key": "id", "value": [new_face_id]}])
    coord_dict = extract_coord_system(resp)
    for f in bd["bodies"][0]["faces"]:
        if f["id"] == new_face_id:
            bbox = np.array([np.array(f["box"]["minCorner"]),
                             np.array(f["box"]["maxCorner"])])
            coord_dict["bbox"] = bbox
    coord_dict["plane_id"] = new_face_id
    return coord_dict, opposite_face_id

def get_max_fillet_radius(edges_3d, fillet_edge_ids, bd):
    min_dist = 100.0
    for fillet_edge_id in fillet_edge_ids:
        # go through each neighbouring face of current edge
        for f in bd["bodies"][0]["faces"]:
            edge_in_face = False
            for loop in f["loops"]:
                for coedge in loop["coedges"]:
                    if fillet_edge_id == coedge["edgeId"]:
                        edge_in_face = True
                        break
                if edge_in_face:
                    break
            if not edge_in_face:
                continue

            # remove all neighbouring edges
            non_neighbouring_edges = []
            for loop in f["loops"]:
                for vec_id, coedge in enumerate(loop["coedges"]):
                    if not (loop["coedges"][(vec_id-1)%len(loop["coedges"])]["edgeId"] == fillet_edge_id or
                            loop["coedges"][(vec_id+1)%len(loop["coedges"])]["edgeId"] == fillet_edge_id or
                            coedge["edgeId"] == fillet_edge_id):
                        non_neighbouring_edges.append(coedge["edgeId"])

            current_edge_3d = np.array(edges_3d[fillet_edge_id])
            if len(current_edge_3d) == 2:
                current_edge_3d = np.array(edges_3d[fillet_edge_id])
                current_vec = current_edge_3d[1] - current_edge_3d[0]
                current_edge_3d = current_edge_3d[0] + np.linspace(0.0, 1.0, 20).reshape(-1, 20).T*current_vec.reshape(-1, 3)
            # closest distance to other edge
            for other_edge_id in non_neighbouring_edges:
                other_edge_3d = np.array(edges_3d[other_edge_id])

                if len(other_edge_3d) == 2 and len(edges_3d) == 2:
                    other_vec = other_edge_3d[1] - other_edge_3d[0]
                    other_vec /= np.linalg.norm(other_vec)
                    current_vec /= np.linalg.norm(current_vec)
                    edge_dist = Line(current_edge_3d[0], current_vec).distance_line(Line(other_edge_3d[0], other_vec))
                    # half the distance if the closest edge is also a fillet edge
                    if other_edge_id in fillet_edge_ids:
                        edge_dist /= 2.0
                    min_dist = min(edge_dist, min_dist)
                    continue
                if len(other_edge_3d) == 2:
                    other_edge_3d = np.array(edges_3d[other_edge_id])
                    other_vec = other_edge_3d[1] - other_edge_3d[0]
                    other_edge_3d = other_edge_3d[0] + np.linspace(0.0, 1.0, 20).reshape(-1, 20).T*other_vec.reshape(-1, 3)

                x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric="l2").fit(current_edge_3d)
                min_y_to_x = x_nn.kneighbors(other_edge_3d)[0]
                edge_dist = np.min(min_y_to_x)

                # half the distance if the closest edge is also a fillet edge
                if other_edge_id in fillet_edge_ids:
                    edge_dist /= 2.0
                min_dist = min(edge_dist, min_dist)
    return 100.0*min_dist

def plane_inside_shape(plane_dict, bd):
    bbox = plane_dict["bbox"]
    middle_bbox_center = (bbox[0]+bbox[1])/2

    points = []
    for f in bd["bodies"][0]["faces"]:
        points.append(f["box"]["minCorner"])
        points.append(f["box"]["maxCorner"])
    points = np.array(points)
    bbox_max = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
    bbox_min = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
    bbox_mid = (bbox_min + bbox_max)/2.0
    bbox_min = bbox_min - (bbox_min-bbox_mid)*0.05
    bbox_max = bbox_max - (bbox_max-bbox_mid)*0.05
    if np.any(middle_bbox_center < bbox_min) or np.any(middle_bbox_center > bbox_max):
        return False
    return True

def synthetize_new_object_opensketch(sequence_length, sequence=[], data_folder=None):
    #new_url = "https://cad.onshape.com/documents/1a540b4735e9f5aabc518593/w/b82673996812d5d775d4b6a6/e/4b924d7aad8a05ae109e3827"
    #new_url = "https://cad.onshape.com/documents/6c509c865fb2e7701d927f5d/w/9545fc742a4ce0945398fabb/e/0424ce2a669654602386d441"
    #features = call.get_features(url=new_url, logging=False)
    #with open("up_to_face_features.json", "w") as fp:
    #    json.dump(features, fp, indent=4)
    #exit()
    template_url = "https://cad.onshape.com/documents/acd08328e8b5aa0fcf4a0b3a/w/5d4586ae42cdd2a81b2880d1/e/bdb451d8aa05bdde64341237"
    new_url = copy_workspace(template_url, new_name=str(data_folder))
    print(new_url)
    features = call.get_features(url=new_url, logging=False)
    #print(features)
    docid, wid, eid = call._parse_url(new_url)
    client = call._create_client(False)
    EXTRUDE_BODY_TYPES = ["ADD", "REMOVE", "INTERSECT"]
    EXTRUDE_BODY_TYPES = ["ADD"]
    EXTRUDE_BODY_TYPES = ["ADD", "REMOVE"]
    EXTRUDE_SIGN = [-1, +1]

    OP_TYPE = ["EXTRUDE", "FILLET", "CHAMFER", "REVOLVE"]
    SKETCH_TYPE = ["RECTANGLE", "SQUARE"]#, "CIRCLE"]
    SKETCH_TYPE = ["SQUARE", "CIRCLE"]

    sketch = "SQUARE"
    prev_op = "None"

    # randomize radius size
    fillet_radius_sizes = [0.25, 0.5, 0.75, 1.0]
    extrude_depth_sizes = [1.0, 1.5, 2.0]

    max_iter = 20
    iter = 0
    feat_id = 0
    already_square = False
    while feat_id < sequence_length:
    #for feat_id in range(sequence_length):
        possible_ops = []
        if iter > max_iter:
            break
        iter += 1

        if feat_id == 0:
            possible_ops.append("EXTRUDE")

        else:
            possible_ops.append("EXTRUDE")
            possible_ops.append("REMOVE")
            #possible_ops.append("DUPLICATE")
        if prev_op == "EXTRUDE":
            possible_ops.append("CHAMFER")


        op = np.random.choice(possible_ops)
        if feat_id >= sequence_length - 1:
            op = np.random.choice(["EXTRUDE", "FILLET"],
                                  p=[0.2, 0.8])
        sketch = np.random.choice(SKETCH_TYPE, p=[0.6, 0.4])

        extrude_depth_size = np.random.choice(extrude_depth_sizes, 1)[0]
        fillet_radius_size = np.random.choice(fillet_radius_sizes, 1)[0]

        if len(sequence) > 0:
            op = sequence[feat_id]["op"]
            sketch = sequence[feat_id]["sketch"]
            extrude_depth_size = sequence[feat_id]["extrude_depth_size"]
            fillet_radius_size = sequence[feat_id]["fillet_radius_size"]

        #sketch = np.random.choice(SKETCH_TYPE, p=[1.0])
        if sketch == "SQUARE":
            already_square = True
        if feat_id > 0 and sketch == "CIRCLE" and not already_square:
            sketch = "SQUARE"
        #if feat_id == 1:
        #    op = "DUPLICATE"
        #sketch = "SQUARE"
        print("feat_id", feat_id, "feature", op)

        bd = call.get_bodydetails(new_url)
        faces = call.get_tesselated_faces(new_url)
        faces_dict = {}
        for part in faces:
            for face in part["faces"]:
                facets = []
                vertices = []
                faces_ids = []
                for facet in face["facets"]:
                    facets.append(facet["vertices"])
                    faces_ids.append([len(vertices), len(vertices)+1, len(vertices)+2])
                    vertices.append(facet["vertices"][0])
                    vertices.append(facet["vertices"][1])
                    vertices.append(facet["vertices"][2])
                #faces_dict[face["id"]] = facets
                faces_dict[face["id"]] = Trimesh(vertices, faces_ids)

        features = call.get_features(url=new_url, logging=False)

        if op == "EXTRUDE":
            if feat_id == 0:
                body_type = "NEW"
            else:
                body_type = "ADD"
            plane_dict = get_plane(client, feat_id, new_url, bd, orientation=np.random.choice(["VERTICAL", "HORIZONTAL"]))
            if len(plane_dict.keys()) == 0:
                continue
            face_mesh = []
            if feat_id > 0 and plane_inside_shape(plane_dict, bd):
                continue
            if feat_id > 0:
                face_mesh = faces_dict[plane_dict["plane_id"]]
            sketch_dict, sketch_radius = get_sketch(plane_dict, sketch, feat_id=feat_id, face_mesh=face_mesh)
            if sketch_radius < 0.0:
                continue
            sketch_feat_id = upload_op(sketch, sketch_dict, features, client, new_url, plane_dict["plane_id"])
            last_extrude_depth = extrude_depth_size*sketch_radius
            extrude_dict = get_extrude_dict(sketch_feat_id, depth=last_extrude_depth, body_type=body_type)
            client.add_feature(docid, wid, eid, payload=extrude_dict)

        elif op == "REMOVE":
            body_type = "REMOVE"
            plane_dict = get_plane(client, feat_id, new_url, bd)
            if len(plane_dict.keys()) == 0:
                continue
            if plane_inside_shape(plane_dict, bd):
                continue
            if feat_id > 0:
                face_mesh = faces_dict[plane_dict["plane_id"]]
                #print(face_mesh)
            sketch_dict, sketch_radius = get_sketch(plane_dict, sketch, feat_id=feat_id, face_mesh=face_mesh, body_type=body_type)
            if sketch_radius < 0.0:
                continue
            sketch_feat_id = upload_op(sketch, sketch_dict, features, client, new_url, plane_dict["plane_id"])
            last_extrude_depth = extrude_depth_size*sketch_radius
            extrude_dict = get_extrude_dict(sketch_feat_id, depth=last_extrude_depth, body_type=body_type)
            client.add_feature(docid, wid, eid, payload=extrude_dict)

        elif op == "DUPLICATE":
            body_type = "ADD"
            plane_dict = get_plane(client, feat_id, new_url, bd)
            if len(plane_dict.keys()) == 0:
                continue
            if plane_inside_shape(plane_dict, bd):
                continue
            extrude_dict = get_extrude_dict(plane_dict["plane_id"], depth=last_extrude_depth, body_type=body_type, is_face=True)
            #print(extrude_dict)
            client.add_feature(docid, wid, eid, payload=extrude_dict)

        elif op == "FILLET":
            fillet_edge_ids = sample_symmetric_edges(bd, axis_id=np.random.choice([0, 1]))# + sample_symmetric_edges(bd, axis_id=1)
            if len(fillet_edge_ids) == 0:
                continue
            # get max_fillet radius
            edges_3d = {}
            edges = call.get_tesselated_edges(new_url)
            for edge in edges:
                for subedge in edge["edges"]:
                    edges_3d[subedge["id"]] = subedge["vertices"]
            max_radius = get_max_fillet_radius(edges_3d, fillet_edge_ids, bd)
            fillet_dict = get_fillet_dict(fillet_edge_ids, radius=fillet_radius_size*max_radius)
            try:
                client.add_feature(docid, wid, eid, payload=fillet_dict)
            except:
                continue

        elif op == "CHAMFER":
            sketch = "CHAMFER"
            if prev_sketch == "SQUARE":
                plane_dict, opposite_face_id = get_newly_created_faces_plane(prev_bd, bd, client, new_url)
                if plane_dict == None:
                    continue
                sketch_dict, sketch_radius = get_sketch(plane_dict, sketch)
                if sketch_radius < 0.0:
                    continue
                sketch_feat_id = upload_op(sketch, sketch_dict, features, client, new_url, plane_dict["plane_id"])
                extrude_dict = get_extrude_dict_up_to(sketch_feat_id, depth=1, body_type="UP_TO_FACE", next_face_id=opposite_face_id)
                client.add_feature(docid, wid, eid, payload=extrude_dict)
            # if cylinder, create new offset plane
            elif prev_sketch == "CIRCLE":
                continue
                # get faces created by last extrusion
                bbox = get_newly_created_faces_bbox(prev_bd, bd)
                print("TODO")
            # create chamfer sketch
        prev_op = op
        prev_bd = bd
        prev_sketch = sketch
        feat_id += 1
    return new_url
    #exit()

def get_object(seq_template):

    print(seq_template)
    obj_id = 193
    #main_data_folder = os.path.join("data")
    main_data_folder = os.path.join("/data2/fhahnlei/synthetic_concept_sketches/data")
    while os.path.exists(os.path.join(main_data_folder, str(obj_id))):
        obj_id += 1
    data_folder = os.path.join(main_data_folder, str(obj_id))
    print("data_folder", data_folder)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    url = synthetize_new_object_opensketch(4, seq_template, data_folder=obj_id)
    get_intermediate_shapes(url, data_folder)
    #exit()

if __name__ == "__main__":

    #seq_template = {'op': 'EXTRUDE', 'sketch': 'SQUARE', 'extrude_depth_size': 1.0, 'fillet_radius_size': 0.0}, {'op': 'CHAMFER', 'sketch': 'SQUARE', 'extrude_depth_size': 1.0, 'fillet_radius_size': 0.0}, {'op': 'EXTRUDE', 'sketch': 'SQUARE', 'extrude_depth_size': 2.0, 'fillet_radius_size': 0.5}, {'op': 'FILLET', 'sketch': '', 'extrude_depth_size': 0.0, 'fillet_radius_size': 0.75}
    #get_object(seq_template)
    #exit()

    seq_template = [{"op": "EXTRUDE",
                     "sketch": "SQUARE",
                     "extrude_depth_size": 1.0,
                     "fillet_radius_size": 0.0},
                    {"op": "CHAMFER",
                     "sketch": "SQUARE",
                     "extrude_depth_size": 1.0,
                     "fillet_radius_size": 0.0},
                    {"op": "EXTRUDE",
                     "sketch": "CIRCLE",
                     "extrude_depth_size": 1.0,
                     "fillet_radius_size": 0.5},
                    {"op": "FILLET",
                     "sketch": "",
                     "extrude_depth_size": 0.0,
                     "fillet_radius_size": 0.5}]
    #fillet_radius_sizes = [0.25, 0.5, 0.75, 1.0]
    fillet_radius_sizes = [0.5, 0.6, 0.75, 1.0]
    extrude_depth_sizes = [1.0, 1.5, 2.0]
    counter = 192
    # first step is always a box
    for ext_depth_size in extrude_depth_sizes:
        seq_template[0]["extrude_depth_size"] = ext_depth_size
        # second step
        for op in ["CHAMFER", "EXTRUDE", "REMOVE"]:
            seq_template[1]["op"] = op
            if op in ["EXTRUDE", "REMOVE"]:
                for sketch in ["SQUARE", "CIRCLE"]:
                    for ext_depth_size in extrude_depth_sizes:
                        seq_template[1]["sketch"] = sketch
                        seq_template[1]["extrude_depth_size"] = ext_depth_size
                        # third step
                        for op in ["CHAMFER", "EXTRUDE", "REMOVE"]:
                            seq_template[2]["op"] = op
                            if op in ["EXTRUDE", "REMOVE"]:
                                for sketch in ["SQUARE", "CIRCLE"]:
                                    for ext_depth_size in extrude_depth_sizes:
                                        seq_template[2]["sketch"] = sketch
                                        seq_template[2]["extrude_depth_size"] = ext_depth_size
                                        # fourth step
                                        for fillet_radius_size in fillet_radius_sizes:
                                            seq_template[3]["op"] = "FILLET"
                                            seq_template[3]["fillet_radius_size"] = fillet_radius_size
                                            #if counter > 885:
                                            get_object(seq_template)
                                            counter += 1

            # third step
            for op in ["CHAMFER", "EXTRUDE", "REMOVE"]:
                seq_template[2]["op"] = op
                if op in ["EXTRUDE", "REMOVE"]:
                    for sketch in ["SQUARE", "CIRCLE"]:
                        for ext_depth_size in extrude_depth_sizes:
                            seq_template[2]["sketch"] = sketch
                            seq_template[2]["extrude_depth_size"] = ext_depth_size
                            # fourth step
                            for fillet_radius_size in fillet_radius_sizes:
                                seq_template[3]["op"] = "FILLET"
                                seq_template[3]["fillet_radius_size"] = fillet_radius_size
                                #if counter > 885:
                                get_object(seq_template)
                                counter += 1

                # fourth step
                for fillet_radius_size in fillet_radius_sizes:
                    seq_template[3]["op"] = "FILLET"
                    seq_template[3]["fillet_radius_size"] = fillet_radius_size
                    #if counter > 885:
                    get_object(seq_template)
                    counter += 1
    print("counter", counter)
    exit()

    for i in range(100):
        obj_id = 0
        while os.path.exists(os.path.join("data", str(obj_id))):
            obj_id += 1
        data_folder = os.path.join("data", str(obj_id))
        print("data_folder", data_folder)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        #url = synthetize_new_object_opensketch(4)
        url = synthetize_new_object_opensketch(
            4, [{"op": "EXTRUDE",
                 "sketch": "SQUARE",
                 "extrude_depth_size": 1.0,
                 "fillet_radius_size": 0.0},
                {"op": "CHAMFER",
                 "sketch": "SQUARE",
                 "extrude_depth_size": 1.0,
                 "fillet_radius_size": 0.0},
                {"op": "EXTRUDE",
                 "sketch": "CIRCLE",
                 "extrude_depth_size": 1.0,
                 "fillet_radius_size": 0.5},
                {"op": "FILLET",
                 "sketch": "",
                 "extrude_depth_size": 0.0,
                 "fillet_radius_size": 0.5}])
        get_intermediate_shapes(url, data_folder)
        exit()
