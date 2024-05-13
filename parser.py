import pprint
import os
import copy
import numpy as np
from collections import OrderedDict
from utils import xyz_list2dict, angle_from_vector_to_x
import matplotlib.pyplot as plt

# OnShape naming to Fusion360 naming format
EXTENT_TYPE_MAP = {'BLIND': 'OneSideFeatureExtentType', 'SYMMETRIC': 'SymmetricFeatureExtentType'}
OPERATION_MAP = {'NEW': 'NewBodyFeatureOperation', 'ADD': 'JoinFeatureOperation',
                 'REMOVE': 'CutFeatureOperation', 'INTERSECT': 'IntersectFeatureOperation'}
CONSTRAINT_TYPES = {"COINCIDENT", "HORIZONTAL", "VERTICAL", "MIDPOINT"}


class FeatureListParser(object):
    """A parser for OnShape feature list (construction sequence)"""
    def __init__(self, client, did, wid, eid, sketch_info, data_id=None):
        self.c = client

        self.did = did
        self.wid = wid
        self.eid = eid
        self.data_id = data_id
        self.sketch_info = sketch_info

        self.feature_list = self.c.get_features(did, wid, eid).json()

        self.profile2sketch = {}

    @staticmethod
    def parse_feature_param(feat_param_data):
        param_dict = {}
        for i, param_item in enumerate(feat_param_data):
            param_msg = param_item['message']
            param_id = param_msg['parameterId']
            if 'queries' in param_msg:
                param_value = []
                for i in range(len(param_msg['queries'])):
                    param_value.extend(param_msg['queries'][i]['message']['geometryIds']) # FIXME: could be error-prone
            elif 'expression' in param_msg:
                param_value = param_msg['expression']
            elif 'value' in param_msg:
                param_value = param_msg['value']
            elif 'items' in param_msg:
                param_value = param_msg['items']
            else:
                raise NotImplementedError('param_msg:\n{}'.format(param_msg))

            param_dict.update({param_id: param_value})
        return param_dict

    def _parse_sketch(self, feature_data):
        #print("parse_sketch")
        sket_parser = SketchParser(self.c, feature_data, self.did, self.wid, self.eid, self.sketch_info)
        save_dict = sket_parser.parse_to_fusion360_format()
        return save_dict
        #return sket_parser

    def _expr2meter(self, expr):
        return self.c.expr2meter(self.did, self.wid, self.eid, expr)

    def _locateSketchProfile(self, geo_ids):
        return [{"profile": k, "sketch": self.profile2sketch[k]} for k in geo_ids]

    def _parse_extrude_simple(self, feature_data):
        param_dict = self.parse_feature_param(feature_data['parameters'])
        extent_one = self._expr2meter(param_dict['depth'])
        extent_two = 0.0
        if ("SYMMETRIC" in param_dict["endBound"]) or ("symmetric" in param_dict and param_dict['symmetric']):
            extent_one = extent_one / 2
            extent_two = extent_one
        elif "hasSecondDirection" in param_dict and param_dict["hasSecondDirection"]:
            extent_two = self._expr2meter(param_dict["secondDirectionDepth"])
            if param_dict["secondDirectionOppositeDirection"]:
                extent_two = -extent_two
        opposite_direction = False
        if 'oppositeDirection' in param_dict and param_dict['oppositeDirection'] is True:
            extent_one = -extent_one
            opposite_direction = True

        #if 'hasSecondDirection' in param_dict and param_dict['hasSecondDirection'] is True:
        #if param_dict['secondDirectionBound'] != 'BLIND':
        #    raise NotImplementedError("secondDirectionBound type not supported: {}".format(param_dict['endBound']))

        if 'secondDirectionOppositeDirection' in param_dict \
                and str(param_dict['secondDirectionOppositeDirection']) == 'true':
            extent_two = self._expr2meter(param_dict['secondDirectionDepth'])
            extent_two = -extent_two

        save_dict = {"name": feature_data["name"],
                     "type": "extrude",
                     "extrude_type": param_dict["endBound"],
                     "entities": param_dict["entities"],
                     "extent_one": extent_one,
                     "extent_two": extent_two,
                     "use_depth": param_dict["endBound"] in ["BLIND", "SYMMETRIC"],
                     "opposite_direction": opposite_direction
                     }
        return save_dict

    def _parse_fillet(self, feature_data):
        param_dict = self.parse_feature_param(feature_data['parameters'])
        save_dict = {"name": feature_data["name"],
                     "type": "fillet",
                     #"fillet_type": param_dict["filletType"],
                     "entities": param_dict["entities"],
                     #"cross_section": param_dict["crossSection"],
                     "radius": self._expr2meter(param_dict["radius"]),
                     }
        return save_dict

    def _parse_extrude(self, feature_data):
        param_dict = self.parse_feature_param(feature_data['parameters'])
        if 'hasOffset' in param_dict and param_dict['hasOffset'] is True:
            raise NotImplementedError("extrude with offset not supported: {}".format(param_dict['hasOffset']))

        entities = param_dict['entities'] # geometryIds for target face
        #profiles = self._locateSketchProfile(entities)

        extent_one = self._expr2meter(param_dict['depth'])
        if param_dict['endBound'] == 'SYMMETRIC':
            extent_one = extent_one / 2
        if 'oppositeDirection' in param_dict and param_dict['oppositeDirection'] is True:
            extent_one = -extent_one
        extent_two = 0.0
        if param_dict['endBound'] not in ['BLIND', 'SYMMETRIC']:
            raise NotImplementedError("endBound type not supported: {}".format(param_dict['endBound']))
        elif 'hasSecondDirection' in param_dict and param_dict['hasSecondDirection'] is True:
        #if 'hasSecondDirection' in param_dict and param_dict['hasSecondDirection'] is True:
            if param_dict['secondDirectionBound'] != 'BLIND':
                raise NotImplementedError("secondDirectionBound type not supported: {}".format(param_dict['endBound']))
            extent_type = 'TwoSidesFeatureExtentType'
            extent_two = self._expr2meter(param_dict['secondDirectionDepth'])
            if 'secondDirectionOppositeDirection' in param_dict \
                and str(param_dict['secondDirectionOppositeDirection']) == 'true':
                extent_two = -extent_two
        else:
            extent_type = EXTENT_TYPE_MAP[param_dict['endBound']]

        operation = OPERATION_MAP[param_dict['operationType']]

        save_dict = {"name": feature_data['name'],
                    "type": "ExtrudeFeature",
                    #"profiles": profiles,
                    "entities": entities,
                    "operation": operation,
                    "start_extent": {"type": "ProfilePlaneStartDefinition"},
                    "extent_type": extent_type,
                    "extent_one": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_one,
                            "name": "none",
                            "role": "AlongDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    "extent_two": {
                        "distance": {
                            "type": "ModelParameter",
                            "value": extent_two,
                            "name": "none",
                            "role": "AgainstDistance"
                        },
                        "taper_angle": {
                            "type": "ModelParameter",
                            "value": 0.0,
                            "name": "none",
                            "role": "Side2TaperAngle"
                        },
                        "type": "DistanceExtentDefinition"
                    },
                    }
        return save_dict

    def _parse_boundingBox(self):
        bbox_info = self.c.eval_boundingBox(self.did, self.wid, self.eid)
        result = {"type": "BoundingBox3D",
                  "max_point": xyz_list2dict(bbox_info['maxCorner']),
                  "min_point": xyz_list2dict(bbox_info['minCorner'])}
        return result

    def parse(self):
        """parse into fusion360 gallery format, 
        only sketch and extrusion are supported.
        """
        result = {"entities": OrderedDict(), "properties": {}, "sequence": []}
        try:
            bbox = self._parse_boundingBox()
        except Exception as e:
            print(self.data_id, "bounding box failed:", e)
            return result
        result["properties"].update({"bounding_box": bbox})

        for i, feat_item in enumerate(self.feature_list['features']):
            feat_data = feat_item['message']
            feat_type = feat_data['featureType']
            feat_Id = feat_data['featureId']

            #try:
            if feat_type == 'newSketch':
                feat_dict = self._parse_sketch(feat_data)
                for k in feat_dict['profiles'].keys():
                    self.profile2sketch.update({k: feat_Id})
            elif feat_type == 'extrude':
                #feat_dict = self._parse_extrude(feat_data)
                feat_dict = self._parse_extrude_simple(feat_data)
                #self.c.eval_extrude(self.did, self.wid, self.eid, feat_Id)
                #topo = self.c.eval_sketch_topology_by_adjacency(self.did, self.wid, self.eid, feat_Id)
            else:
                feat_dict = {"type": feat_type,
                             }
                #continue
                #raise NotImplementedError(self.data_id, "unsupported feature type: {}".format(feat_type))
            #except Exception as e:
            #    print(self.data_id, "parse feature failed:", e)
            #    break
            #exit()
            result["entities"].update({feat_Id: feat_dict})
            result["sequence"].append({"index": i, "type": feat_dict['type'], "entity": feat_Id})
        return result


class SketchParser(object):
    """A parser for OnShape sketch feature list"""
    def __init__(self, client, feat_data, did, wid, eid, sketch_info, data_id=None):
        self.c = client
        self.feat_id = feat_data['featureId']
        #print(self.feat_id)
        self.feat_name = feat_data['name']
        self.feat_param = FeatureListParser.parse_feature_param(feat_data['parameters'])
        self.sketch_info = sketch_info

        self.did = did
        self.wid = wid
        self.eid = eid
        self.data_id = data_id

        #print(self.feat_param)
        #print(self.feat_param["sketchPlane"])
        geo_id = self.feat_param["sketchPlane"][0]
        response = self.c.get_entity_by_id(did, wid, eid, [geo_id], "FACE")
        self.plane = self.c.parse_face_msg(response.json()['result']['message']['value'])[0]

        self.geo_topo = self.c.eval_sketch_topology_by_adjacency(did, wid, eid, self.feat_id, self.sketch_info)
        #print("after geo_topo")
        #print(self.geo_topo)
        #self._to_local_coordinates()
        self._build_lookup()
        #print("after build_lookup")
        self.constraints = self.parse_constraints(feat_data, sketch_info)
        #print("after parse constraints")

    def parse_constraints(self, feat_data, sketch_info):
        constraints = []
        #print(self.vert_table)
        #print(self.geo_topo)
        for sketch in sketch_info["sketches"]:
            if sketch["featureId"] == self.feat_id:
                for constraint in feat_data["constraints"]:
                    reject_constraint = False
                    constraint_dict = {}
                    constraint_dict["constraintType"] = constraint["message"]["constraintType"]
                    if not constraint_dict["constraintType"] in CONSTRAINT_TYPES:
                        continue
                    #print(constraint)
                    if len(constraint["message"]["parameters"]) < 2:
                        continue
                    for param in constraint["message"]["parameters"]:
                        if param["message"]["parameterId"] == "localFirst":
                            constraint_dict["first_id"] = param["message"]["value"]
                            constraint_dict["first_local"] = True
                            for geom_ent in sketch["geomEntities"]:
                                if geom_ent["id"] == constraint_dict["first_id"]:
                                    #print(geom_ent)
                                    if geom_ent["entityType"] != "point":
                                        reject_constraint = True
                                        break
                                    constraint_dict["first_3d"] = geom_ent["point"]
                        if param["message"]["parameterId"] == "externalFirst":
                            constraint_dict["first_id"] = param["message"]["queries"][0]["message"]["geometryIds"][0]
                            constraint_dict["first_local"] = False
                            if constraint_dict["first_id"] in self.vert_table.keys():
                                constraint_dict["first_3d"] = self.vert_table[constraint_dict["first_id"]]["param"]["Vector"]
                        if param["message"]["parameterId"] == "localSecond":
                            constraint_dict["second_id"] = param["message"]["value"]
                            constraint_dict["second_local"] = True
                            for geom_ent in sketch["geomEntities"]:
                                if geom_ent["id"] == constraint_dict["second_id"]:
                                    #print(geom_ent)
                                    if geom_ent["entityType"] != "point":
                                        reject_constraint = True
                                        break
                                    constraint_dict["second_3d"] = geom_ent["point"]
                                    # get all 3d points of second_id element
                                    second_base_id = constraint_dict["second_id"].split(".")[0]
                                    second_base_pts = []
                                    for base_ent in sketch["geomEntities"]:
                                        #if constraint_dict["constraintType"] == "HORIZONTAL":
                                        #    print(second_base_id)
                                        #    exit()
                                        if second_base_id in base_ent["id"] and \
                                            base_ent["entityType"] == "point" and \
                                                ("start" in base_ent["id"] or "end" in base_ent["id"]) :
                                            if "isConstruction" in base_ent.keys() and base_ent["isConstruction"]:
                                                continue
                                            second_base_pts.append(base_ent["point"])
                                    constraint_dict["second_element_pts"] = second_base_pts
                        if param["message"]["parameterId"] == "localEntity1":
                            constraint_dict["first_id"] = param["message"]["value"]
                            constraint_dict["first_local"] = True
                            for geom_ent in sketch["geomEntities"]:
                                if geom_ent["id"] == constraint_dict["first_id"]:
                                    #print(geom_ent)
                                    if geom_ent["entityType"] == "point":
                                        constraint_dict["first_3d"] = geom_ent["point"]
                                    elif geom_ent["entityType"] == "lineSegment":
                                        constraint_dict["first_3d"] = [geom_ent["startPoint"], geom_ent["endPoint"]]
                                    else:
                                        reject_constraint = True
                                        break
                        if param["message"]["parameterId"] == "localEntity2":
                            constraint_dict["second_id"] = param["message"]["value"]
                            constraint_dict["second_local"] = True
                            for geom_ent in sketch["geomEntities"]:
                                if geom_ent["id"] == constraint_dict["second_id"]:
                                    #print(geom_ent)
                                    if geom_ent["entityType"] == "point":
                                        constraint_dict["second_3d"] = geom_ent["point"]
                                    elif geom_ent["entityType"] == "lineSegment":
                                        constraint_dict["second_3d"] = [geom_ent["startPoint"], geom_ent["endPoint"]]
                                    else:
                                        reject_constraint = True
                                        break

                        if param["message"]["parameterId"] == "localMidpoint":
                            constraint_dict["midpoint_id"] = param["message"]["value"]
                            constraint_dict["midpoint_local"] = True
                            for geom_ent in sketch["geomEntities"]:
                                if geom_ent["id"] == constraint_dict["midpoint_id"]:
                                    #print(geom_ent)
                                    if geom_ent["entityType"] == "point":
                                        constraint_dict["midpoint_3d"] = geom_ent["point"]
                                    elif geom_ent["entityType"] == "lineSegment":
                                        constraint_dict["midpoint_3d"] = [geom_ent["startPoint"], geom_ent["endPoint"]]
                                    else:
                                        reject_constraint = True
                                        break


                        if param["message"]["parameterId"] == "externalSecond":
                            constraint_dict["second_id"] = param["message"]["queries"][0]["message"]["geometryIds"][0]
                            constraint_dict["second_local"] = False
                            if constraint_dict["second_id"] in self.vert_table.keys():
                                constraint_dict["second_3d"] = self.vert_table[constraint_dict["second_id"]]["param"]["Vector"]
                    if constraint_dict["constraintType"] == "MIDPOINT" and not "midpoint_3d" in constraint_dict.keys():
                        # choose one of the two local entities
                        midpoint_key = "localEntity1"
                        if not "second_3d" in constraint_dict.keys():
                            continue
                        if len(constraint_dict["second_3d"]) != 2:
                            midpoint_key = "localEntity2"
                        for tmp_param in constraint["message"]["parameters"]:
                            if not tmp_param["message"]["parameterId"] == midpoint_key:
                                continue
                            constraint_dict["midpoint_id"] = tmp_param["message"]["value"]
                            constraint_dict["midpoint_local"] = True
                            for geom_ent in sketch["geomEntities"]:
                                if geom_ent["id"] == constraint_dict["midpoint_id"]:
                                    #print(geom_ent)
                                    if geom_ent["entityType"] == "point":
                                        constraint_dict["midpoint_3d"] = geom_ent["point"]
                                    elif geom_ent["entityType"] == "lineSegment":
                                        constraint_dict["midpoint_3d"] = [geom_ent["startPoint"], geom_ent["endPoint"]]
                                    else:
                                        reject_constraint = True
                                        break

                    if constraint_dict["constraintType"] in CONSTRAINT_TYPES:
                        # search for the sketch element which needs this midpoint
                        affected_id = constraint["message"]["entityId"]
                        #print(constraint)
                        #print(constraint_dict)
                        if constraint_dict["constraintType"] == "MIDPOINT" and constraint_dict["midpoint_local"]:
                            affected_id = constraint_dict["midpoint_id"]
                        #print(constraint_dict)
                        if constraint_dict["constraintType"] in ["HORIZONTAL", "VERTICAL"]:
                            if not ("second_local" in constraint_dict.keys() and constraint_dict["second_local"]):
                                constraint_dict["affected_element"] = []
                                continue
                            affected_id = constraint_dict["second_id"]
                        #print(constraint_dict["constraintType"])
                        #print(affected_id)
                        if ".start" in affected_id or ".end" in affected_id:
                            affected_id = ".".join(affected_id.split(".")[:-1])
                        elif "." in affected_id:
                            affected_id = affected_id.split(".")[0]
                        found_affected_id = False
                        for geom_ent in sketch["geomEntities"]:
                            if geom_ent["id"] == affected_id and geom_ent["entityType"] != "point":
                                #print(geom_ent)
                                if geom_ent["entityType"] in ["interpolatedSplineSegment", "unknownGeometry"]:
                                    #if "midpoint_id" in constraint_dict.keys():
                                    #    print("if affected id", affected_id)
                                    # look for endpoints
                                    tmp_geom_ent = {"entityType": "lineSegment"}
                                    for point_geom_ent in sketch["geomEntities"]:
                                        if point_geom_ent["id"] == geom_ent["endPointIds"][0]:
                                            tmp_geom_ent["startPoint"] = point_geom_ent["point"]
                                        if point_geom_ent["id"] == geom_ent["endPointIds"][1]:
                                            tmp_geom_ent["endPoint"] = point_geom_ent["point"]
                                    constraint_dict["affected_element"] = tmp_geom_ent
                                else:
                                    #if "midpoint_id" in constraint_dict.keys():
                                    #    print("else affected id", affected_id)
                                    constraint_dict["affected_element"] = geom_ent
                                found_affected_id = True
                            #if constraint_dict["affected_element"]["entityType"] == "unknownGeometry":
                            #    end_points = []
                        #print(found_affected_id)
                        if not found_affected_id:# and constraint_dict["constraintType"] in ["MIDPOINT", "HORIZONTAL", "VERTICAL"]:
                            #if "midpoint_id" in constraint_dict.keys():
                            #    print("not found affected id", affected_id)
                            if "midpoint_id" in constraint_dict.keys():
                                affected_id = constraint_dict["midpoint_id"]
                            if "." in affected_id:
                                affected_id = affected_id.split(".")[0]
                            found_affected_id = False
                            for geom_ent in sketch["geomEntities"]:
                                if geom_ent["id"] == affected_id:
                                    #print(geom_ent)
                                    constraint_dict["affected_element"] = geom_ent

                    if reject_constraint:
                        continue
                    constraints.append(constraint_dict)
        #print(constraints)
        return constraints

    def _to_local_coordinates(self):
        """transform into local coordinate system"""
        self.origin = np.array(self.plane["origin"])
        self.z_axis = np.array(self.plane["normal"])
        self.x_axis = np.array(self.plane["x"])
        self.y_axis = np.cross(self.plane["normal"], self.plane["x"])
        for item in self.geo_topo["vertices"]:
            old_vec = np.array(item["param"]["Vector"])
            new_vec = old_vec - self.origin
            item["param"]["Vector"] = [np.dot(new_vec, self.x_axis), 
                                       np.dot(new_vec, self.y_axis), 
                                       np.dot(new_vec, self.z_axis)]

        for item in self.geo_topo["edges"]:
            if item["param"]["type"] == "Circle":
                old_vec = np.array(item["param"]["coordSystem"]["origin"])
                new_vec = old_vec - self.origin
                item["param"]["coordSystem"]["origin"] = [np.dot(new_vec, self.x_axis),
                                                          np.dot(new_vec, self.y_axis),
                                                          np.dot(new_vec, self.z_axis)]

    def _build_lookup(self):
        """build a look up table with entity ID as key"""
        edge_table = {}
        for item in self.geo_topo["edges"]:
            edge_table.update({item["id"]: item})
        self.edge_table = edge_table

        vert_table = {}
        for item in self.geo_topo["vertices"]:
            vert_table.update({item["id"]: item})

        vert_table["IB"] = {'id': 'IB', 'param': {'Vector': (0.0, 0.0, 0.0), 'unit': ('METER', 1)}}
        # include origin IB
        self.vert_table = vert_table

    def _parse_edges_to_loops(self, all_edge_ids):
        #print("new face")
        #print(all_edge_ids)
        """sort all edges of a face into loops."""
        # FIXME: this can be error-prone. bug situation: one vertex connected to 3 edges
        vert2edge = {}
        for edge_id in all_edge_ids:
            item = self.edge_table[edge_id]
            for vert in item["vertices"]:
                if vert not in vert2edge.keys():
                    vert2edge.update({vert: [item["id"]]})
                else:
                    vert2edge[vert].append(item["id"])

        all_loops = []
        unvisited_edges = copy.copy(all_edge_ids)
        pts = []
        for vert in vert2edge:
            #print(self.vert_table[vert])
            pts.append(self.vert_table[vert]["param"]["Vector"])
        pts = np.array(pts)
        while len(unvisited_edges) > 0:
            #print("unvisited edges", unvisited_edges)
            cur_edge = unvisited_edges[0]
            #print("cur_edge", cur_edge)
            unvisited_edges.remove(cur_edge)
            loop_edge_ids = [cur_edge]
            if len(self.edge_table[cur_edge]["vertices"]) == 0:  # no corresponding vertices
                pass
            else:
                loop_start_point, cur_end_point = self.edge_table[cur_edge]["vertices"][0], \
                                                  self.edge_table[cur_edge]["vertices"][-1]
                while cur_end_point != loop_start_point:
                    # find next connected edge
                    edges = vert2edge[cur_end_point][:]
                    #print("edges", edges)
                    edges.remove(cur_edge)
                    cur_edge = edges[0]
                    #print("cur_edge", cur_edge)
                    loop_edge_ids.append(cur_edge)
                    unvisited_edges.remove(cur_edge)

                    # find next enc_point
                    points = self.edge_table[cur_edge]["vertices"][:]
                    points.remove(cur_end_point)
                    cur_end_point = points[0]
            all_loops.append(loop_edge_ids)
        return all_loops

    def _parse_edge_to_fusion360_format(self, edge_id):
        """parse a edge into fusion360 gallery format. Only support 'Line', 'Circle' and 'Arc'."""
        edge_data = self.edge_table[edge_id]
        edge_type = edge_data["param"]["type"]
        if edge_type == "Line":
            start_id, end_id = edge_data["vertices"]
            start_point = xyz_list2dict(self.vert_table[start_id]["param"]["Vector"])
            end_point = xyz_list2dict(self.vert_table[end_id]["param"]["Vector"])
            curve_dict = OrderedDict({"type": "Line3D", "start_point": start_point,
                                      "end_point": end_point, "curve": edge_id})
        elif edge_type == "Circle" and len(edge_data["vertices"]) == 2: # an Arc
            radius = edge_data["param"]["radius"]
            start_id, end_id = edge_data["vertices"]
            start_point = xyz_list2dict(self.vert_table[start_id]["param"]["Vector"])
            end_point = xyz_list2dict(self.vert_table[end_id]["param"]["Vector"])
            center_point = xyz_list2dict(edge_data["param"]["coordSystem"]["origin"])
            normal = xyz_list2dict(edge_data["param"]["coordSystem"]["zAxis"])

            start_vec = np.array(self.vert_table[start_id]["param"]["Vector"]) - \
                        np.array(edge_data["param"]["coordSystem"]["origin"])
            end_vec = np.array(self.vert_table[end_id]["param"]["Vector"]) - \
                      np.array(edge_data["param"]["coordSystem"]["origin"])
            start_vec = start_vec / np.linalg.norm(start_vec)
            end_vec = end_vec / np.linalg.norm(end_vec)

            start_angle = angle_from_vector_to_x(start_vec)
            end_angle = angle_from_vector_to_x(end_vec)
            # keep it counter-clockwise first
            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle
                start_vec, end_vec = end_vec, start_vec
            sweep_angle = abs(start_angle - end_angle)

            # # decide direction arc by curve length
            # edge_len = self.c.eval_curveLength(self.did, self.wid, self.eid, edge_id)
            # _len = sweep_angle * radius
            # _len_other = (2 * np.pi - sweep_angle) * radius
            # if abs(edge_len - _len) > abs(edge_len - _len_other):
            #     sweep_angle = 2 * np.pi - sweep_angle
            #     start_vec = end_vec

            # decide direction by middle point
            midpoint = self.c.eval_curve_midpoint(self.did, self.wid, self.eid, edge_id)
            mid_vec = np.array(midpoint) - self.origin
            mid_vec = np.array([np.dot(mid_vec, self.x_axis), np.dot(mid_vec, self.y_axis), np.dot(mid_vec, self.z_axis)])
            mid_vec = mid_vec - np.array(edge_data["param"]["coordSystem"]["origin"])
            mid_vec = mid_vec / np.linalg.norm(mid_vec)
            mid_angle_real = angle_from_vector_to_x(mid_vec)
            mid_angle_now = (start_angle + end_angle) / 2            
            if round(mid_angle_real, 3) != round(mid_angle_now, 3):
                sweep_angle = 2 * np.pi - sweep_angle
                start_vec = end_vec

            ref_vec_dict = xyz_list2dict(list(start_vec))
            curve_dict = OrderedDict({"type": "Arc3D", "start_point": start_point, "end_point": end_point,
                          "center_point": center_point, "radius": radius, "normal": normal,
                          "start_angle": 0.0, "end_angle": sweep_angle, "reference_vector": ref_vec_dict,
                          "curve": edge_id})
        elif edge_type == "Circle" and len(edge_data["vertices"]) < 2:
            # NOTE: treat the circle with only one connected vertex as a full circle
            radius = edge_data["param"]["radius"]
            center_point = xyz_list2dict(edge_data["param"]["coordSystem"]["origin"])
            normal = xyz_list2dict(edge_data["param"]["coordSystem"]["zAxis"])
            curve_dict = OrderedDict({"type": "Circle3D", "center_point": center_point, "radius": radius, "normal": normal,
                          "curve": edge_id})
        else:
            raise NotImplementedError(edge_type, edge_data["vertices"])
        return curve_dict

    def parse_to_fusion360_format(self):
        """parse sketch feature into fusion360 gallery format"""
        name = self.feat_name

        # transform & reference plane
        transform_dict = {"origin": xyz_list2dict(self.plane["origin"]),
                          "z_axis": xyz_list2dict(self.plane["normal"]),
                          "x_axis": xyz_list2dict(self.plane["x"]),
                          "y_axis": xyz_list2dict(list(np.cross(self.plane["normal"], self.plane["x"])))}
        ref_plane_dict = {}

        # faces
        profiles_dict = {"faces": self.geo_topo["faces"],
                         "edges": self.edge_table,
                         "vertices": self.vert_table}
        ## profiles
        #profiles_dict = {}
        #for item in self.geo_topo['faces']:
        #    # profile level
        #    profile_id = item['id']
        #    all_edge_ids = item['edges']
        #    print(item)
        #    edge_ids_per_loop = self._parse_edges_to_loops(all_edge_ids)
        #    all_loops = []
        #    for loop in edge_ids_per_loop:
        #        curves = [self._parse_edge_to_fusion360_format(edge_id) for edge_id in loop]
        #        loop_dict = {"is_outer": True, "profile_curves": curves}
        #        all_loops.append(loop_dict)
        #    profiles_dict.update({profile_id: {"loops": all_loops, "properties": {}}})

        entity_dict = {"name": name, "type": "Sketch", "profiles": profiles_dict,
                       "transform": transform_dict, "reference_plane": ref_plane_dict,
                       "constraints": self.constraints}
        return entity_dict
