'''
client
======

Convenience functions for working with the Onshape API
'''

from .onshape import Onshape

import mimetypes
import random
import string
import os
import numpy as np
import polyscope as ps
from skspatial.objects import Line

def plot_curves(curves, name_prefix="", color=(0, 0, 1), radius=0.005, enabled=True, type_ids=None, type_colors=None):
    for curve_id, curve_geom in enumerate(curves):
        if len(curve_geom) == 1:
            edges_array = np.array([[0, 0]])
        else:
            edges_array = np.array([[i, i + 1] for i in range(len(curve_geom) - 1)])
        edge_color = color
        if type_ids is not None:
            edge_color = type_colors[type_ids[curve_id]]
        ps.register_curve_network(name_prefix + "_" + str(curve_id), nodes=np.array(curve_geom),
                                  edges=edges_array, color=edge_color, radius=radius, enabled=enabled)

def passes_through_both_points(seg1, seg2):
    v1 = seg1[1] - seg1[0]
    v1 /= np.linalg.norm(seg1)
    v2 = seg2[1] - seg2[0]
    v2 /= np.linalg.norm(seg2)
    if np.isclose(np.abs(np.dot(v1, v2)), 1.0, atol=1e-4) and \
            (np.isclose(np.linalg.norm(seg1[0]-seg2[0]), 0.0, atol=1e-4) or
             np.isclose(np.linalg.norm(seg1[1]-seg2[0]), 0.0, atol=1e-4) or
             np.isclose(np.linalg.norm(seg1[0]-seg2[1]), 0.0, atol=1e-4) or
             np.isclose(np.linalg.norm(seg1[1]-seg2[1]), 0.0, atol=1e-4)):
        return True
    return False

class Client():
    '''
    Defines methods for testing the Onshape API. Comes with several methods:

    - Create a document
    - Delete a document
    - Get a list of documents

    Attributes:
        - stack (str, default='https://cad.onshape.com'): Base URL
        - logging (bool, default=True): Turn logging on or off
    '''

    def __init__(self, stack='https://cad.onshape.com', logging=True):
        '''
        Instantiates a new Onshape client.

        Args:
            - stack (str, default='https://cad.onshape.com'): Base URL
            - logging (bool, default=True): Turn logging on or off
        '''

        self._stack = stack
        self._api = Onshape(stack=stack, logging=logging)

    def new_document(self, name='Test Document', owner_type=0, public=False):
        '''
        Create a new document.

        Args:
            - name (str, default='Test Document'): The doc name
            - owner_type (int, default=0): 0 for user, 1 for company, 2 for team
            - public (bool, default=False): Whether or not to make doc public

        Returns:
            - requests.Response: Onshape response data
        '''

        payload = {
            'name': name,
            'ownerType': owner_type,
            'isPublic': public
        }

        return self._api.request('post', '/api/documents', body=payload)

    def rename_document(self, did, name):
        '''
        Renames the specified document.

        Args:
            - did (str): Document ID
            - name (str): New document name

        Returns:
            - requests.Response: Onshape response data
        '''

        payload = {
            'name': name
        }

        return self._api.request('post', '/api/documents/' + did, body=payload)

    def delete_document(self, did):
        '''
        Delete the specified document.

        Args:
            - did (str): Document ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('delete', '/api/documents/' + did)

    def get_document(self, did):
        '''
        Get details for a specified document.

        Args:
            - did (str): Document ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/documents/' + did)

    def list_documents(self):
        '''
        Get list of documents for current user.

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/documents')

    def create_assembly(self, did, wid, name='My Assembly'):
        '''
        Creates a new assembly element in the specified document / workspace.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - name (str, default='My Assembly')

        Returns:
            - requests.Response: Onshape response data
        '''

        payload = {
            'name': name
        }

        return self._api.request('post', '/api/assemblies/d/' + did + '/w/' + wid, body=payload)

    def copy_workspace(self, did, wid, new_name, timeout=None):

        payload = {
            "newName": new_name,
            "isPublic": "True"
        }

        return self._api.request('post', '/api/documents/' + did + '/workspaces/' + wid + '/copy', body=payload)

    def update_feature_studio_content(self, did, wid, eid, query, timeout=None):

        return self._api.request(
            'post', '/api/featurestudios/d/' + did + '/w/' + wid + '/e/' + eid,
            body=query,
            timeout=timeout)

    def update_feature_rollback(self, did, wid, eid, query, timeout=None):

        return self._api.request(
            'post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features/rollback',
            body=query,
            timeout=timeout)

    def export_stl(self, did, wid, eid, timeout=None):

        return self._api.request(
            'get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/stl',
            timeout=timeout)

    def get_elements(self, did, wid, timeout=None):

        return self._api.request(
            'get', '/api/documents/d/' + did + '/w/' + wid + '/elements',
            timeout=timeout)

    def get_features(self, did, wid, eid, timeout=None):
        '''
        Gets the feature list for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - timeout (float): Timeout passed to requests.request().

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request(
            'get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features',
            timeout=timeout)

    # example of a featureId use case
    #timeout = timeout, query = {"featureId": "FA89C3SaEuH4uAq_0"})

    def sketch_information(self, did, wid, eid, payload=None):
        '''
        Get information for sketches in a part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/sketches', query=payload)

    def get_bodydetails(self, did, wid, eid, payload=None):

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/bodydetails', query=payload)

    def get_thumbnail_with_size(self, did, wid, sz):
        '''
        Gets the thumbnail image for specified document / workspace with size sz.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - sz (int): Dimension of square image

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/thumbnails/d/' + did + '/w/' + wid +  ('/s/%ix%i' % (sz, sz)), headers={'Accept': 'application/vnd.onshape.v1+octet-stream', 'Content-Type': 'application/json'})

    def get_tess_sketch_entities(self, did, wid, eid, sid):
        '''
        Gets the tessellations of the sketch entities in a sketch.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - sid (str): Sketch feature ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/sketches/' + sid + '/tessellatedentities')

    def add_feature(self, did, wid, eid, payload):
        '''
        Add feature for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''
        return self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features', body=payload).json()

    def delete_feature(self, did, wid, eid, fid):
        '''
        Delete feature for specified document / workspace / part studio / feature.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - fid (str): Feature ID

        Returns:
            - requests.Response: Onshape response data
        '''
        return self._api.request('delete', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features/featureid/' + fid)

    def get_partstudio_tessellatededges(self, did, wid, eid):
        '''
        Gets the tessellation of the edges of all parts in a part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/tessellatededges')

    def get_partstudio_tessellatedface(self, did, wid, eid, fid):
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/tessellatedfaces',
                                 query={"faceId": fid}).json()

    def get_partstudio_tessellatedfaces(self, did, wid, eid):
        '''
        Gets the tessellation of the edges of all parts in a part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/tessellatedfaces')

    def upload_blob(self, did, wid, filepath='./blob.json'):
        '''
        Uploads a file to a new blob element in the specified doc.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - filepath (str, default='./blob.json'): Blob element location

        Returns:
            - requests.Response: Onshape response data
        '''

        chars = string.ascii_letters + string.digits
        boundary_key = ''.join(random.choice(chars) for i in range(8))

        mimetype = mimetypes.guess_type(filepath)[0]
        encoded_filename = os.path.basename(filepath)
        file_content_length = str(os.path.getsize(filepath))
        blob = open(filepath)

        req_headers = {
            'Content-Type': 'multipart/form-data; boundary="%s"' % boundary_key
        }

        # build request body
        payload = '--' + boundary_key + \
            '\r\nContent-Disposition: form-data; name="encodedFilename"\r\n\r\n' + \
            encoded_filename + '\r\n'
        payload += '--' + boundary_key + \
            '\r\nContent-Disposition: form-data; name="fileContentLength"\r\n\r\n' + \
            file_content_length + '\r\n'
        payload += '--' + boundary_key + \
            '\r\nContent-Disposition: form-data; name="file"; filename="' + \
            encoded_filename + '"\r\n'
        payload += 'Content-Type: ' + mimetype + '\r\n\r\n'
        payload += blob.read()
        payload += '\r\n--' + boundary_key + '--'

        return self._api.request('post', '/api/blobelements/d/' + did + '/w/' + wid, headers=req_headers, body=payload)

    def part_studio_stl(self, did, wid, eid):
        '''
        Exports STL export from a part studio

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        req_headers = {
            'Accept': 'application/vnd.onshape.v1+octet-stream'
        }
        #return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/stl', headers=req_headers)
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/stl',
                                 query={"units": "meter"})#, headers=req_headers)

    def get_entity_by_id(self, did, wid, eid, geo_id, entity_type):
        """get the parameters of geometry entity for specified entity id and type

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - geo_id (str): geometry entity ID
            - entity_type (str): 'VERTEX', 'EDGE' or 'FACE'

        Returns:
            - requests.Response: OnShape response data
        """
        func_dict = {"VERTEX": ("evVertexPoint", "vertex"),
                     "EDGE": ("evCurveDefinition", "edge"),
                     "FACE": ("evSurfaceDefinition", "face")}
        body = {
            "script":
                "function(context is Context, queries) { " +
                "   var res_list = [];"
                "   var q_arr = evaluateQuery(context, queries.id);"
                "   for (var i = 0; i < size(q_arr); i+= 1){"
                "       var res = %s(context, {\"%s\": q_arr[i]});" % (func_dict[entity_type][0], func_dict[entity_type][1]) +
                "       res_list = append(res_list, res);"
                "   }"
                "   return res_list;"
                "}",
            "queries": [{ "key" : "id", "value" : geo_id }]
        }
        #body = {
        #    "script":
        #        "function(context is Context, queries) { " +
        #        "   var res_list = [];"
        #        #"   var q_faces = qHasAttributeWithValue(qEverything(EntityType.EDGE), \"GeometryType\", \"CIRCLE\");"
        #        #"   var q_faces = qEverything(EntityType.EDGE);"
        #        #"   var q_arr = evaluateQuery(context, q_faces);"
        #        "   var lineQuery = qCreatedBy(id + \"Sketch 1\", EntityType.EDGE);"
        #        "   var q_arr = evaluateQuery(context, lineQuery);"
        #        #"   var attributes = getAttributes(context, {\"entities\": q_arr});"
        #        #"   return attributes;"
        #        #"   return q_arr;"
        #        "   for (var i = 0; i < size(q_arr); i+= 1){"
        #        #"       var res = %s(context, {\"%s\": q_arr[i]});" % (func_dict[entity_type][0], func_dict[entity_type][1]) +
        #        "       var res = evCurveDefinition(context, {\"edge\": q_arr[i]});"
        #        #"       var attribute = getAttributes(context, {\"entities\": [res]});"
        #        "       res_list = append(res_list, res);"
        #        #"       res_list = append(res_list, attribute);"
        #        "   }"
        #        "   return res_list;"
        #        "}",
        #    "queries": []
        #}
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        #"   var q_arr = evaluateQuery(context, queries.id);"
        #"queries": [{"key": "id", "value": geo_id}]

        return res

    @staticmethod
    def parse_face_msg(response):
        """parse face parameters from OnShape response data"""
        # data = response.json()['result']['message']['value']
        data = [response] if not isinstance(response, list) else response
        faces = []
        for item in data:
            face_msg = item['message']['value']
            #face_type = item['message']['typeTag'].encode('utf-8')
            face_type = item['message']['typeTag']
            face_param = {'type': face_type}
            for msg in face_msg:
                #k = msg['message']['key']['message']['value'].encode('utf-8')
                k = msg['message']['key']['message']['value']
                v_item = msg['message']['value']['message']['value']
                if k == 'coordSystem':
                    v = Client.parse_coord_msg(v_item)
                elif isinstance(v_item, list):
                    v = [round(x['message']['value'], 8) for x in v_item]
                else:
                    if isinstance(v_item, float):
                        v = round(v_item, 8)
                    else:
                        v = v_item
                        #v = v_item.encode('utf-8')
                face_param.update({k: v})
            faces.append(face_param)
        return faces

    def eval_boundingBox(self, did, wid, eid):
        '''
        Get bounding box of all solid bodies for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - dict: {'maxCorner': [], 'minCorner': []}
        '''
        body = {
            "script":
                "function(context is Context, queries) { " +
                "   var q_body = qBodyType(qEverything(EntityType.BODY), BodyType.SOLID);"
                "   var bbox = evBox3d(context, {'topology': q_body});"
                "   return bbox;"
                "}",
            "queries": []
        }
        response = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        bbox_values = response.json()['result']['message']['value']
        result = {}
        for item in bbox_values:
            k = item['message']['key']['message']['value']
            point_values = item['message']['value']['message']['value']
            v = [x['message']['value'] for x in point_values]
            result.update({k: v})
        return result

    def eval_extrude(self, did, wid, eid, feat_id):
        body = {
            "script":
                "function(context is Context, queries) { "
                # "   var q_face = qSketchRegion(makeId(\"%s\"));" % feat_id +
                "   var q_face = qCreatedBy(makeId(\"%s\"), EntityType.FACE);" % feat_id +
                "   return q_face;"
                "}",
                "queries": []
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)

    def create_feature_studio(self, did, wid):
        res = self._api.request('post', '/api/featurestudios/d/' + did + '/w/' + wid)
        return res.json()["id"]

    def eval_featurescript(self, did, wid, eid, script, queries=[]):
        body = {
            "script": script,
            "queries": queries
        }

        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        return res.json()

    def sketch_entity_curve_definition(self, did, wid, eid, feat_id, sketch_ent_id):
        body = {
            "script":
                "function(context is Context, queries) { "
                #"   return evCurveDefinition(context, {edge: qCreatedBy(makeId(\"hlTPiZy2Btf7\"), EntityType.EDGE)});" ,
                #"   return  evCurveDefinition(context, {edge: sketchEntityQuery(makeId(\"Fxd6guPXhQ23lk3_83\"), EntityType.EDGE, \"hlTPiZy2Btf7.startHandle\")});"
                "   return evCurveDefinition(context, {edge: sketchEntityQuery(makeId(\""+feat_id+"\"), EntityType.EDGE, \""+sketch_ent_id+"\")});"
                #"   return  evCurveDefinition(context, {edge: sketchEntityQuery(makeId(\"FUiQ2isKDvaRJsE_82\"), EntityType.EDGE, \"JQrTd64MuK64.bottom\")});"
                #"return transientQueriesToStrings(evaluateQuery(context, sketchEntityQuery(makeId(\"Fxd6guPXhQ23lk3_83\"), EntityType.EDGE, \"hlTPiZy2Btf7.startHandle\")));"
                #"   var q_face = qCreatedBy(makeId(\"Fxd6guPXhQ23lk3_83\"), EntityType.EDGE);"
                #"   return evaluateQuery(context, q_face);"
                "}",
            "queries": []
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        return res.json()

    def eval_sketch_topology_by_adjacency(self, did, wid, eid, feat_id, sketch_info):
        """parse the hierarchical parametric geometry&topology (face -> edges -> vertex)
        from a specified sketch feature ID.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - feat_id (str): Feature ID of a sketch

        Returns:
            - dict: a hierarchical parametric representation
        """
        body = {
            "script":
                "function(context is Context, queries) { "
                "   var topo = {};"
                "   topo.faces = [];"
                "   topo.edges = [];"
                "   topo.vertices = [];"
                "   var all_edge_ids = [];"
                "   var all_vertex_ids = [];"
                "                           "
                #"   var q_face = qSketchRegion(makeId(\"%s\"));" % feat_id +
                "   var q_face = qCreatedBy(makeId(\"%s\"), EntityType.FACE);" % feat_id +
                "   var face_arr = evaluateQuery(context, q_face);"
                "   for (var i = 0; i < size(face_arr); i += 1) {"
                "       var face_topo = {};"
                "       const face_id = transientQueriesToStrings(face_arr[i]);"
                "       face_topo.id = face_id;"
                "       face_topo.edges = [];"
                "       face_topo.param = evSurfaceDefinition(context, {face: face_arr[i]});"
                "                            "
                # "       var q_edge = qLoopEdges(q_face);"
                "       var q_edge = qConstructionFilter(qAdjacent(face_arr[i], AdjacencyType.EDGE, EntityType.EDGE), ConstructionObject.NO);"
                "       var edge_arr = evaluateQuery(context, q_edge);"
                "       for (var j = 0; j < size(edge_arr); j += 1) {"
                "           var edge_topo = {};"
                "           const edge_id = transientQueriesToStrings(edge_arr[j]);"
                "           edge_topo.id = edge_id;"
                "           edge_topo.vertices = [];"
                "           edge_topo.param = evCurveDefinition(context, {edge: edge_arr[j]});" # 
                "           face_topo.edges = append(face_topo.edges, edge_id);"
                "                                  "
                "           var q_vertex = qAdjacent(edge_arr[j], AdjacencyType.VERTEX, EntityType.VERTEX);"
                "           var vertex_arr = evaluateQuery(context, q_vertex);"
                "           for (var k = 0; k < size(vertex_arr); k += 1) {"
                "               var vertex_topo = {};"
                "               const vertex_id = transientQueriesToStrings(vertex_arr[k]);"
                "               vertex_topo.id = vertex_id;"
                "               vertex_topo.param = evVertexPoint(context, {vertex: vertex_arr[k]});"
                "               edge_topo.vertices = append(edge_topo.vertices, vertex_id);"
                "               if (isIn(vertex_id, all_vertex_ids)){continue;}"
                "               all_vertex_ids = append(all_vertex_ids, vertex_id);"
                "               topo.vertices = append(topo.vertices, vertex_topo);"
                "           }"
                "           if (isIn(edge_id, all_edge_ids)){continue;}"
                "           all_edge_ids = append(all_edge_ids, edge_id);"
                "           topo.edges = append(topo.edges, edge_topo);"
                "       }"
                "       topo.faces = append(topo.faces, face_topo);"
                "   }"
                "   if (size(topo.faces) == 0){"
                "   "
                "       var face_topo = {};"
                "       face_topo.id = 0;"
                "       face_topo.edges = [];"
                "   var q_edge = qCreatedBy(makeId(\"%s\"), EntityType.EDGE);" % feat_id +
                "       var edge_arr = evaluateQuery(context, q_edge);"
                "       for (var j = 0; j < size(edge_arr); j += 1) {"
                "           var edge_topo = {};"
                "           const edge_id = transientQueriesToStrings(edge_arr[j]);"
                "           edge_topo.id = edge_id;"
                "           edge_topo.vertices = [];"
                "           edge_topo.param = evCurveDefinition(context, {edge: edge_arr[j]});" # 
                "           face_topo.edges = append(face_topo.edges, edge_id);"
                "                                  "
                "           var q_vertex = qAdjacent(edge_arr[j], AdjacencyType.VERTEX, EntityType.VERTEX);"
                "           var vertex_arr = evaluateQuery(context, q_vertex);"
                "           for (var k = 0; k < size(vertex_arr); k += 1) {"
                "               var vertex_topo = {};"
                "               const vertex_id = transientQueriesToStrings(vertex_arr[k]);"
                "               vertex_topo.id = vertex_id;"
                "               vertex_topo.param = evVertexPoint(context, {vertex: vertex_arr[k]});"
                "               edge_topo.vertices = append(edge_topo.vertices, vertex_id);"
                "               if (isIn(vertex_id, all_vertex_ids)){continue;}"
                "               all_vertex_ids = append(all_vertex_ids, vertex_id);"
                "               topo.vertices = append(topo.vertices, vertex_topo);"
                "           }"
                "           if (isIn(edge_id, all_edge_ids)){continue;}"
                "           all_edge_ids = append(all_edge_ids, edge_id);"
                "           topo.edges = append(topo.edges, edge_topo);"
                "       }"
                "       topo.faces = append(topo.faces, face_topo);"
                "}"
                "   return topo;"
                "}",
            "queries": []
        }
        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body=body)
        #print(res.json())

        # query curve definitions for handle lines
        handle_ids = []
        handle_curves = {}
        for sketch in sketch_info["sketches"]:
            if sketch["featureId"] == feat_id:
                for ent in sketch["geomEntities"]:
                    if "Handle" in ent["id"] and ent["entityType"] == "lineSegment":
                        handle_ids.append(ent["id"])
                        handle_curves[ent["id"]] = [np.array(ent["startPoint"], dtype=np.float64),
                                                    np.array(ent["endPoint"], dtype=np.float64)]
        #print(handle_ids)
        #handle_curve_defs = []
        #for handle_id in handle_ids:
        #    handle_curve_def = self.sketch_entity_curve_definition(did, wid, eid, feat_id, handle_id)
        #    handle_curve_defs.append(handle_curve_def)
        #    print(handle_curve_def)
        #print("handle_curves")
        #print(handle_curves)

        res_msg = res.json()['result']['message']['value']
        tmp_vertices = {}
        for item in res_msg:
            item_msg = item['message']
            k_str = item_msg['key']['message']['value']
            v_item = item_msg['value']['message']['value']
            # add vertices only when they figure as non-handle edge
            for item_x in v_item:
                v_item_x = item_x['message']['value']
                for item_y in v_item_x:
                    k = item_y['message']['key']['message']['value']
                    v_msg = item_y['message']['value']

                    if k == 'param':
                        if k_str == 'vertices':
                            v = Client.parse_vertex_msg(v_msg)[0]
                            v_id = [i["message"]["value"]["message"]["value"] for i in item_x["message"]["value"] if i["message"]["key"]["message"]["value"] == "id"][0]
                            #exit()
                            tmp_vertices[v_id] = v

        topo = {}
        vertices_list = []
        vertices_geo_dict = {}
        for item in res_msg:
            item_msg = item['message']
            #k_str = item_msg['key']['message']['value'].encode('utf-8')  # faces, edges
            k_str = item_msg['key']['message']['value']
            if k_str == 'vertices':
                continue
            v_item = item_msg['value']['message']['value']
            outer_list = []
            # add vertices only when they figure as non-handle edge
            for item_x in v_item:
                v_item_x = item_x['message']['value']
                geo_dict = {}
                for item_y in v_item_x:
                    #k = item_y['message']['key']['message']['value'].encode('utf-8')  # id, edges/vertices
                    k = item_y['message']['key']['message']['value']
                    v_msg = item_y['message']['value']

                    if k_str == "edges" and k == "vertices":
                        vertices_ids = [i["message"]["value"] for i in v_msg["message"]["value"]]
                        vertices = [np.array(tmp_vertices[v_id]["Vector"], dtype=np.float64) for v_id in vertices_ids]
                        is_handle = False
                        if len(vertices) == 2:
                            for handle_curve in handle_curves.values():
                                #print("comparison")
                                #print(handle_curve)
                                #print(vertices)
                                if passes_through_both_points(vertices, handle_curve):
                                #if (np.isclose(np.linalg.norm(vertices[0]-handle_curve[0]), 0.0, atol=1e-4) and np.isclose(np.linalg.norm(vertices[1]-handle_curve[1]), 0.0, atol=1e-4)) or \
                                #    (np.isclose(np.linalg.norm(vertices[0]-handle_curve[1]), 0.0, atol=1e-4) and np.isclose(np.linalg.norm(vertices[1]-handle_curve[0]), 0.0, atol=1e-4)):
                                    is_handle = True
                                    break
                        #print(is_handle, vertices_ids)
                        #ps.init()
                        #ps.remove_all_structures()
                        #plot_curves(handle_curves.values(), name_prefix="handle_")
                        #plot_curves([vertices], name_prefix="vertices_")
                        #ps.show()
                        if not is_handle:
                            for v_id in vertices_ids:
                            #    vertices_geo_dict.update({"param": tmp_vertices[v_id]})
                                already_added = False
                                for backup_v in vertices_list:
                                    if backup_v["id"] == v_id:
                                        already_added = True
                                        break
                                if not already_added:
                                    vertices_list.append({"id": v_id, "param": tmp_vertices[v_id]})

                    #if k.decode('utf-8') == 'param':
                    if k == 'param':
                        #if k_str.decode('utf-8') == 'faces':
                        #    v = Client.parse_face_msg(v_msg)[0]
                        #elif k_str.decode('utf-8') == 'edges':
                        #    v = Client.parse_edge_msg(v_msg)[0]
                        #elif k_str.decode('utf-8') == 'vertices':
                        #    v = Client.parse_vertex_msg(v_msg)[0]
                        if k_str == 'faces':
                            v = Client.parse_face_msg(v_msg)[0]
                        elif k_str == 'edges':
                            skip = False
                            #for handle_curve_def in handle_curve_defs:
                            #    print("HANDLE CURVE DEF")
                            #    print(handle_curve_def)
                            #    print(v_msg)
                            #    #print()
                            #    print(v_msg == handle_curve_def["result"])
                            #    #if feat_id == "Fxd6guPXhQ23lk3_83":
                            #    #    exit()
                            #    #if v_msg == handle_curve_def["result"]:
                            #    #    print("SKIPPED")
                            #    #    skip = True
                            #    #    exit()
                            #    #    break
                            if skip:
                                continue
                            v = Client.parse_edge_msg(v_msg)[0]
                        #elif k_str == 'vertices':
                        #    v = Client.parse_vertex_msg(v_msg)[0]
                        else:
                            raise ValueError
                    elif isinstance(v_msg['message']['value'], list):
                        #v = [a['message']['value'].encode('utf-8') for a in v_msg['message']['value']]
                        v = [a['message']['value'] for a in v_msg['message']['value']]
                    else:
                        #v = v_msg['message']['value'].encode('utf-8')
                        v = v_msg['message']['value']
                    geo_dict.update({k: v})
                outer_list.append(geo_dict)
            topo.update({k_str: outer_list})
            #print({k_str: outer_list})

        topo.update({"vertices": vertices_list})
        #print({"vertices": vertices_list})
        #exit()
        return topo
    @staticmethod
    def parse_vertex_msg(response):
        """parse vertex parameters from OnShape response data"""
        # data = response.json()['result']['message']['value']
        data = [response] if not isinstance(response, list) else response
        vertices = []
        for item in data:
            xyz_msg = item['message']['value']
            #xyz_type = item['message']['typeTag'].encode('utf-8')
            xyz_type = item['message']['typeTag']
            p = []
            for msg in xyz_msg:
                p.append(round(msg['message']['value'], 8))
            unit = xyz_msg[0]['message']['unitToPower'][0]
            #unit_exp = (unit['key'].encode('utf-8'), unit['value'])
            unit_exp = (unit['key'], unit['value'])
            vertices.append({xyz_type: tuple(p), 'unit': unit_exp})
        return vertices

    @staticmethod
    def parse_coord_msg(response):
        """parse coordSystem parameters from OnShape response data"""
        coord_param = {}
        for item in response:
            k_msg = item['message']['key']
            #k = k_msg['message']['value'].encode('utf-8')
            k = k_msg['message']['value']
            v_msg = item['message']['value']
            v = [round(x['message']['value'], 8) for x in v_msg['message']['value']]
            coord_param.update({k: v})
        return coord_param

    @staticmethod
    def parse_edge_msg(response):
        """parse edge parameters from OnShape response data"""
        # data = response.json()['result']['message']['value']
        data = [response] if not isinstance(response, list) else response
        edges = []
        for item in data:
            edge_msg = item['message']['value']
            #edge_type = item['message']['typeTag'].encode('utf-8')
            edge_type = item['message']['typeTag']
            #print("edge_type", edge_type)
            #if edge_type == "Line":
            #    print("here")
            #    print(edge_msg)
            edge_param = {'type': edge_type}
            for msg in edge_msg:
                #k = msg['message']['key']['message']['value'].encode('utf-8')
                k = msg['message']['key']['message']['value']
                v_item = msg['message']['value']['message']['value']
                #if k.decode('utf-8') == 'coordSystem':
                if k == 'coordSystem':
                    v = Client.parse_coord_msg(v_item)
                elif isinstance(v_item, list):
                    #print(v_item)
                    new_v = []
                    for sub_v in v_item:
                        if sub_v["typeName"] != "BTFSValueArray":
                            new_v.append(round(sub_v['message']['value'], 8))
                        #elif v_item["typeName"] == "BTFSValueArray":BTFSValueWithUnits
                        else:
                            new_v.append([round(x['message']['value'], 8) for x in sub_v["message"]["value"]])
                        v = new_v
                else:
                    if isinstance(v_item, float):
                        v = round(v_item, 8)
                    else:
                        #v = v_item.encode('utf-8')
                        v = v_item
                edge_param.update({k: v})
            edges.append(edge_param)
        return edges

    def expr2meter(self, did, wid, eid, expr):
        """convert value expresson to meter unit"""
        body = {
            "script":
                "function(context is Context, queries) { "
                "   return lookupTableEvaluate(\"%s\") * meter;" % (expr) +
                "}",
            "queries": []
        }

        res = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid +
                                '/featurescript', body=body).json()
        return res['result']['message']['value']

    def eval_curve_midpoint(self, did, wid, eid, geo_id):
        """get the midpoint of a curve specified by its entity ID"""
        body = {
            "script":
                "function(context is Context, queries) { " +
                "   var q_arr = evaluateQuery(context, queries.id);"
                "   var midpoint = evEdgeTangentLine(context, {\"edge\": q_arr[0], \"parameter\": 0.5 }).origin;"
                "   return midpoint;"
                "}",
            "queries": [{"key": "id", "value": [geo_id]}]
        }
        # res = c.get_entity_by_id(did, wid, eid, 'JGV', 'EDGE')
        response = self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript',
                                     body=body)
        point_info = response.json()['result']['message']['value']
        midpoint = [x['message']['value'] for x in point_info]
        return midpoint
