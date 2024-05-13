import numpy as np

import utils
from utils import print_pretty
import polyscope as ps
import circle_3d
import trimesh
from skspatial.objects import Plane, Line, Point, Points
from trimesh.intersections import mesh_plane

class PerspectiveBBox:
    def __init__(self, x_min=0, x_max=0, y_min=0, y_max=0):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

class SketchPoint:
    def __init__(self, id=None, point=None, segment_ids=[]):
        self.id = id
        self.point = point
        self.segment_ids = segment_ids

class Quad:
    def __init__(self):
        self.corner_points = None
        self.plane_normal = None
        self.mesh = None

    def intersection_plane(self, plane):
        line_segs = mesh_plane(self.mesh, plane.normal, plane.point)
        if len(line_segs) == 0:
            return line_segs
        if len(line_segs) == 1:
            return line_segs[0]
        #dir_vec = line_segs[0][1] - line_segs[0][0]
        #dir_vec /= np.linalg.norm(dir_vec)
        points = line_segs.reshape(-1, 3)
        #dots = np.dot(dir_vec, points.T)
        #new_line = np.array([points[np.argmin(dots)], points[np.argmax(dots)]])
        new_line, _ = utils.line_segment_from_points(points)
        #print("comp")
        #print(new_line)
        #print(test_line)
        return new_line


class Segment:
    def __init__(self):
        self.endpoints = []
        self.line = None

class PerspectiveGrid:
    def __init__(self):
        self.sketch = None
        self.sketch_points = {}
        self.circle_points = {}
        self.sketch_faces = {}
        self.segments = {}
        self.quads = []
        self.x_axis = None
        self.y_axis = None
        self.plane_normal = None
        self.plane_origin = None
        self.bbox = None

    def parse_sketch(self, sketch, curves):
        #print_pretty(sketch)
        self.sketch = sketch
        #if sketch["name"] == "Sketch 3":
        #    print(sketch)
        #    print(sketch["profiles"])
        sketch_profiles = sketch["profiles"]
        real_geometry_endpoints = np.array([c[i] for i in [0, -1] for c in curves if len(c) > 1])
        for vert_id in sketch_profiles["vertices"].keys():
            vert = sketch_profiles["vertices"][vert_id]
            dists = np.linalg.norm(np.array(vert["param"]["Vector"]) - real_geometry_endpoints, axis=-1)
            if np.isclose(np.min(dists), 0.0, atol=1e-4):
                self.sketch_points[vert_id] = SketchPoint(id=vert_id, point=np.array(vert["param"]["Vector"]), segment_ids=[])
        #vert_id = 0
        #for c in curves:
        #    self.sketch_points[str(vert_id)] = SketchPoint(id=str(vert_id), point=np.array(c[0]), segment_ids=[])
        #    vert_id += 1
        #    self.sketch_points[str(vert_id)] = SketchPoint(id=str(vert_id), point=np.array(c[-1]), segment_ids=[])
        #    vert_id += 1


        #if sketch["name"] == "Sketch 3":
        #    for edge_id in sketch_profiles["edges"].keys():
        #        print(sketch_profiles["edges"][edge_id])
        #    for vert_id in sketch_profiles["vertices"].keys():
        #        print(vert_id)

            #exit()
        for edge_id in sketch_profiles["edges"].keys():
            edge = sketch_profiles["edges"][edge_id]
            # TODO: filter out spline control points
            pt_id = 0
            if edge["param"]["type"] == "Circle":
                circle_lines = circle_3d.add_square(edge)
                circle_points = []
                for l in circle_lines:
                    circle_points.append(l[0])
                    self.sketch_points[edge_id+"_"+str(pt_id)] = SketchPoint(id=edge_id+"_"+str(pt_id),
                                                                             point=l[0], segment_ids=[])
                    pt_id += 1
                    circle_points.append(l[1])
                    self.sketch_points[edge_id + "_" + str(pt_id)] = SketchPoint(id=edge_id + "_" + str(pt_id),
                                                                                 point=l[1], segment_ids=[])
                    pt_id += 1
                self.circle_points[edge_id] = np.array(circle_points)
        self.x_axis = np.array([sketch["transform"]["x_axis"]["x"],
                                sketch["transform"]["x_axis"]["y"],
                                sketch["transform"]["x_axis"]["z"]])
        self.y_axis = np.array([sketch["transform"]["y_axis"]["x"],
                                sketch["transform"]["y_axis"]["y"],
                                sketch["transform"]["y_axis"]["z"]])
        self.plane_normal = np.array([sketch["transform"]["z_axis"]["x"],
                                      sketch["transform"]["z_axis"]["y"],
                                      sketch["transform"]["z_axis"]["z"]])
        self.plane_origin = np.array([sketch["transform"]["origin"]["x"],
                                      sketch["transform"]["origin"]["y"],
                                      sketch["transform"]["origin"]["z"]])

        points = np.array([sketch_pt.point for sketch_pt in self.sketch_points.values()])
        for circle_points in self.circle_points.values():
            points = np.concatenate([points, circle_points])
        x_proj = np.dot(points, self.x_axis)
        y_proj = np.dot(points, self.y_axis)
        self.bbox = PerspectiveBBox(x_min=np.min(x_proj),
                                    x_max=np.max(x_proj),
                                    y_min=np.min(y_proj),
                                    y_max=np.max(y_proj))
        self.bbox_points = np.array([self.bbox.x_min*self.x_axis + self.bbox.y_min*self.y_axis + np.dot(self.plane_origin, self.plane_normal)*self.plane_normal,
                                     self.bbox.x_min*self.x_axis + self.bbox.y_max*self.y_axis + np.dot(self.plane_origin, self.plane_normal)*self.plane_normal,
                                     self.bbox.x_max*self.x_axis + self.bbox.y_min*self.y_axis + np.dot(self.plane_origin, self.plane_normal)*self.plane_normal,
                                     self.bbox.x_max*self.x_axis + self.bbox.y_max*self.y_axis + np.dot(self.plane_origin, self.plane_normal)*self.plane_normal])
        new_quad = Quad()
        new_quad.plane_normal = self.plane_normal
        new_quad.corner_points = [self.bbox_points[0], self.bbox_points[1], self.bbox_points[3], self.bbox_points[2]]
        new_quad.mesh = trimesh.Trimesh(vertices=new_quad.corner_points, faces=[[0, 1, 2], [0, 2, 3]])
        self.quads.append(new_quad)

        seg = Segment()
        seg.endpoints = np.array([self.bbox_points[0], self.bbox_points[1]])
        seg.line = Line.from_points(self.bbox_points[0], self.bbox_points[1])
        self.segments["bbox_0"] = seg
        seg = Segment()
        seg.endpoints = np.array([self.bbox_points[0], self.bbox_points[2]])
        seg.line = Line.from_points(self.bbox_points[0], self.bbox_points[2])
        self.segments["bbox_1"] = seg
        seg = Segment()
        seg.endpoints = np.array([self.bbox_points[2], self.bbox_points[3]])
        seg.line = Line.from_points(self.bbox_points[2], self.bbox_points[3])
        self.segments["bbox_2"] = seg
        seg = Segment()
        seg.endpoints = np.array([self.bbox_points[1], self.bbox_points[3]])
        seg.line = Line.from_points(self.bbox_points[1], self.bbox_points[3])
        self.segments["bbox_3"] = seg

        #ps.init()
        #ps.register_curve_network("bbox_0", nodes=np.array([self.bbox_points[0], self.bbox_points[1]]),
        #                          edges=np.array([[0, 1]]))
        #ps.register_curve_network("bbox_1", nodes=np.array([self.bbox_points[0], self.bbox_points[2]]),
        #                          edges=np.array([[0, 1]]))
        #ps.register_curve_network("bbox_2", nodes=np.array([self.bbox_points[2], self.bbox_points[3]]),
        #                          edges=np.array([[0, 1]]))
        #ps.register_curve_network("bbox_3", nodes=np.array([self.bbox_points[1], self.bbox_points[3]]),
        #                          edges=np.array([[0, 1]]))
        #ps.register_point_cloud("points", points)

        # create segments
        #if len(self.sketch_points.values()) > 100:
        #    raise Exception("Too many sketch elements!")
        for sketch_pt in self.sketch_points.values():
            # 2 segments per point
            l_x = Line(point=sketch_pt.point, direction=self.x_axis)
            x_intersections = []
            l_y = Line(point=sketch_pt.point, direction=self.y_axis)
            y_intersections = []
            # intersect with bbox_segs
            for seg_id in self.segments.keys():
                if not "bbox" in seg_id:
                    continue
                seg = self.segments[seg_id]
                #tmp_pts = points([sketch_pt.point, seg.endpoints[0], seg.endpoints[1]])
                #print("affine_rank", tmp_pts.affine_rank())
                #print(seg.line.point, seg.line.to_point())
                #print(l_x.point, l_x.to_point())
                #tmp_pts = Points([l_x.point, l_x.to_point(), seg.line.point, seg.line.to_point()])
                #print("affine_rank", tmp_pts.affine_rank(tol=1e-7))
                if not seg.line.direction.is_parallel(l_x.direction):
                    #inter = seg.line.intersect_line(l_x)
                    inter = utils.intersect_lines(seg.line, l_x)
                    x_intersections.append(inter)
                if not seg.line.direction.is_parallel(l_y.direction):
                    #inter = seg.line.intersect_line(l_y)
                    inter = utils.intersect_lines(seg.line, l_y)
                    y_intersections.append(inter)
            seg_x = Segment()
            seg_x.endpoints = np.array(x_intersections)
            seg_x.line = Line.from_points(x_intersections[0], x_intersections[1])
            self.segments[sketch_pt.id+"_x"] = seg_x
            seg_y = Segment()
            seg_y.endpoints = np.array(y_intersections)
            seg_y.line = Line.from_points(y_intersections[0], y_intersections[1])
            self.segments[sketch_pt.id+"_y"] = seg_y

        # assign segments to sketch_points
        for sketch_pt_id in self.sketch_points.keys():
            sketch_pt = self.sketch_points[sketch_pt_id]
            for seg_id in self.segments.keys():
                dist = np.linalg.norm(sketch_pt.point - self.segments[seg_id].line.project_point(Point(sketch_pt.point)))
                if np.isclose(dist, 0.0):
                    self.sketch_points[sketch_pt_id].segment_ids.append(seg_id)

        # trim down segments
        replace_ids = []
        for vec_id_one, seg_id in enumerate(sorted(self.segments.keys())):
            line = self.segments[seg_id].line
            for vec_id_two, other_seg_id in enumerate(sorted(self.segments.keys())):
                if vec_id_two <= vec_id_one:
                    continue
                if seg_id == other_seg_id:
                    continue
                other_line = self.segments[other_seg_id].line
                if line.direction.is_parallel(other_line.direction) and np.isclose(line.distance_line(other_line), 0.0):
                    replace_ids.append(other_seg_id)
                    for sketch_pt_id in self.sketch_points.keys():
                        self.sketch_points[sketch_pt_id].segment_ids = np.where(np.array(self.sketch_points[sketch_pt_id].segment_ids) == other_seg_id,
                                                                                seg_id, np.array(self.sketch_points[sketch_pt_id].segment_ids))
        replace_ids = np.unique(replace_ids)
        for repl_id in replace_ids:
            self.segments.pop(repl_id)

        for sketch_pt_id in self.sketch_points.keys():
            self.sketch_points[sketch_pt_id].segment_ids = np.unique(self.sketch_points[sketch_pt_id].segment_ids)

    def plot_grid(self):
        points = np.array([sketch_pt.point for sketch_pt in self.sketch_points.values()])
        ps.register_point_cloud(self.sketch["name"], points)
        for circle_id in self.circle_points.keys():
            ps.register_point_cloud(self.sketch["name"]+"_"+circle_id, self.circle_points[circle_id])
        ps.register_point_cloud(self.sketch["name"]+"_bbox", self.bbox_points)
        for seg_id in self.segments.keys():
            ps.register_curve_network(self.sketch["name"]+"_"+seg_id, nodes=self.segments[seg_id].endpoints,
                                      edges=np.array([[0, 1]]))
        for quad_id, quad in enumerate(self.quads):
            ps.register_surface_mesh(self.sketch["name"]+"_"+str(quad_id), vertices=quad.mesh.vertices,
                                     faces=quad.mesh.faces)

    def get_grid_lines(self):
        lines = []
        for seg in self.segments.values():
            lines.append(seg.endpoints)
        return lines

    def extrude_grids(self, vertices_ids, extrude_normal, extrude_depth_one, extrude_depth_two):
        extrude_lines = []
        new_quads = []
        already_extended_seg_ids = set()
        # collect new vertices_ids
        new_vertices_ids = []
        for vert_id in vertices_ids:
            if vert_id in self.sketch_points.keys():
                new_vertices_ids.append(vert_id)
            else:
                for circle_vert_id in self.sketch_points.keys():
                    if vert_id+"_" in circle_vert_id:
                        new_vertices_ids.append(circle_vert_id)

        for vert_id in new_vertices_ids:
            if not vert_id in self.sketch_points.keys():
                continue
            seg_ids = self.sketch_points[vert_id].segment_ids
            # intersect with old quads
            for seg_id in seg_ids:
                if seg_id in already_extended_seg_ids:
                    continue
                if not seg_id in self.segments.keys():
                    continue
                already_extended_seg_ids.add(seg_id)
                seg = self.segments[seg_id]
                new_plane_normal = np.cross(extrude_normal, seg.line.direction)
                new_plane_normal /= np.linalg.norm(new_plane_normal)
                new_plane = Plane(point=seg.endpoints[0], normal=new_plane_normal)
                for quad_id in range(len(self.quads)):
                    new_line = self.quads[quad_id].intersection_plane(new_plane)
                    #for new_line in self.quads[quad_id].intersection_plane(new_plane):
                    if len(new_line) > 0:
                        extrude_lines.append(new_line)
                # form new quad
                new_quad = Quad()
                new_quad.plane_normal = new_plane_normal
                new_quad.corner_points = [seg.endpoints[0] + extrude_depth_two*extrude_normal,
                                          seg.endpoints[0] + extrude_depth_one * extrude_normal,
                                          seg.endpoints[1] + extrude_depth_one * extrude_normal,
                                          seg.endpoints[1] + extrude_depth_two * extrude_normal]
                new_quad.mesh = trimesh.Trimesh(vertices=new_quad.corner_points, faces=[[0, 1, 2], [0, 2, 3]])
                #extrude_lines += [[new_quad.corner_points[0], new_quad.corner_points[1]],
                #                  [new_quad.corner_points[1], new_quad.corner_points[2]],
                #                  [new_quad.corner_points[2], new_quad.corner_points[3]],
                #                  [new_quad.corner_points[3], new_quad.corner_points[0]]]
                extrude_lines += [
                    [new_quad.corner_points[3], new_quad.corner_points[0]],
                    [new_quad.corner_points[0], new_quad.corner_points[1]],
                    [new_quad.corner_points[2], new_quad.corner_points[3]],
                    [new_quad.corner_points[1], new_quad.corner_points[2]],
                ]
                new_quads.append(new_quad)
        self.quads += new_quads
        return extrude_lines

    def extrude_vertices(self, vertices_ids, extrude_normal, extrude_depth_one, extrude_depth_two):
        extrude_lines = []
        #print("vertices_ids", vertices_ids)
        # collect new vertices_ids
        new_vertices_ids = []
        for vert_id in np.unique(vertices_ids):
            if vert_id in self.sketch_points.keys():
                new_vertices_ids.append(vert_id)
            else:
                for circle_vert_id in self.sketch_points.keys():
                    if vert_id+"_" in circle_vert_id:
                        new_vertices_ids.append(circle_vert_id)

        for vert_id in new_vertices_ids:
            if not vert_id in self.sketch_points.keys():
                continue
            vert_pt = self.sketch_points[vert_id].point
            extrude_lines.append([vert_pt+extrude_depth_two*extrude_normal,
                                  vert_pt+extrude_depth_one*extrude_normal])
        return extrude_lines

