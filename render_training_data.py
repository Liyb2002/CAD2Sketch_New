import json
from camera import Camera
import polyscope as ps
from math import atan
from trimesh.primitives import Sphere, Box
import matplotlib.pyplot as plt

from xvfbwrapper import Xvfb
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import os
import igl
import imageio
import numpy as np
#from utils import spherical_to_cartesian_coords

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

def look_at(eye, center, up):
    modelview_mat = np.zeros([4, 4])
    F = center - eye
    f = F/np.linalg.norm(F)
    up_prim = up/np.linalg.norm(up)
    s = np.cross(f, up_prim)
    u = np.cross(s/np.linalg.norm(s), f)
    modelview_mat[0, :3] = s
    modelview_mat[1, :3] = u
    modelview_mat[2, :3] = -f
    modelview_mat[3, 3] = 1
    return modelview_mat

def perspective(fov, aspect, near, far):
    f = np.arctan(fov/2.0)
    proj_mat = np.zeros([4, 4])
    proj_mat[0, 0] = f/aspect
    proj_mat[1, 1] = f
    proj_mat[2, 2] = (far+near)/(near-far)
    proj_mat[2, 3] = 2*far*near/(near-far)
    proj_mat[3, 2] = -1
    return proj_mat


def xvfb_exists():
    """Check that Xvfb is available on PATH and is executable."""
    paths = os.environ['PATH'].split(os.pathsep)
    return any(os.access(os.path.join(path, 'Xvfb'), os.X_OK)
               for path in paths)

def project_lines_opengl(lines, display, cam_pos, center, up_vec):
    if xvfb_exists():
        vdisplay = Xvfb()
        vdisplay.start()
    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    #up_vec = np.array([0.0, 0.0, 1.0])
    #up_vec = np.array([0.0, 0.0, -1.0])
    eye = cam_pos + center
    #center = np.zeros(3)
    #p = [-0.125, 0.250, 0.750]
    #print(cam_pos)
    #proj_mat = perspective(45.0, (display[0] / display[1]), 0.001, 2.0)
    #print(proj_mat)
    #model_mat = look_at(eye, center, up_vec)
    #print(model_mat)
    gluPerspective(45.0, (display[0] / display[1]), 0.001, 10.0)
    #print(center)
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up_vec[0], up_vec[1], up_vec[2])
#    proj_mat = glGetDoublev(GL_PROJECTION_MATRIX)
#    print(proj_mat)
#    model_mat = glGetDoublev(GL_MODELVIEW_MATRIX)
#    print(model_mat)
#    view = glGetIntegerv(GL_VIEWPORT)
#    print(view)
    #win_coord = gluProject(objX=p[0], objY=p[1], objZ=p[2], model=model_mat, proj=proj_mat, view=view)
    projected_lines = []
    for line in lines:
        proj_line = []
        for p in line:
            win_coord = gluProject(objX=p[0], objY=p[1], objZ=p[2])
            #proj_line.append([win_coord[0], display[1] - win_coord[1]])
            proj_line.append([win_coord[0], win_coord[1]])
        projected_lines.append(proj_line)
    #print(win_coord)
    #print(proj_mat)
    #glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    pygame.quit()
    if xvfb_exists():
        vdisplay.stop()
    return projected_lines

def render_lines(line_vertices, line_edges):
    glLineWidth(3.0)
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_LINES)
    for edge in line_edges:
        for v_id in edge:
            glVertex3fv(line_vertices[v_id])
    glEnd()

def render_mesh(v, n, f):
    glBegin(GL_TRIANGLES)
    for f_id, face in enumerate(f):
        #color = n[f_id]
        #glColor3f(color[0], color[1], color[2])
        for v_id, vertex in enumerate(face):
            color = n[vertex]
            glColor3f(color[0], color[1], color[2])
            #glColor3f(1.0, 0.0, 1.0)
            glVertex3fv(v[vertex])
    glEnd()

def prep_rendering():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glClearColor(0.5, 0.5, 0.5, 1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    #glShadeModel(GL_SMOOTH)
    #glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ZERO)
    #glEnable(GL_LINE_SMOOTH)
    #glEnable(GL_POLYGON_SMOOTH)
    #glEnable(GL_MULTISAMPLE)

def get_normal_map_single_mesh(patches, display, cam_pos, center, up_vec):
    if xvfb_exists():
        vdisplay = Xvfb()
        vdisplay.start()
    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    #center = np.array([-4.5667941522421733,2.8447793496222289,3.6736773995095557])
    #up_vec = np.array([0.0, 0.0, 1.0])
    #up_vec = np.array([-0.012760008001773696,0.99933159166544194,0.034257146629705651])
    eye = cam_pos + center
    #eye = np.array([-5.3985948958037646,2.8151575987354316,4.2279607881157677])
    #f = 3.0140139059542448
    #fov = np.rad2deg(2.0 * atan(1.0/f))
    fov = 45.0
    gluPerspective(fov, (display[0] / display[1]), 0.001, 10.0)
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up_vec[0], up_vec[1], up_vec[2])
    #glFrustum(-2, 2, -2, 2, 0.001, 10.0)
    look_at_vec = center - eye
    look_at_vec /= np.linalg.norm(look_at_vec)
    left_vec = np.cross(up_vec, look_at_vec)
    left_vec /= np.linalg.norm(left_vec)
    #glMatrixMode(GL_MODELVIEW)
    #mat = glGetFloatv(GL_MODELVIEW_MATRIX)
    #print(mat)
    #exit()
    #mv = np.array([[0.5549276576331178,-0.012760008001773696,-0.83180074356159106,0],
    #               [-0.021422459565022114,0.99933159166544194,-0.029621750886797273,0],
    #               [0.831622734790245,0.034257146629705651,0.55428338860621185,0],
    #               [-0.45993109333219095,-3.026999910665261,-6.7506537839515648,1]])
    ##mv = np.array([[0.40419759525719856,0.23932061863545481,-0.88280798902379742,0],
    ##               [0.069268443412078329,0.95438314121657675,0.29043881026620327,0],
    ##               [0.91204505740427422,-0.17854540391038665,0.36918200390445804,0],
    ##               [-1.286469312002209,-0.74683879985950774,-6.8618384241154819,1]])
    #glLoadMatrixf(mv.T)
    normal_pixels = get_normal_map(patches, display, look_at_vec, up_vec, left_vec, cam_pos+center)
    pygame.quit()
    if xvfb_exists():
        vdisplay.stop()
    return normal_pixels

def get_normal_map(patches, display, look_at_vec, up_vec, left_vec, eye):
    patches_data = []
    #mat = glGetFloatv(GL_MODELVIEW_MATRIX)
    #inv_mat = np.linalg.inv(mat).T[:3, :3]
    #inv_mat = mat[:3, :3]
    #inv_mat = np.identity(4)
    #print(inv_mat)
    for patch in patches.values():
        if (not type(patch) == dict) or (not "faces" in patch.keys()):
            patch_vertices = np.array([v for f in patch for v in f])
            dists = squareform(pdist(patch_vertices))
            adj_mat = np.isclose(dists, 0.0)
            vertex_clusters = [list(c) for c in nx.connected_components(nx.from_numpy_matrix(adj_mat))]
            new_patch_vertices = np.array([patch_vertices[c[0]] for c in vertex_clusters])
            faces = np.arange(0, 3*len(patch))
            for c_id, c in enumerate(vertex_clusters):
                for v_id in c:
                    faces[v_id] = c_id
            faces = faces.reshape(-1, 3)
        else:
            new_patch_vertices = patch["vertices"]
            faces = patch["faces"]

        left_vec = np.cross(up_vec, look_at_vec)
        left_vec /= np.linalg.norm(left_vec)
        up_vec = np.cross(look_at_vec, left_vec)
        e = eye
        d = look_at_vec
        d /= np.linalg.norm(d)
        r = np.cross(d, up_vec)
        r /= np.linalg.norm(r)
        u = np.cross(r, d)
        u /= np.linalg.norm(u)
        mat_r = np.array([
            [r[0], r[1], r[2], 0],
            [u[0], u[1], u[2], 0],
            [-d[0], -d[1], -d[2], 0],
            [0, 0, 0, 1]
        ])
        mat_t = np.array([
            [1, 0, 0, -e[0]],
            [0, 1, 0, -e[1]],
            [0, 0, 1, -e[2]],
            [0, 0, 0, 1],
        ])
        mat_m = np.matmul(mat_r, mat_t)
        n = igl.per_vertex_normals(new_patch_vertices, faces)

        hom_n = np.zeros([len(n), 4])
        hom_n[:, :3] = n
        n = np.dot(mat_m, hom_n.T).T
        #n = np.array([np.dot(mat_m, p) for p in hom_n])
        #print(hom_n.shape)
        #n /= n[:, -1].reshape(-1, 1)
        #n[:, :3] /= n[:, -1].reshape(-1, 1)
        n = n[:, :3]
        #n /= np.linalg.norm(n, axis=-1).reshape(-1, 1)
        #n = hom_n[:, :3]
        n = np.vstack([n[:, 0], n[:, 1], n[:, 2]]).T
        #print(n.shape)

#        print(look_at_vec)
#        print(left_vec)
#        print(up_vec)
#        v_to_cam_vecs = cam_pos - new_patch_vertices
#        v_to_cam_vecs /= np.linalg.norm(v_to_cam_vecs, axis=-1).reshape(-1, 1)
#        n = n-v_to_cam_vecs
#        n /= np.linalg.norm(n, axis=-1).reshape(-1, 1)
#        x = np.dot(look_at_vec, n.T)
#        print(x)
#        y = np.dot(left_vec, n.T)
#        z = np.dot(up_vec, n.T)
#        #n = np.vstack([-y, -z, -x]).T
#        n = np.vstack([y, z, x]).T
#        n /= np.linalg.norm(n, axis=-1).reshape(-1, 1)
#        print(n)

#        glMatrixMode(GL_MODELVIEW)
#        modelview_mat = glGetDoublev(GL_MODELVIEW_MATRIX)
#        print(modelview_mat)
#        #modelview_mat = modelview_mat[:3, :3]
#
#        #up_vec = np.array([0, 0, 1])
#        #model_view = np.zeros([3, 3])
#        #model_view[:, 0] = left_vec
#        #model_view[:, 1] = up_vec
#        #model_view[:, 2] = look_at_vec
#        #print(model_view)
#        #exit()
#        normal_mat = np.linalg.inv(modelview_mat).T
#        normal_mat = normal_mat[:3, :3]
#        print(normal_mat)
#        #n = np.dot(normal_mat, n.T).T
#        n = np.dot(n, normal_mat)
#        print(n)
#        n /= np.linalg.norm(n, axis=-1).reshape(-1, 1)
#        print(n)
#        #exit()
        #print("normals")
        #print(n)
        #print(np.min(n[:, 0]), np.max(n[:, 0]))
        #n[:, 1] = -n[:, 1]
        #print(np.min(n[:, 1]), np.max(n[:, 1]))
        #n[:, 2] += 1.0
        #n[:, 2] /= 2.0
        #print(np.min(n[:, 2]), np.max(n[:, 2]))
        #exit()

        n += 1.0
        n /= 2.0
        #n *= 0.5
        #n += 0.5
        #n /= np.linalg.norm(n, axis=-1).reshape(-1, 1)
        #n /= np.linalg.norm(n, axis=-1).reshape(-1, 1)
        #n = igl.per_vertex_normals(new_patch_vertices, faces)

        #glClearColor(1.0, 1.0, 1.0, 1.0)
        render_mesh(new_patch_vertices, n, faces)
        patches_data.append([new_patch_vertices, n, faces])

    # normal map rendering
    prep_rendering()
    for v, n, f in patches_data:
        render_mesh(v, n, f)
    normal_pixels = glReadPixels(0, 0, display[0], display[1], format=GL_RGB, type=GL_FLOAT)
    #normal_pixels = glReadPixels(0, 0, display[0], display[1], format=GL_RGBA, type=GL_FLOAT)
    #normal_pixels = np.array(normal_pixels, dtype=np.uint8)
    #print(normal_pixels.dtype)
    return normal_pixels

def convert_feature_lines(feature_lines):
    feature_lines_vertices = []
    feature_lines_edges = []
    for line in feature_lines:
        for p_id in range(len(line["geometry"])-1):
            feature_lines_edges.append([len(feature_lines_vertices)+p_id, len(feature_lines_vertices)+p_id+1])
        for p in line["geometry"]:
            feature_lines_vertices.append(p)
    return feature_lines_vertices, feature_lines_edges

from scipy import ndimage, misc
def get_silhouette_points_2(mesh_depth, display):
    sobel_result = ndimage.sobel(mesh_depth)
    #print(sobel_result.shape)
    #print(sobel_result>0.02)
    #plt.imshow(sobel_result)
    #plt.show()
    return np.abs(sobel_result) > 0.01
    print(sobel_result.dtype)
    print(np.max(sobel_result))
    print(np.min(sobel_result))
    exit()
    #silhouette_points = np.zeros([display[0], display[1]], dtype=bool)
    #eps = 1e-2
    #for i in range(1, display[0]-1):
    #    for j in range(1, display[1]-1):
    #        gap = np.max([np.abs(mesh_depth[i, j]-mesh_depth[i-1, j]),
    #                      np.abs(mesh_depth[i, j]-mesh_depth[i-1, j+1]),
    #                      np.abs(mesh_depth[i, j]-mesh_depth[i-1, j-1]),
    #                      np.abs(mesh_depth[i, j]-mesh_depth[i, j+1]),
    #                      np.abs(mesh_depth[i, j]-mesh_depth[i, j-1]),
    #                      np.abs(mesh_depth[i, j]-mesh_depth[i+1, j-1]),
    #                      np.abs(mesh_depth[i, j]-mesh_depth[i+1, j]),
    #                      np.abs(mesh_depth[i, j]-mesh_depth[i+1, j+1])])
    #        if gap > eps:
    #            silhouette_points[i, j] = True
    #return silhouette_points

def get_silhouette_points(mesh_depth, display):
    silhouette_points = np.zeros([display[0], display[1]], dtype=bool)
    eps = 1e-2
    for i in range(1, display[0]-1):
        for j in range(1, display[1]-1):
            gap = np.max([np.abs(mesh_depth[i, j]-mesh_depth[i-1, j]),
                          np.abs(mesh_depth[i, j]-mesh_depth[i-1, j+1]),
                          np.abs(mesh_depth[i, j]-mesh_depth[i-1, j-1]),
                          np.abs(mesh_depth[i, j]-mesh_depth[i, j+1]),
                          np.abs(mesh_depth[i, j]-mesh_depth[i, j-1]),
                          np.abs(mesh_depth[i, j]-mesh_depth[i+1, j-1]),
                          np.abs(mesh_depth[i, j]-mesh_depth[i+1, j]),
                          np.abs(mesh_depth[i, j]-mesh_depth[i+1, j+1])])
            if gap > eps:
                silhouette_points[i, j] = True
    return silhouette_points

def get_per_line_visibility_scores(mesh, lines, display, cam_pos, center, up_vec):
    if xvfb_exists():
        vdisplay = Xvfb()
        vdisplay.start()
    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    eye = cam_pos + center
    fov = 45.0
    gluPerspective(fov, (display[0] / display[1]), 0.001, 10.0)
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up_vec[0], up_vec[1], up_vec[2])
    look_at_vec = center - eye
    look_at_vec /= np.linalg.norm(look_at_vec)
    left_vec = np.cross(up_vec, look_at_vec)
    left_vec /= np.linalg.norm(left_vec)
    #normal_pixels = get_normal_map(patches, display, look_at_vec, up_vec, left_vec, cam_pos+center)
    prep_rendering()
    render_mesh(mesh.vertices, [], mesh.faces)
    mesh_depth = glReadPixels(0, 0, display[0], display[1], GL_DEPTH_COMPONENT, type=GL_FLOAT)
    imageio.imsave(os.path.join("mesh_depth.png"), mesh_depth)
    for line_id, line in enumerate(lines):
        feature_lines_vertices, feature_lines_edges = convert_feature_lines(lines)
        prep_rendering()
        render_lines(feature_lines_vertices, feature_lines_edges)
        line_depth = glReadPixels(0, 0, display[0], display[1], GL_DEPTH_COMPONENT, type=GL_FLOAT)
        imageio.imsave(os.path.join("line_depth.png"), line_depth)
        print(line_depth)
        mask_visibile = np.logical_or(np.abs(mesh_depth-line_depth) < 1e-4,
            line_depth < mesh_depth)
        mask_hidden = np.logical_and(mesh_depth < line_depth,
            mesh_depth < 1.0)
        print(line_id, np.sum(mask_visibile)/np.sum(mask_hidden))
    pygame.quit()
    if xvfb_exists():
        vdisplay.stop()
    #return normal_pixels

def render_baseline_data(display, data_folder, render_folder, last_obj_id, visibility_check=True,
                         lambda_0=-1, prefix=""):
    if xvfb_exists():
        vdisplay = Xvfb()
        vdisplay.start()
    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    mesh_file = os.path.join(data_folder, "shape_"+str(last_obj_id)+".obj")
    global_v, _, global_n, global_f, _, _ = igl.read_obj(mesh_file)
    #global_v[:] *= 10.0
    bbox_v, _ = igl.bounding_box(global_v)
    bbox_diag = igl.bounding_box_diagonal(global_v)
    center = np.mean(bbox_v, axis=0)
    with open(os.path.join(data_folder, "feature_faces_"+str(last_obj_id)+".json"), "r") as fp:
        patches = json.load(fp)
    if lambda_0 == -1:
        with open(os.path.join(data_folder, "feature_lines_"+str(last_obj_id)+".json"), "r") as fp:
            feature_lines = json.load(fp)
    else:
        with open(os.path.join(data_folder, "decluttered_lambda0_"+str(lambda_0)+".json"), "r") as fp:
            feature_lines = json.load(fp)

    #gluLookAt(0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    eye = center + np.ones(3)*bbox_diag

    img_counter = 0
    #for theta in range(5, 85, 10):
    for theta in np.linspace(115, 175, 1):
        for phi in np.linspace(5, 175, 1):
            for radius in np.linspace(2.0, 1.5, 1):
                for fov in [45]:
                    r = radius*bbox_diag
                    cam_coords = spherical_to_cartesian_coords(r, np.deg2rad(theta), np.deg2rad(phi))
                    #gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], 0.0, 0.0, -1.0)
                    eye = center + cam_coords
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity()

                    gluPerspective(fov, (display[0] / display[1]), 0.001, 2.0)
                    up_vec = np.array([0.0, 0.0, 1.0])
                    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up_vec[0], up_vec[1], up_vec[2])

                    look_at_vec = center - eye
                    look_at_vec /= np.linalg.norm(look_at_vec)
                    left_vec = np.cross(up_vec, look_at_vec)
                    left_vec /= np.linalg.norm(left_vec)
                    normal_pixels = get_normal_map(patches, display, look_at_vec, up_vec, left_vec)
                    #imageio.imsave(os.path.join(render_folder, "normal_map.png"), normal_pixels)

                    # feature-line rendering
                    # create visibility-check mask
                    prep_rendering()
                    render_mesh(global_v, global_n, global_f)
                    mesh_depth = glReadPixels(0, 0, display[0], display[1], GL_DEPTH_COMPONENT, type=GL_FLOAT)
                    #imageio.imsave(os.path.join(render_folder, "mesh_depth.png"), mesh_depth)
                    feature_lines_vertices, feature_lines_edges = convert_feature_lines(feature_lines)
                    prep_rendering()
                    render_lines(feature_lines_vertices, feature_lines_edges)
                    line_depth = glReadPixels(0, 0, display[0], display[1], GL_DEPTH_COMPONENT, type=GL_FLOAT)
                    #imageio.imsave(os.path.join(render_folder, "line_depth.png"), line_depth)
                    mask = np.logical_and(np.abs(mesh_depth-line_depth) > 1e-4,
                                          mesh_depth < line_depth)
                    silhouette_points = get_silhouette_points_2(mesh_depth, display)
                    prep_rendering()
                    render_lines(feature_lines_vertices, feature_lines_edges)
                    pixels = glReadPixels(0, 0, display[0], display[1], format=GL_RGB, type=GL_FLOAT)
                    #imageio.imsave(os.path.join(render_folder, "feature_lines.png"), np.array(np.invert(np.array(pixels, dtype=bool)), dtype=float))
                    if visibility_check:
                        pixels[mask] = [0.0, 0.0, 0.0]
                    #imageio.imsave(os.path.join(render_folder, "feature_lines_visible.png"), np.array(np.invert(np.array(pixels, dtype=bool)), dtype=float))
                    pixels[silhouette_points] = [1.0, 1.0, 1.0]
                    #imageio.imsave(os.path.join(render_folder, "feature_lines_silhouette.png"), np.array(np.invert(np.array(pixels, dtype=bool)), dtype=float))
                    pixels = np.array(np.invert(np.array(pixels, dtype=bool)), dtype=float)
                    tmp_file_name = str(np.char.zfill(str(img_counter), 5))
                    #imageio.imsave(os.path.join(render_folder, prefix+"_"+tmp_file_name+".png"),
                    #               imageio.core.image_as_uint(np.hstack((pixels, normal_pixels))))
                    imageio.imsave(os.path.join(render_folder, prefix+"_"+tmp_file_name+".png"),
                                   pixels)
                    img_counter += 1
    pygame.quit()
    if xvfb_exists():
        vdisplay.stop()

def render_construction_line_data(display):
    if xvfb_exists():
        vdisplay = Xvfb()
        vdisplay.start()
    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    data_folder = os.path.join("data", "24")
    mesh_file = os.path.join(data_folder, "shape_5.obj")
    global_v, _, global_n, global_f, _, _ = igl.read_obj(mesh_file)
    bbox_v, _ = igl.bounding_box(global_v)
    bbox_diag = igl.bounding_box_diagonal(global_v)
    center = np.mean(bbox_v, axis=0)
    with open(os.path.join(data_folder, "feature_faces_5.json"), "r") as fp:
        patches = json.load(fp)
    with open(os.path.join(data_folder, "decluttered_lambda0_4.json"), "r") as fp:
        feature_lines = json.load(fp)

    gluPerspective(45, (display[0]/display[1]), 0.001, 2.0)
    eye = center + np.ones(3)*bbox_diag
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], 0.0, 0.0, -1.0)

    normal_pixels = get_normal_map(patches, display)
    #imageio.imsave(os.path.join(data_folder, "normal_map_construction.png"), normal_pixels)

    # feature-line rendering
    feature_lines_vertices, feature_lines_edges = convert_feature_lines(feature_lines)
    prep_rendering()
    render_lines(feature_lines_vertices, feature_lines_edges)
    pixels = glReadPixels(0, 0, display[0], display[1], GL_RGB, type=GL_FLOAT)
    # add silhouette points
    prep_rendering()
    render_mesh(global_v, global_n, global_f)
    mesh_depth = glReadPixels(0, 0, display[0], display[1], GL_DEPTH_COMPONENT, type=GL_FLOAT)
    silhouette_points = get_silhouette_points(mesh_depth, display)
    pixels[silhouette_points] = [1.0, 1.0, 1.0]
    pixels = np.array(np.invert(np.array(pixels, dtype=bool)), dtype=float)
    imageio.imsave(os.path.join(data_folder, "rendered_construction_line_data.png"), np.hstack((pixels, normal_pixels)))
    pygame.quit()
    if xvfb_exists():
        vdisplay.stop()

def lift_point(p, lambda_val, k_inv, r_t_inv):
    u, v = p[0], p[1]

    p_cam = np.dot(k_inv, np.array([[u], [v], [1.0]]))
    p_cam *= lambda_val
    p_cam = np.expand_dims(p_cam, 0)

    p_world = np.ones(shape=(4, 1))
    p_world[:3] = p_cam
    p_world = np.dot(r_t_inv, p_world)
    p_world[:] /= p_world[3]
    p_world = p_world[:3]
    p_world = np.transpose(p_world)
    return p_world[0]


def compute_inverse_matrices(r_t, k):

    r_t_inv = np.ones(shape=(4, 4))
    r_t_inv[:3, :] = r_t
    r_t_inv[3, :3] = 0.0
    r_t_inv = np.linalg.inv(r_t_inv)
    k_inv = np.linalg.inv(k)
    return r_t_inv, k_inv

if __name__ == "__main__":
    #data_folder = os.path.join("data", "24")
    #display = (800, 800)
    #render_baseline_data(display, data_folder, 5)
    #render_construction_line_data(display)
    import trimesh
    import cv2
    import polyscope as ps
    from scipy.spatial.transform import Rotation
    correspondences_file_name = os.path.join("correspondanciesJSON/professional2_vacuum_cleaner_v1_points.json")
    correspondences_file_name = os.path.join("correspondanciesJSON/student8_house_v1_points.json")
    with open(correspondences_file_name, "r") as fp:
        corr = json.loads(fp.read())
    corr_2d = np.array(corr["points_2D_sketch"], dtype=np.float64)
    #corr_2d -= width/2
    #corr_2d *= 2/width
    corr_3d = np.array(corr["points_3D_object"], dtype=np.float64)
    print(corr_2d)
    print(corr_3d)
    cam_mat = np.zeros([3, 3])
    cam_mat[0, 0] = 1.0
    cam_mat[1, 1] = 1.0
    cam_mat[0, 2] = 972/2
    cam_mat[1, 2] = 972/2
    cam_mat[2, 2] = 1.0
    dist_coeffs = np.zeros([4, 1])
    #corr_3d = corr_3d[:, [0, 2, -1]]
    retval, rvec, tvec = cv2.solvePnP(objectPoints=corr_3d, imagePoints=corr_2d,
                                      cameraMatrix=cam_mat, distCoeffs=dist_coeffs)
    print(retval)
    print(rvec)
    print(tvec)
    rot_mat = cv2.Rodrigues(rvec)[0].transpose()
    #rot_mat = cv2.Rodrigues(rvec)[0]
    print(rot_mat)
    t_vec = np.array(tvec).reshape(-1, 3)
    print(t_vec)
    cam_pos = np.dot(-rot_mat, t_vec.T).T

    #cam_pos = np.dot(rot_mat, t_vec.T).T
    print("cam_pos", cam_pos)
    with open(os.path.join("../../Downloads/objects_obj_files/house.obj"), "r") as fp:
        mesh = trimesh.load(fp, file_type="obj")
    cam_file_name = os.path.join("camera_parameters/Professional2_vacuum_cleaner_view1_camparam.json")
    cam_file_name = os.path.join("camera_parameters/student8_house_view1_camparam.json")
    with open(cam_file_name, "r") as fp:
        cam_params = json.loads(fp.read())
    width = cam_params["width"]
    cam_params = cam_params["restricted"]
    cam_pos = np.array(cam_params["C"])
    K = np.array([[cam_params["f"], 0, cam_params["u"], 0],
                  [0, cam_params["f"], cam_params["v"], 0],
                  [0, 0, 1, 0]])
    print("Here")
    print(0.05*(972/2))

    K = np.array([[cam_params["f"], 0, cam_params["u"]+0.02097902516369585, 0],
                  [0, cam_params["f"], cam_params["v"]-0.8384695532590567, 0],
                  [0, 0, 1, 0]])
    K = np.array([[cam_params["f"], 0, cam_params["u"], 0],
                  [0, cam_params["f"], cam_params["v"], 0],
                  [0, 0, 1, 0]])
    K = np.array([[cam_params["f"], 0, cam_params["u"]+0.00017229428312506514, 0],
                  [0, cam_params["f"], cam_params["v"]-0.8530708036658408, 0],
                  [0, 0, 1, 0]])
    #K = np.array([[724.11907256293341,0,168.38543531004632],[0,724.11907256293341,-81.530723168877273],[0,0,1]])
    plt.gca().set_aspect("equal")
    pp = np.array([cam_params["u"], cam_params["v"]])
    rt = np.array(cam_params["mvMatrix"]).T
    conv_mat = np.identity(4)
    conv_mat[1, 1] = -1
    conv_mat[2, 2] = -1
    rt = np.matmul(conv_mat, rt)
    P = np.matmul(K, rt)
    print(P)
    cam = Camera(proj_mat=P)

    projected_focal_point = np.array(cam.project_point(cam_params["focal_point"]))
    print("projected_focal_point")
    print(projected_focal_point)
    projected_focal_point += 1.0
    projected_focal_point /= 2.0
    projected_focal_point *= 972
    plt.scatter([projected_focal_point[0]], [projected_focal_point[1]], c="green")
    projected_pts = np.array([cam.project_point(p) for p in corr_3d])
    #projected_pts[:, 0] *= -1
    #projected_pts += pp
    projected_pts += 1.0
    projected_pts /= 2.0
    projected_pts *= 972
    print(projected_pts)
    translation_vec = corr_2d[0] - projected_pts[0]
    a_mat = np.zeros([2*len(corr_2d), 2])
    b_mat = np.zeros(2*len(corr_2d))
    for i in range(len(corr_2d)):
        a_mat[2*i, 0] = 1
        a_mat[2*i+1, 1] = 1
        b_mat[2*i] = corr_2d[i][0] - projected_pts[i][0]
        b_mat[2*i+1] = corr_2d[i][1] - projected_pts[i][1]
    translation_vec, _, _, _ = np.linalg.lstsq(a_mat, b_mat, rcond=None)

    print(translation_vec)
    print(2*(translation_vec[0]/972))
    print(2*(translation_vec[1]/972))
    #exit()

    #print(projected_pts)
    before_pts = np.array([[ 792.69599169,  971.37917125],
 [ 812.67728507,  868.98755977],
 [ 800.77836386,  832.58524763],
 [ 757.95520034,  785.13702819],
 [ 681.89907323,  700.864676  ],
 [ 532.35903367, 1041.64071418],
 [ 782.53270337,  815.35081049],
 [ 424.12423732,  447.55619077],
 [ 153.89783922,  547.88786973],
 [ 659.5374998,   530.26193222],
 [ 363.52312878,  636.15941052],
 [ 171.30521361,  633.12441468],
 [ 171.46355968,  867.37337821],
 [ 481.57629591,  498.32370203],
 [ 464.94698626,  695.19419332],
 [ 100.89293778,  814.54362798],
 [ 404.17996993,  655.7259295 ],
 [ 372.00379724,  957.23962845],
 [ 644.46789395,  757.1020473 ]])
    after_pts = np.array([[ 768.39599169,  947.07917125],
 [ 788.37728507,  844.68755977],
 [ 776.47836386,  808.28524763],
 [ 733.65520034,  760.83702819],
 [ 657.59907323,  676.564676  ],
 [ 508.05903367, 1017.34071418],
 [ 758.23270337,  791.05081049],
 [ 399.82423732,  423.25619077],
 [ 129.59783922,  523.58786973],
 [ 635.2374998,   505.96193222],
 [ 339.22312878,  611.85941052],
 [ 147.00521361,  608.82441468],
 [ 147.16355968,  843.07337821],
 [ 457.27629591,  474.02370203],
 [ 440.64698626,  670.89419332],
 [  76.59293778,  790.24362798],
 [ 379.87996993,  631.4259295 ],
 [ 347.70379724,  932.93962845],
 [ 620.16789395,  732.8020473 ]])
    print(before_pts-after_pts)

    plt.scatter(corr_2d[:, 0], corr_2d[:, 1], c="blue"),
    plt.scatter(projected_pts[:, 0], projected_pts[:, 1], c="r")

    for p_id in range(len(corr_2d)):
        plt.plot([corr_2d[p_id][0], projected_pts[p_id][0]],
                 [corr_2d[p_id][1], projected_pts[p_id][1]])
    plt.gca().invert_yaxis()
    plt.show()
    exit()

    rt[:3, :3] = Rotation.from_matrix(np.identity(3)).as_matrix()
    print(rt)
    print(K)
    corr_2d /= 972
    corr_2d *= 2.0
    corr_2d -= 1.0
    print(corr_2d)



    r_t_inv, k_inv = compute_inverse_matrices(rt[:3, :], K[:, :3])
    lifted_points_3d = np.array([lift_point(p, 40.0, k_inv, r_t_inv) for p in corr_2d])
    lifted_points_3d_vec = lifted_points_3d - cam_pos
    lifted_points_3d_vec /= np.linalg.norm(lifted_points_3d_vec, axis=-1).reshape(-1, 1)

    corr_3d_vec = corr_3d - cam_pos
    corr_3d_vec /= np.linalg.norm(corr_3d_vec, axis=-1).reshape(-1, 1)
    best_rot_mat = Rotation.align_vectors(corr_3d_vec, lifted_points_3d_vec)[0]
    print(best_rot_mat)

    rotated_corr_vecs = best_rot_mat.apply(lifted_points_3d_vec)
    print(rt)
    #rt[:3, :3] = (best_rot_mat*Rotation.from_matrix(rt[:3, :3])).as_matrix()
    rt[:3, :3] = best_rot_mat.as_matrix()
    r_t_inv, k_inv = compute_inverse_matrices(rt[:3, :], K[:, :3])
    # get the correct depth per point
    relifted_points_3d = np.array([lift_point(p, 40.0, k_inv, r_t_inv) for p in corr_2d])
    relifted_points_3d_vec = relifted_points_3d - cam_pos
    relifted_points_3d_vec /= np.linalg.norm(relifted_points_3d_vec, axis=-1).reshape(-1, 1)

    ps.init()
    ps.register_surface_mesh("house", mesh.vertices, mesh.faces)
    ps.register_point_cloud("coords_3d", corr_3d)
    ps.register_point_cloud("cam", np.array([cam_pos]))
    ps.register_point_cloud("focal", np.array([[-4.3272834781443787,3.2533572214795483,-4.1014842896681953]]))
    #ps.register_point_cloud("cam", np.array([[-5.2080756422992192,3.61694255580905,-4.40481722263779]]))
    ps.register_point_cloud("lifted_points", lifted_points_3d)
    ps.register_point_cloud("lifted_points_vec", cam_pos+lifted_points_3d_vec)
    ps.register_point_cloud("corr_3d_vec", cam_pos+corr_3d_vec)
    ps.register_point_cloud("rotated_corr_3d_vec", cam_pos+rotated_corr_vecs)
    ps.register_point_cloud("relifted_points", cam_pos+relifted_points_3d_vec)
    ps.show()

    pygame.init()
    pygame.display.set_mode((972, 972), DOUBLEBUF | OPENGL)
    mat = glGetFloatv(GL_MODELVIEW_MATRIX)
    print("mat")
    print(mat)

    gluPerspective(45, 1, 1.0, 10000.0)
    mat = glGetFloatv(GL_MODELVIEW_MATRIX)
    print("mat")
    print(mat)

    gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2], 0.0, 0.0, 0.0, cam_params["up"][0], cam_params["up"][1], cam_params["up"][2])
    prep_rendering()
    #glLoadMatrixf(np.array(cam_params["mvMatrix"]))
    mat = glGetFloatv(GL_MODELVIEW_MATRIX)
    print("mat")
    print(mat)
    glMatrixMode(GL_MODELVIEW)
    #rt[:, 3] = mat[:, 3]
    #rt[3, :] = mat[3, :]
    glLoadMatrixf(np.array(rt))
    print(rt)
    render_mesh(mesh.vertices, np.repeat([1.0, 0.0, 0.0], len(mesh.faces)), mesh.faces)
    pygame.display.flip()
    pygame.time.wait(1000)
    pygame.quit()
    exit()

    lifted_points_3d = np.array([lift_point(p, 1.0, k_inv, r_t_inv) for p in corr_2d])
    import cv2
    dist_coeffs = np.zeros([4, 1])
    #retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectpoints=[corr_3d], imagepoints=[corr_2d], imagesize=[972, 972], cameramatrix=k, distcoeffs=none,
    #                                                                     flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_FOCAL_LENGTH)
    print(corr_2d)
    print(corr_3d)
    cam_file_name = os.path.join("camera_parameters/student7_vacuum_cleaner_view1_camparam.json")
    with open(cam_file_name, "r") as fp:
        cam_params = json.loads(fp.read())
    width = cam_params["width"]
    cam_params = cam_params["restricted"]
    cam_pos = np.array(cam_params["C"])
    print(cam_params)
    t = np.array([0.30716593699723493,0.5053782684627891,1])
    R = np.array([[-0.70639254874283552,0.70614133294794312,0.048723556757515232],[0.034458884182849939,-0.03444662950095035,0.998812302195411],[0.706981012744666,0.7072325272627995,1.7347234759768071E-17]])
    RT = np.zeros([3, 4])
    RT[0, 0] = 1
    RT[1, 1] = 1
    RT[2, 2] = 1
    RT[:, 3] = cam_pos
    #RT[:3, :3] = R
    #RT[:3, 3] = t
    print(RT)

    #K = np.array([[cam_params["f"], 0, width/2],
    #              [0, cam_params["f"], width/2],
    #              [0, 0, 1]])
    cam_params["f"] = 0.3
    K = np.array([[cam_params["f"], 0, 0],
                  [0, cam_params["f"], 0],
                  [0, 0, 1]])
    print(K)
    #K = np.array([[724.11907256293341,0,168.38543531004632],[0,724.11907256293341,-81.530723168877273],[0,0,1]])
    r_t_inv, k_inv = compute_inverse_matrices(RT, K)

    correspondences_file_name = os.path.join("correspondanciesJSON/student8_house_v1_points.json")
    with open(correspondences_file_name, "r") as fp:
        corr = json.loads(fp.read())
    corr_2d = np.array(corr["points_2D_sketch"], dtype=np.float64)
    #corr_2d -= width/2
    #corr_2d *= 2/width
    corr_3d = np.array(corr["points_3D_object"], dtype=np.float64)
    lifted_points_3d = np.array([lift_point(p, 1.0, k_inv, r_t_inv) for p in corr_2d])
    import cv2
    cam_mat = np.zeros([3, 3])
    dist_coeffs = np.zeros([4, 1])
    #retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectpoints=[corr_3d], imagepoints=[corr_2d], imagesize=[972, 972], cameramatrix=k, distcoeffs=none,
    #                                                                     flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_FOCAL_LENGTH)
    print(corr_2d)
    print(corr_3d)
    retval, rvec, tvec = cv2.solvePnP(objectPoints=corr_3d[:7], imagePoints=corr_2d[:7],
                                      cameraMatrix=cam_mat, distCoeffs=dist_coeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    print(retval)
    print(rvec)
    print(tvec)
    exit()
    #print(retval)
    #print(cameraMatrix)
    #print(rvecs)
    #print(tvecs)

    for p_id in range(len(lifted_points_3d)):
        v = lifted_points_3d[p_id] - cam_pos
        v /= np.linalg.norm(v)
        lifted_points_3d[p_id] = v + cam_pos
    for p_id in range(len(corr_3d)):
        v = corr_3d[p_id] - cam_pos
        v /= np.linalg.norm(v)
        corr_3d[p_id] = v + cam_pos
    print(corr)

    print(corr_2d)
    print(lifted_points_3d)
    mesh_file = os.path.join("objects_obj_files/vacuum_cleaner.obj")
    global_v, _, global_n, global_f, _, _ = igl.read_obj(mesh_file)
    ps.init()
    ps.register_surface_mesh("mesh", global_v, global_f)
    ps.register_point_cloud("corr_3d", corr_3d)
    #ps.register_point_cloud("corr_2d", corr_2d)
    ps.register_point_cloud("lifted_3d", lifted_points_3d)
    ps.register_point_cloud("cam_pos", np.array([cam_pos]))
    ps.show()
    # TODO: custom renderer with rotated camera
    exit()
    #s = Sphere(subdivsions=100)
    s = Box()
    print(s.triangles)
    #normal_pixels = get_normal_map_single_mesh({"1": s.triangles}, (800, 800), np.array([2, 2, -1]), np.zeros(3))
    #triangles = [[global_v[f[0]], global_v[f[1]], global_v[f[2]]] for f in global_f]
    triangles = []
    for f_id, f in enumerate(global_f):
        print(f_id)
        triangles.append([global_v[f[0]], global_v[f[1]], global_v[f[2]]])
    #exit()
    #print("here")
    #print(triangles)
    #exit()
    normal_pixels = get_normal_map_single_mesh(
        {"1": {"triangles": triangles, "faces": global_f, "vertices": global_v, "normals": global_n}},
        (800, 800), np.array([2, 2, -1]), np.zeros(3))
    plt.imshow(normal_pixels)
    plt.gca().invert_yaxis()
    plt.show()
    plt.gcf().subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                              bottom=0.0,
                              top=1.0)
    plt.gca().imshow(normal_pixels)
    plt.axis("off")
    plt.gcf().set_size_inches(800/100, 800/100)
    plt.gca().invert_yaxis()
    normal_map_name = "test.png"
    plt.savefig(normal_map_name, dpi=100)
    plt.close(plt.gcf())
