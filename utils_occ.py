import numpy as np
import skspatial
from math import acos
import polyscope as ps
from skspatial.objects import Line

from OCC.Core.GeomAPI import GeomAPI_ExtremaCurveCurve, GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnCurve
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.Geom import Geom_Circle, Geom_Line
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import (
    gp_Pnt,
    gp_Vec,
    gp_Pnt2d,
    gp_Lin,
    gp_Dir,
    gp_Ax2,
    gp_Quaternion,
    gp_QuaternionSLerp,
    gp_XYZ,
    gp_Mat,
)

def fit_curve(pnts):

    pts = TColgp_Array1OfPnt(0, len(pnts) - 1)
    for n, i in enumerate(pnts):
        pts.SetValue(n, gp_Pnt(i[0], i[1], i[2]))
    crv = GeomAPI_PointsToBSpline(pts, 1, 30, GeomAbs_C2, 1.0e-8)
    #crv = GeomAPI_PointsToBSpline(pts, 1, 50, GeomAbs_C2, 1.0e-8)
    return crv.Curve()

def compute_dist_curves(c1, c2, return_parameters=False):
    result = GeomAPI_ExtremaCurveCurve(c1, c2)
    dist = result.LowerDistance()
    if return_parameters:
        u1, u2 = result.LowerDistanceParameters()
        return dist, u1, u2
    return dist

def sample_fitted_curve(c, N=10):
    c_pts = []
    for t in np.linspace(0.0, 1.0, N):
        p = gp_Pnt()
        c.D0(t, p)
        c_pts.append([p.Coord()[0], p.Coord()[1], p.Coord()[2]])
    return np.array(c_pts)

def dist_curve_curve(c1, c2):
    #print("intersection_curve_curve")
    result = GeomAPI_ExtremaCurveCurve(c1, c2)
    dist = result.LowerDistance()
    #dist = result.Distance()
    return dist

def intersection_curve_curve(c1, c2, dist_eps=1e-4, VERBOSE=False):
    #print("intersection_curve_curve")
    result = GeomAPI_ExtremaCurveCurve(c1, c2)
    dist = np.min([result.Distance(i+1) for i in range(result.NbExtrema())])
    #if VERBOSE:
    #    print("extremas")
    #    for i in range(result.NbExtrema()):
    #        dist = result.Distance(i+1)
    #        print(dist)
    #    print("lower_dist", result.LowerDistance())

    #    final_parameters = []
    #    u1 = 1.0
    #    c1_p = gp_Pnt()
    #    c1.D0(u1, c1_p)
    #    #print(c1_p.Coord())
    #    result = GeomAPI_ProjectPointOnCurve(c1_p, c2)
    #    #print(result.NbPoints())
    #    if result.NbPoints() > 0:
    #        dist = result.LowerDistance()
    #        print("u1_dist", dist)
    #    u1 = 0.0
    #    c1_p = gp_Pnt()
    #    c1.D0(u1, c1_p)
    #    #print(c1_p.Coord())
    #    result = GeomAPI_ProjectPointOnCurve(c1_p, c2)
    #    if result.NbPoints() > 0:
    #        #print(result.NbPoints())
    #        dist = result.LowerDistance()
    #        print("u1_dist", dist)

    #    u2 = 1.0
    #    c2_p = gp_Pnt()
    #    c2.D0(u2, c2_p)
    #    result = GeomAPI_ProjectPointOnCurve(c2_p, c1)
    #    if result.NbPoints() > 0:
    #        #print(result.NbPoints())
    #        dist = result.LowerDistance()
    #        print("u2_dist", dist)

    #    u2 = 0.0
    #    c2_p = gp_Pnt()
    #    c2.D0(u2, c2_p)
    #    result = GeomAPI_ProjectPointOnCurve(c2_p, c1)
    #    if result.NbPoints() > 0:
    #        dist = result.LowerDistance()
    #        print("u2_dist", dist)


    #dist = result.LowerDistance()
    #print("result.NbExtrema()")
    #print(result.NbExtrema())
    #print("dist", dist)
    #if not np.isclose(dist, 0.0, atol=5e-5):
    if VERBOSE:
        pts_c1 = np.array(sample_fitted_curve(c1, 100))
        pts_c2 = np.array(sample_fitted_curve(c2, 100))
        ps.init()
        ps.register_curve_network("c1", pts_c1, np.array([[i, i+1] for i in range(len(pts_c1)-1)]))
        ps.register_curve_network("c2", pts_c2, np.array([[i, i+1] for i in range(len(pts_c2)-1)]))
        ps.show()
    if not np.isclose(dist, 0.0, atol=dist_eps):
        return False, None, False

    #print(result)
    #print(dist)
    try:
        #u1, u2 = result.LowerDistanceParameters()
        #print(result.TotalLowerDistanceParameters())
        _, u1, u2 = result.TotalLowerDistanceParameters()
    except:
        # check if it's due to an extrema
        final_parameters = []
        u1 = 1.0
        c1_p = gp_Pnt()
        c1.D0(u1, c1_p)
        #print(c1_p.Coord())
        result = GeomAPI_ProjectPointOnCurve(c1_p, c2)
        #print(result.NbPoints())
        dist = result.LowerDistance()
        if np.isclose(dist, 0.0, atol=dist_eps):
            final_parameters = [u1, result.LowerDistanceParameter()]
        u1 = 0.0
        c1_p = gp_Pnt()
        c1.D0(u1, c1_p)
        #print(c1_p.Coord())
        result = GeomAPI_ProjectPointOnCurve(c1_p, c2)
        #print(result.NbPoints())
        dist = result.LowerDistance()
        if np.isclose(dist, 0.0, atol=dist_eps):
            final_parameters = [u1, result.LowerDistanceParameter()]

        #DEBUG
        #c1_pts = []
        #for t in np.linspace(0.0, 1.0, 360):
        #    p = gp_Pnt()
        #    c1.D0(t, p)
        #    c1_pts.append([p.Coord()[0], p.Coord()[1], p.Coord()[2]])
        #c1_pts = np.array(c1_pts)
        #c2_pts = []
        #for t in np.linspace(0.0, 1.0, 360):
        #    p = gp_Pnt()
        #    c2.D0(t, p)
        #    c2_pts.append([p.Coord()[0], p.Coord()[1], p.Coord()[2]])
        #c2_pts = np.array(c2_pts)

        #ps.init()
        #ps.register_curve_network("c1", np.array(c1_pts),
        #                          np.array([[i, i+1] for i in range(len(c1_pts)-1)]))
        #ps.register_curve_network("new_pts", np.array(c2_pts),
        #                          np.array([[i, i+1] for i in range(len(c2_pts)-1)]))
        #ps.show()

        u2 = 1.0
        c2_p = gp_Pnt()
        c2.D0(u2, c2_p)
        result = GeomAPI_ProjectPointOnCurve(c2_p, c1)
        dist = result.LowerDistance()

        if np.isclose(dist, 0.0, atol=dist_eps):
            final_parameters = [result.LowerDistanceParameter(), u2]

        u2 = 0.0
        c2_p = gp_Pnt()
        c2.D0(u2, c2_p)
        result = GeomAPI_ProjectPointOnCurve(c2_p, c1)
        dist = result.LowerDistance()
        if np.isclose(dist, 0.0, atol=dist_eps):
            final_parameters = [result.LowerDistanceParameter(), u2]
        if len(final_parameters) == 0:
            return False, None, False
        else:
            u1 = final_parameters[0]
            u2 = final_parameters[1]

    c1_inter = gp_Pnt()
    c1_tan = gp_Vec()
    c1.D1(u1, c1_inter, c1_tan)
    c2_inter = gp_Pnt()
    c2_tan = gp_Vec()
    c2.D1(u2, c2_inter, c2_tan)

    c1_tan = np.array(c1_tan.Coord())
    c1_tan /= np.linalg.norm(c1_tan)
    c2_tan = np.array(c2_tan.Coord())
    c2_tan /= np.linalg.norm(c2_tan)
    #angle = np.abs(np.rad2deg(acos(np.abs(np.dot(c1_tan.Coord(), c2_tan.Coord())))))
    angle = np.abs(np.rad2deg(acos(np.minimum(np.abs(np.dot(c1_tan, c2_tan)), 1.0))))
    #print(c1_tan)
    #print(c2_tan)
    #print(np.dot(c1_tan, c2_tan))
    #print("angle", angle)

    return True, [c1_inter.Coord()], [angle < 2.0]

if __name__ == "__main__":
    pts = np.array([[-0.80155825, -0.01619669,  0.97688918],
                    [-0.7918093,  -0.01619669,  0.97688919],
                    [-0.77329081, -0.01619669,  0.97688918],
                    [-0.75640815, -0.01619669,  0.97688919],
                    [-0.74079253, -0.0161967,   0.97688918],
                    [-0.73255884, -0.0161967,   0.97688918],
                    [-0.71902668, -0.01619669,  0.97688919],
                    [-0.50871131, -0.01619669,  0.97688918],
                    [-0.47368862, -0.01619669,  0.97688918],
                    [-0.43230802, -0.01619669,  0.97688919],
                    [-0.39053966, -0.01619669,  0.97688918],
                    [-0.33925787, -0.01619668,  0.97688919],
                    [-0.28667306, -0.01619668,  0.97688919],
                    [-0.2324398,  -0.01619671,  0.97688918],
                    [-0.17589351, -0.01619671,  0.97688917],
                    [-0.12483382, -0.01619668,  0.97688919],
                    [-0.07171037, -0.01619671,  0.97688917],
                    [0.02773642, -0.01619668,  0.97688919],
                    [0.12152607, -0.01619672,  0.97688917],
                    [0.20974401, -0.01619668,  0.97688919],
                    [0.29240864, -0.0161967,   0.97688918],
                    [0.33457722, -0.0161967,   0.97688918],
                    [0.37920357, -0.01619669,  0.97688919],
                    [0.45426847, -0.01619668,  0.97688919],
                    [0.52676385, -0.01619667,  0.9768892],
                    [0.59749696, -0.01619673,  0.97688916],
                    [0.66726765, -0.01619669,  0.97688919],
                    [0.73677303, -0.01619669,  0.97688919],
                    [0.80005894, -0.0161967,   0.97688918]])
    l = Line.best_fit(pts)
    dots = np.dot(l.vector, pts.T)
    p0 = np.array(np.min(dots)*l.vector)
    p1 = np.array(np.max(dots)*l.vector)
    new_points = np.array(pts).reshape(-1, 3)
    l = Line.best_fit(new_points)
    dots = np.dot(l.direction, new_points.T)
    is_good_line_fit = np.all([np.isclose(np.linalg.norm(l.project_point(p)-p), 0.0) for p in new_points])
    pts = np.array([new_points[np.argmin(dots)], new_points[np.argmax(dots)]])
    print(dots)
    print(p0)
    print(p1)
    fit_curve(pts)
    #fit_curve(pts)