import numpy as np

def add_square(circle):
    square_lines = []
    origin = np.array(circle["param"]["coordSystem"]["origin"])
    normal = np.array(circle["param"]["coordSystem"]["zAxis"])
    radius = np.array(circle["param"]["radius"])
    x_axis = np.zeros(3)

    if np.isclose(normal[2], 0.0):
        x_axis[2] = 1
    elif np.isclose(normal[0], 0.0):
        x_axis[0] = 1
    elif np.isclose(normal[1], 0.0):
        x_axis[1] = 1
    else:
        return []
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    square_lines.append([origin + radius*x_axis + radius*y_axis, origin + radius*x_axis - radius*y_axis])
    square_lines.append([origin - radius*x_axis + radius*y_axis, origin - radius*x_axis - radius*y_axis])
    square_lines.append([origin - radius*x_axis + radius*y_axis, origin + radius*x_axis + radius*y_axis])
    square_lines.append([origin - radius*x_axis - radius*y_axis, origin + radius*x_axis - radius*y_axis])

    return square_lines