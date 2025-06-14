# import numpy as np
import math


def coordinate_trans(x0, y0, theta, x1, y1):
    """
    param x0，y0: 观察坐标系原点在世界坐标系下的坐标
    param theta: 观察坐标系y轴顺时针旋转到世界坐标系的角
    param x1，y1:观察点在观察坐标系下的坐标
    return: x，y：观察点在世界坐标系下的坐标
    """
    x = x1 * math.cos(theta) - y1 * math.sin(theta) + x0
    y = x1 * math.sin(theta) + y1 * math.cos(theta) + y0
    return x, y


# def p_trans(p, theta, t):
#     """
#     :param p: 观察坐标系坐标
#     :param theta: 旋转角，通过此确定旋转矩阵r
#     :param t: 平移矩阵
#     :return: 世界坐标系坐标
#     """
#     # 旋转矩阵
#     r = np.array([[math.cos(theta), -math.sin(theta)],
#                   [math.sin(theta), math.cos(theta)]])
#     # 世界坐标系坐标值
#     p0 = np.dot(r, p) + t
#     return p0


if __name__ == '__main__':
    x_0 = float(input('Input the x of p1: '))
    y_0 = float(input('Input the y of p1: '))
    theta_0 = math.pi / 6
    # t1 = np.array([[1], [2]])
    print(coordinate_trans(1, 2, theta_0, x_0, y_0))
