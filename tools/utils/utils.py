import re
import numpy as np

"""
input: calib txt path
return: P2: (4,4) 3D camera coordinates to 2D image pixels
        vtc_mat: (4,4) 3D velodyne Lidar coordinates to 3D camera coordinates
"""


def read_calib(calib_path):
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)
                P2 = P2.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)
                R0 = R0.reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    # vtc_mat = np.matmul(R0, vtc_mat)
    return (P2, vtc_mat)


"""
description: read lidar data given
input: lidar bin path "path", cam 3D to cam 2D image matrix (4,4), lidar 3D to cam 3D matrix (4,4)
output: valid points in lidar coordinates (PointsNum,4)
"""


def read_velodyne(path, P, vtc_mat, IfReduce=True):
    max_row = 374  # y
    max_col = 1241  # x
    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    if not IfReduce:
        return lidar

    mask = lidar[:, 0] > 0
    lidar = lidar[mask]
    lidar_copy = np.zeros(shape=lidar.shape)
    lidar_copy[:, :] = lidar[:, :]

    velo_tocam = vtc_mat
    lidar[:, 3] = 1
    lidar = np.matmul(lidar, velo_tocam.T)
    img_pts = np.matmul(lidar, P.T)
    velo_tocam = np.mat(velo_tocam).I
    velo_tocam = np.array(velo_tocam)
    normal = velo_tocam
    normal = normal[0:3, 0:4]
    lidar = np.matmul(lidar, normal.T)
    lidar_copy[:, 0:3] = lidar
    x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
    mask = np.logical_and(
        np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row)
    )

    return lidar_copy[mask]


"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""


def cam_to_velo(cloud, vtc_mat):
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.mat(mat)
    normal = np.mat(vtc_mat).I
    normal = normal[0:3, 0:4]
    transformed_mat = normal * mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    return T


"""
description: convert Lidar 3D coordinates to 3D camera coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""


def vel_to_cam_pose(box, vtc_mat):
    box.append(1)
    # print(f"box : {box}")
    mat = np.array(box)
    # mat = np.mat(mat)
    # normal = np.mat(vtc_mat)
    # normal = normal[0:3, 0:4]
    # print(box)
    # print(vtc_mat)
    transformed_mat = vtc_mat @ mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    # print(T)
    return T


def velo_to_cam(cloud, vtc_mat):
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.mat(mat)
    normal = np.mat(vtc_mat)
    normal = normal[0:3, 0:4]
    transformed_mat = normal * mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    return T


def corners3d_to_img_boxes(P2, corners3d):
    """
    :param corners3d: (N, 8, 3) corners in rect coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
    """
    sample_num = corners3d.shape[0]
    corners3d_hom = np.concatenate(
        (corners3d, np.ones((sample_num, 8, 1))), axis=2
    )  # (N, 8, 4)

    img_pts = np.matmul(corners3d_hom, P2.T)  # (N, 8, 3)

    x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
    x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
    x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

    img_boxes = np.concatenate(
        (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)),
        axis=1,
    )
    boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, 1242 - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, 375 - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, 1242 - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, 375 - 1)

    return img_boxes, boxes_corner


def bb3d_2_bb2d(bb3d, P2):

    x, y, z, l, w, h, yaw = (
        bb3d[0],
        bb3d[1],
        bb3d[2],
        bb3d[3],
        bb3d[4],
        bb3d[5],
        bb3d[6],
    )

    pt1 = [l / 2, 0, w / 2, 1]
    pt2 = [l / 2, 0, -w / 2, 1]
    pt3 = [-l / 2, 0, w / 2, 1]
    pt4 = [-l / 2, 0, -w / 2, 1]
    pt5 = [l / 2, -h, w / 2, 1]
    pt6 = [l / 2, -h, -w / 2, 1]
    pt7 = [-l / 2, -h, w / 2, 1]
    pt8 = [-l / 2, -h, -w / 2, 1]
    pts = np.array([[pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]])
    transpose = np.array(
        [
            [np.cos(np.pi - yaw), 0, -np.sin(np.pi - yaw), x],
            [0, 1, 0, y],
            [np.sin(np.pi - yaw), 0, np.cos(np.pi - yaw), z],
            [0, 0, 0, 1],
        ]
    )
    pts = np.matmul(pts, transpose.T)
    box, _ = corners3d_to_img_boxes(P2, pts[:, :, 0:3])

    return box
