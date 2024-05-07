import numpy as np
from numba import jit


@jit
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@jit
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@jit
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@jit
def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


@jit
def _poses_from_oxts(oxts_packets):
    """Helper method to compute SE(3) pose matrices from OXTS packets."""
    # https://github.com/pratikac/kitti/blob/master/pykitti/raw.py

    er = 6378137.0  # earth radius (approx.) in meters

    # compute scale from first lat value
    scale = np.cos(oxts_packets[0].lat * np.pi / 180.0)

    t_0 = []  # initial position
    poses = []  # list of poses computed from oxts
    for packet in oxts_packets:
        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + packet.lat) * np.pi / 360.0))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # We want the initial position to be the origin, but keep the ENU
        # coordinate system
        if len(t_0) == 0:
            t_0 = t

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        poses.append(
            transform_from_rot_trans(R, t - t_0)
        )  # store transformation matrix

    return np.stack(poses)


class Box3D:
    def __init__(
        self,
        x=None,
        y=None,
        z=None,
        h=None,
        w=None,
        l=None,
        ry=None,
        s=None,
        thres=None,
    ):
        self.x = x  # center x
        self.y = y  # center y
        self.z = z  # center z
        self.h = h  # height
        self.w = w  # width
        self.l = l  # length
        self.ry = ry  # orientation
        self.s = s  # detection score
        self.corners_3d_cam = None
        self.thres = thres

    def __str__(self):
        return "x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}, thres: {}".format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s, self.thres
        )

    @classmethod
    def bbox2dict(cls, bbox):
        return {
            "center_x": bbox.x,
            "center_y": bbox.y,
            "center_z": bbox.z,
            "height": bbox.h,
            "width": bbox.w,
            "length": bbox.l,
            "heading": bbox.ry,
        }

    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array(
                [
                    bbox.x,
                    bbox.y,
                    bbox.z,
                    bbox.ry,
                    bbox.l,
                    bbox.w,
                    bbox.h,
                    bbox.s,
                    bbox.thres,
                ]
            )

    @classmethod
    def bbox2array_raw(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array(
                [bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s]
            )

    @classmethod
    def array2bbox_raw(cls, data):
        # take the format of data of [h,w,l,x,y,z,theta]

        bbox = Box3D()
        bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def pcdet2bbox_raw(cls, data):
        # take the format of data of [h,w,l,x,y,z,theta]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.h, bbox.w, bbox.l, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        elif len(data) == 9:
            bbox.s = data[-2]
            bbox.thres = data[-1]
        return bbox

    @classmethod
    def array2bbox(cls, data):
        # take the format of data of [x,y,z,theta,l,w,h]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def box2corners3d_camcoord(cls, bbox):
        """Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and
        convert it to the 8 corners of the 3D box, the box is in the camera coordinate
        with right x, down y, front z

        Returns:
            corners_3d: (8,3) array in in rect camera coord

        box corner order is like follows
                1 -------- 0         top is bottom because y direction is negative
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7

        rect/ref camera coord:
        right x, down y, front z

        x -> w, z -> l, y -> h
        """

        # if already computed before, then skip it
        if bbox.corners_3d_cam is not None:
            return bbox.corners_3d_cam

        # compute rotational matrix around yaw axis
        # -1.57 means straight, so there is a rotation here
        R = roty(bbox.ry)

        # 3d bounding box dimensions
        l, w, h = bbox.l, bbox.w, bbox.h

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + bbox.x
        corners_3d[1, :] = corners_3d[1, :] + bbox.y
        corners_3d[2, :] = corners_3d[2, :] + bbox.z
        corners_3d = np.transpose(corners_3d)
        bbox.corners_3d_cam = corners_3d

        return corners_3d
