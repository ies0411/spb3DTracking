# import numpy as np
# from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
from filterpy.kalman import KalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter
from numba import jit


@jit(nopython=True, cache=True)
def get_bbox_distance(pose):
    return np.sqrt(pose[0] ** 2 + pose[1] ** 2 + pose[2] ** 2) / 200


class Filter(object):
    def __init__(self, bbox3D, ID):
        self.initial_pos = bbox3D[:7]
        self.time_since_update = 0
        self.id = ID
        self.hits = True  # number of total hits including the first detection
        self.confidence = bbox3D[-2]
        self.threshold = bbox3D[-1]
        self.distance = get_bbox_distance(bbox3D[:3])


class KF(Filter):
    def __init__(self, bbox3D, ID):
        super().__init__(bbox3D, ID)

        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        self.kf.F = np.array(
            [
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],  # state transition matrix, dim_x * dim_x
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        # self.kf.R[0:,0:] *= 10.

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000.0
        self.kf.P *= 10.0

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.01

        # initialize data
        self.kf.x[:7] = self.initial_pos.reshape((7, 1))

    def compute_innovation_matrix(self):
        """compute the innovation matrix for association with mahalanobis distance"""
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

    def get_velocity(self):
        # return the object velocity in the state

        return self.kf.x[7:]


#
class UKF(Filter):
    def __init__(self, bbox3D, ID):
        super().__init__(bbox3D, ID)
        self.points = MerweScaledSigmaPoints(13, alpha=0.1, beta=2.0, kappa=-1)
        # TODO : set dt
        self.ukf = UnscentedKalmanFilter(
            dim_x=13, dim_z=7, dt=0.1, fx=self.fx, hx=self.hx, points=self.points
        )
        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        self.ukf.R[0:, 0:] *= np.clip(0.05 * (1 - self.confidence), 0.001, 0.1)
        # 0.01

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.ukf.P[7:, 7:] *= 1000.0
        self.ukf.P *= 10.0

        # process uncertainty, make the constant velocity part more certain
        self.ukf.Q[7:, 7:] *= 0.01

        # initialize data

        self.ukf.x[:7] = self.initial_pos
        # reshape((7, 1))

    def hx(self, x):
        H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        return np.dot(H, x)

    # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz, v, a, theta_rate //CA Model (general , since theta is not correct however x,y,z is more accuracy)
    # state x dimension 10: x, y, z, theta, l, w, h, vx, vy, yz, ax, ay, az //CA Model

    def fx(self, x, dt):
        # [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]
        # displacement = x[10] * dt + x[11] * dt**2 / 2
        # yaw_sin, yaw_cos = np.sin(x[3]), np.cos(x[3])
        F = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt**2],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        return np.dot(F, x)

    # def compute_innovation_matrix(self):
    #     """compute the innovation matrix for association with mahalanobis distance"""
    #     return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

    # def get_velocity(self):
    #     # return the object velocity in the state

    #     return self.kf.x[7:]


# class KF(Filter):
#     def __init__(self, bbox3D, ID):
#         super().__init__(bbox3D, ID)

#         self.kf = KalmanFilter(dim_x=10, dim_z=7)
#         # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

#         # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
#         # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
#         # while all others (theta, l, w, h, dx, dy, dz) remain the same
#         self.kf.F = np.array(
#             # state transition matrix, dim_x * dim_x
#             [
#                 [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                 [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
#                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
#                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#             ]
#         )

#         # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
#         self.kf.H = np.array(
#             [
#                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#             ]
#         )

#         # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
#         self.kf.R[0:, 0:] *= 0.01

#         # initial state uncertainty at time 0
#         # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
#         self.kf.P[7:, 7:] *= 1000.0
#         self.kf.P *= 10.0

#         # process uncertainty, make the constant velocity part more certain
#         self.kf.Q[7:, 7:] *= 0.01

#         # initialize data
#         # self.kf.x[:7] = self.initial_pos.reshape((7, 1))
#         self.kf.x[:7] = self.initial_pos.reshape((7, 1))

#     def compute_innovation_matrix(self):
#         """compute the innovation matrix for association with mahalanobis distance"""
#         return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

#     def get_velocity(self):
#         # return the object velocity in the state

#         return self.kf.x[7:]
