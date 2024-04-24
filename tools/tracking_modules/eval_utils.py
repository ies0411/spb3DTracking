
import os
import re

import cv2
import numpy as np


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
  vtc_mat = np.matmul(R0, vtc_mat)

  return P2, vtc_mat



def convertformat_det_to_track(det,frame):
  # print()
  #lidar coor, dx,dy,dz,x,y,z,yaw,id
  calib_path = os.path.join('/mnt/nas3/Data/kitti-processed/object_tracking/training/calib',str(frame).zfill(4)+"txt")
  intrinsic, extrinsic = read_calib(calib_path)
  