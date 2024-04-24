# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os
import re
import warnings

import numpy as np
import yaml
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as R

from .nuScenes_split import get_split


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

def check_valid_bbox(bbox_pose_cam_coordinate, intrinsic, width, height):
    img_ptr_norm = np.matmul(intrinsic,bbox_pose_cam_coordinate)
    img_ptr = img_ptr_norm / img_ptr_norm[2][0]
    u,v,z = img_ptr[0][0], img_ptr[1][0], img_ptr[2][0]
    u_invalid = np.logical_or(u<0,u>width)
    v_invalid = np.logical_or(v<0,v>height)
    return np.logical_or(u_invalid, v_invalid), u, v

def convert_to_kittiforamt(dets, group, frame_id, tracking_cfg):
    root_path, save_path, tracking_type = tracking_cfg.dataset_path, tracking_cfg.save_path, tracking_cfg.tracking_type
    intrinsic, extrinsic_lidar2cam = read_calib(os.path.join(root_path,"calib",str(group).zfill(4)+".txt"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,str(group).zfill(4)+".txt")):
        f = open(os.path.join(save_path,str(group).zfill(4)+".txt"), 'w')
    else:
        f = open(os.path.join(save_path,str(group).zfill(4)+".txt"), 'a')
    for det in dets:
        det = np.array(det)
        bbox_pose_lidar_coordinate = det[:3]
        bbox_pose_lidar_coordinate = np.concatenate((bbox_pose_lidar_coordinate, [1])).reshape(4,1)
        bbox_pose_cam_coordinate = np.matmul(extrinsic_lidar2cam, bbox_pose_lidar_coordinate)
        invalid, u,v = check_valid_bbox(bbox_pose_cam_coordinate,intrinsic, tracking_cfg.image_resolution[0], tracking_cfg.image_resolution[1])
        if invalid:
            continue
        #TODO : nms
        bbox_size = det[3:6]
        score = det[7]
        optimal_score = det[8]
        rotation_matrix_lidar_coordinate = np.array([[np.cos(det[6]), -np.sin(det[6]), 0],
                        [np.sin(det[6]), np.cos(det[6]), 0],
                        [0, 0, 1]])
        rotation_matrix_cam_coordinate= np.matmul(extrinsic_lidar2cam[:3,:3], rotation_matrix_lidar_coordinate)
        yaw_cam_coordinate = R.from_matrix(rotation_matrix_cam_coordinate).as_euler("zyx", degrees=False)[0]
        f.write(f"{frame_id} {tracking_type} {u} {v} {bbox_size[0]} {bbox_size[1]} {bbox_size[2]} {bbox_pose_cam_coordinate[0][0]} {bbox_pose_cam_coordinate[1][0]} {bbox_pose_cam_coordinate[2][0]} {yaw_cam_coordinate} {score} {optimal_score}\n")
    f.close()

def load_detection(file):

	# load from raw file
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		dets = np.loadtxt(file, delimiter=',') 	# load detections, N x 15

	if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0)
	if dets.shape[1] == 0:		# if no detection in a sequence
		return [], False
	else:
		return dets, True

def get_subfolder_seq(dataset, split):

	# dataset setting
	file_path = os.path.dirname(os.path.realpath(__file__))
	if dataset == 'KITTI':				# KITTI
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}

		if split == 'val': subfolder = 'training'
		elif split == 'test': subfolder = 'testing'
		else: assert False, 'error'

		hw = {'image': (375, 1242), 'lidar': (720, 1920)}

		if split == 'train': seq_eval = ['0000', '0002', '0003', '0004', '0005', '0007', '0009', '0011', '0017', '0020']         # train
		if split == 'val':   seq_eval = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']    # val
		if split == 'test':  seq_eval  = ['%04d' % i for i in range(29)]

		data_root = os.path.join(file_path, '../data/KITTI') 		# path containing the KITTI root

	elif dataset == 'nuScenes':			# nuScenes
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Bicycle', 4: 'Motorcycle', 5: 'Bus', \
			6: 'Trailer', 7: 'Truck', 8: 'Construction_vehicle', 9: 'Barrier', 10: 'Traffic_cone'}

		subfolder = split
		hw = {'image': (900, 1600), 'lidar': (720, 1920)}

		if split == 'train': seq_eval = get_split()[0]		# 700 scenes
		if split == 'val':   seq_eval = get_split()[1]		# 150 scenes
		if split == 'test':  seq_eval = get_split()[2]      # 150 scenes

		data_root = os.path.join(file_path, '../data/nuScenes/nuKITTI') 	# path containing the nuScenes-converted KITTI root

	else: assert False, 'error, %s dataset is not supported' % dataset

	return subfolder, det_id2str, hw, seq_eval, data_root

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def Config(filename):
    listfile = open(filename, "r")
    cfg = edict(yaml.safe_load(listfile))
    listfile.close()

    return cfg


def get_subfolder_seq(dataset, split):

    # dataset setting
    file_path = os.path.dirname(os.path.realpath(__file__))
    if dataset == "KITTI":  # KITTI
        det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}

        if split == "val":
            subfolder = "training"
        elif split == "test":
            subfolder = "testing"
        else:
            assert False, "error"

        hw = {"image": (375, 1242), "lidar": (720, 1920)}

        if split == "train":
            seq_eval = [
                "0000",
                "0002",
                "0003",
                "0004",
                "0005",
                "0007",
                "0009",
                "0011",
                "0017",
                "0020",
            ]  # train
        if split == "val":
            seq_eval = [
                "0001",
                "0006",
                "0008",
                "0010",
                "0012",
                "0013",
                "0014",
                "0015",
                "0016",
                "0018",
                "0019",
            ]  # val
        if split == "test":
            seq_eval = ["%04d" % i for i in range(29)]

        data_root = os.path.join(
            file_path, "../data/KITTI"
        )  # path containing the KITTI root

    elif dataset == "nuScenes":  # nuScenes
        det_id2str = {
            1: "Pedestrian",
            2: "Car",
            3: "Bicycle",
            4: "Motorcycle",
            5: "Bus",
            6: "Trailer",
            7: "Truck",
            8: "Construction_vehicle",
            9: "Barrier",
            10: "Traffic_cone",
        }

        subfolder = split
        hw = {"image": (900, 1600), "lidar": (720, 1920)}

        if split == "train":
            seq_eval = get_split()[0]  # 700 scenes
        if split == "val":
            seq_eval = get_split()[1]  # 150 scenes
        if split == "test":
            seq_eval = get_split()[2]  # 150 scenes

        data_root = os.path.join(
            file_path, "../data/nuScenes/nuKITTI"
        )  # path containing the nuScenes-converted KITTI root

    else:
        assert False, "error, %s dataset is not supported" % dataset

    return subfolder, det_id2str, hw, seq_eval, data_root


def get_threshold(dataset, det_name):
    # used for visualization only as we want to remove some false positives, also can be
    # used for KITTI 2D MOT evaluation which uses a single operating point
    # obtained by observing the threshold achieving the highest MOTA on the validation set

    if dataset == "KITTI":
        if det_name == "pointrcnn":
            return {"Car": 3.240738, "Pedestrian": 2.683133, "Cyclist": 3.645319}
        else:
            assert False, (
                "error, detection method not supported for getting threshold" % det_name
            )
    elif dataset == "nuScenes":
        if det_name == "megvii":
            return {
                "Car": 0.262545,
                "Pedestrian": 0.217600,
                "Truck": 0.294967,
                "Trailer": 0.292775,
                "Bus": 0.440060,
                "Motorcycle": 0.314693,
                "Bicycle": 0.284720,
            }
        if det_name == "centerpoint":
            return {
                "Car": 0.269231,
                "Pedestrian": 0.410000,
                "Truck": 0.300000,
                "Trailer": 0.372632,
                "Bus": 0.430000,
                "Motorcycle": 0.368667,
                "Bicycle": 0.394146,
            }
        else:
            assert False, (
                "error, detection method not supported for getting threshold" % det_name
            )
    else:
        assert False, "error, dataset %s not supported for getting threshold" % dataset


# def initialize(cfg, cat, ID_start):
#     # initialize the tracker and provide all path of data needed

#     # image_dir = os.path.join(cfg.data_root, "image_02")

#     # load image for visualization
#     # initiate the tracker


#     # compute the min/max frame
#     # frame_list, _ = load_list_from_folder(img_seq)
#     # frame_list = [fileparts(frame_file)[1] for frame_file in frame_list]

#     return tracker, frame_list


def find_all_frames(root_dir, subset, data_suffix, seq_list):
    # warm up to find union of all frames from results of all categories in all sequences
    # finding the union is important because there might be some sequences with only cars while
    # some other sequences only have pedestrians, so we may miss some results if mainly looking
    # at one single category
    # return a dictionary with each key correspondes to the list of frame ID

    # loop through every sequence
    frame_dict = dict()
    for seq_tmp in seq_list:
        frame_all = list()

        # find all frame indexes for each category
        for subset_tmp in subset:
            data_dir = os.path.join(
                root_dir, subset_tmp, "trk_withid" + data_suffix, seq_tmp
            )  # pointrcnn_ped
            # if not is_path_exists(data_dir):
            #     print("%s dir not exist" % data_dir)
            #     assert False, "error"

            # extract frame string from this category
            frame_list, _ = load_list_from_folder(data_dir)
            frame_list = [fileparts(frame_tmp)[1] for frame_tmp in frame_list]
            frame_all.append(frame_list)

        # merge frame indexes from all categories
        frame_all = merge_listoflist(frame_all, unique=True)
        frame_dict[seq_tmp] = frame_all

    return frame_dict

