import argparse
import glob
import time
from pathlib import Path
import os
import natsort

import torch
import argparse
import time

import copy
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tracking_modules.model import Spb3DMOT
from tracking_modules.utils import Config
from utils.utils import read_calib, bb3d_2_bb2d, velo_to_cam, vel_to_cam_pose


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="./cfgs/custom_models/pv_rcnn_plusplus_kitti.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../sample/lidar",
        help="specify the point cloud data file or directory",
    )
    # /mnt/nas3/Data/kitti-processed/object_tracking/training/velodyne/
    parser.add_argument(
        "--ckpt",
        default="/mnt/nas2/users/eslim/result_log/generalization/pvrcnnplus_anchor_230930_r/ckpt/latest_model.pth",
        type=str,
        help="specify the pretrained model",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument(
        "--tracking_config",
        type=str,
        default="./tracking_modules/configs/config.yml",
        help="tracking config file path",
    )
    parser.add_argument(
        "--tracking_output_dir",
        default="./tracking_result/",
        type=str,
    )
    parser.add_argument(
        "--calib_dir",
        default="../sample/calib/",
        type=str,
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    tracking_cfg = Config(args.tracking_config)
    return args, cfg, tracking_cfg


class TrackerDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        ext=".bin",
        args=None,
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        self.args = args
        path_list = os.listdir(root_path)
        path_list = natsort.natsorted(path_list)
        self.num_file_list = []
        self.lidar_files = []
        for dir_path in path_list:
            files = os.listdir(os.path.join(root_path, dir_path))
            files = natsort.natsorted(files)
            self.num_file_list.append(len(files))
            for file in files:
                self.lidar_files.append(os.path.join(root_path, dir_path, file))

    def __len__(self):
        return len(self.lidar_files) - 1

    def __getitem__(self, index):
        reduced_index = 0
        print(index)
        for idx in range(len(self.num_file_list)):
            reduced_index += self.num_file_list[idx]
            if index < reduced_index:
                frame_idx = idx
                break

        P2, V2C = read_calib(
            os.path.join(self.args.calib_dir, f"{str(frame_idx).zfill(4)}.txt")
        )
        if self.ext == ".bin":

            max_row = 374  # y
            max_col = 1241  # x

            lidar = np.fromfile(self.lidar_files[index], dtype=np.float32).reshape(
                -1, 4
            )

            mask = lidar[:, 0] > 0
            lidar = lidar[mask]

            lidar_copy = np.zeros(shape=lidar.shape)
            lidar_copy[:, :] = lidar[:, :]
            velo_tocam = V2C
            lidar[:, 3] = 1
            lidar = np.matmul(lidar, velo_tocam.T)
            img_pts = np.matmul(lidar, P2.T)

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
            points = lidar_copy[mask]
            # points = read_velodyne(velo_path,self.P2,self.V2C)

        elif self.ext == ".npy":
            pass
            # points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        input_dict = {"points": points, "frame_id": index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def _detection_postprocessing(pred_dicts, num_objects):
    """
    x,y,z,dx,dy,dz,rot,score

    """
    tracking_info_data = {}
    for idx in range(num_objects):
        tracking_info_data.setdefault(str(idx + 1), [])
    # print(f"pred_dicts: {pred_dicts}")
    for idx, pred_bbox in enumerate(pred_dicts[0]["pred_boxes"]):
        label = str(pred_dicts[0]["pred_labels"][idx].item())
        pred_bbox = pred_bbox.tolist()
        pred_bbox.append(pred_dicts[0]["pred_scores"][idx].tolist())
        pred_bbox.append(0.5)
        tracking_info_data[label].append(pred_bbox)
    return tracking_info_data


def main():
    args, detection_cfg, tracking_cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info("----------------- Spb3D Tracker -----------------")
    logger.info(f"cuda available : {torch.cuda.is_available()}")
    tracking_dataset = TrackerDataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
        args=args,
    )
    logger.info(f"Total number of samples: \t{len(tracking_dataset)}")
    model = build_network(
        model_cfg=detection_cfg.MODEL,
        num_class=len(detection_cfg.CLASS_NAMES),
        dataset=tracking_dataset,
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    tracking_time = time.time()
    ID_start_dict = {}
    for class_name in detection_cfg.CLASS_NAMES:
        ID_start_dict.setdefault(class_name, 1)
    logger.info(f"ID_start_dict : {ID_start_dict}")
    id_max = 0
    tracking_results_dict = {}
    tracker_dict = {}
    for class_name in detection_cfg.CLASS_NAMES:
        tracker_dict[class_name] = Spb3DMOT(ID_init=ID_start_dict.get(class_name))
    for idx in range(len(detection_cfg.CLASS_NAMES)):
        tracking_results_dict.setdefault(str(int(idx) + 1), [])
    with torch.no_grad():
        for data_dict in tracking_dataset:
            data_dict = tracking_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            detection_results_dict = _detection_postprocessing(
                pred_dicts, len(detection_cfg.CLASS_NAMES)
            )
            # TODO : nms
            for label, pred_bboxes in detection_results_dict.items():
                id_max = 0
                tracker = tracker_dict[detection_cfg.CLASS_NAMES[int(label) - 1]]
                tracking_result, _ = tracker.track(pred_bboxes)
                tracking_result = tracking_result[0].tolist()
                if len(tracking_result) != 0:
                    id_max = max(id_max, tracking_result[0][-1])
                ID_start_dict[detection_cfg.CLASS_NAMES[int(label) - 1]] = id_max + 1
                tracking_results_dict[label].append(tracking_result)
    logger.info(f"tracking time : { time.time()-tracking_time}")
    logger.info("========= logging.. =========")
    Path(args.tracking_output_dir).mkdir(parents=True, exist_ok=True)

    frame_idx = 0  # TODO : change to frame_id
    P2, V2C = read_calib(os.path.join(args.calib_dir, f"{str(frame_idx).zfill(4)}.txt"))
    with open(os.path.join(args.tracking_output_dir, "result.txt"), "w") as f:
        for class_name, tracking_results in tracking_results_dict.items():
            for frame_idx, tracking_result in enumerate(tracking_results):
                if len(tracking_result) == 0:
                    continue
                tracking_result = tracking_result[0]
                box = copy.deepcopy(tracking_result)
                box[:3] = tracking_result[3:6]
                box[3:6] = tracking_result[:3]
                box[2] -= box[5] / 2
                # box[:, 6] = -box[:, 6] - np.pi / 2
                box[:3] = vel_to_cam_pose(box[:3], V2C)[:3]
                box2d = bb3d_2_bb2d(box, P2)
                f.write(
                    f"{str(frame_idx)} {str(int(tracking_result[-1]))} {detection_cfg.CLASS_NAMES[int(class_name) - 1]} -1 -1 -10 {box2d[0][0]} {box2d[0][1]} {box2d[0][2]} {box2d[0][3]} {str(box[3])} {str(box[4])} {str(box[5])} {str(box[0])} {str(box[1])} {str(box[2])} {str(box[6])} \n"
                )
    logger.info("========= Finish =========")


# https://github.com/pratikac/kitti/blob/master/readme.tracking.txt
# frame, track_id, type, truncated, occluded, alpha, bbox(2d-left,top,right,bottom), dimensions(height,width,lennth), location(3d-x,y,z), rotation_y, score
# 0 2 Pedestrian 0 0 -2.523309 (1106.137292 166.576807 1204.470628 323.876144) (1.714062 0.767881 0.972283) (6.301919 1.652419 8.455685) -1.900245
if __name__ == "__main__":
    main()
