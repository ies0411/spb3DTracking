import argparse
import os
from pathlib import Path

import natsort
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tracking_modules.utils import Config, convert_to_kittiforamt


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--dataset', type=str, default='KITTI', help='KITTI, nuScenes')
    parser.add_argument('--split', type=str, default='val', help='train, val, test')
    parser.add_argument('--det_name', type=str, default='pointrcnn', help='pointrcnn')

    parser.add_argument(
        "--cfg_file",
        type=str,
        default="/cfgs/kitti_models/pv_rcnn_plusplus_resnet.yaml ",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/nas3/Data/kitti-processed/object_tracking/training/velodyne",
        help="specify the point cloud data file or directory",
    )
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
        "--confidence",
        type=float,
        default=0.1,
        help="confidence threshold of detection",
    )
    parser.add_argument(
        "--tracking_output_dir",
        default="./trk_result/",
        type=str,
    )
    parser.add_argument(
        "--mot_output_dir",
        default="./mot_result/",
        type=str,
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    tracking_cfg = Config(args.tracking_config)
    return args, cfg, tracking_cfg

def read_files_in_folders(root_folder,tracking_seqs):
    all_data_list = []
    data_dir_list = os.listdir(root_folder)
    data_dir_list = natsort.natsorted(data_dir_list)
    for data_dir in data_dir_list:
        if int(data_dir) in tracking_seqs:
            file_name_list = os.listdir(os.path.join(root_folder,data_dir))
            file_name_list = natsort.natsorted(file_name_list)
            for file_name in file_name_list:
                all_data_list.append(os.path.join(root_folder,data_dir,file_name))
    return all_data_list

class DemoDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        tracking_cfg,
        training=True,
        root_path=None,
        logger=None,
        ext=".bin",
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
        self.all_data_list = read_files_in_folders(root_path,tracking_cfg["tracking_seqs"])

    def __len__(self):
        return len(self.all_data_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(
                self.all_data_list[index], dtype=np.float32
            ).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.all_data_list[index])
        else:
            raise NotImplementedError
        input_dict = {"points": points, "frame_id": int(self.all_data_list[index].split("/")[-1].split(".")[0]),
                      "group" : int(self.all_data_list[index].split("/")[-2])}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def postprocessing_det(pred_dicts, tracking_type_id, thres, update_confidence):
    results = []
    for idx, pred_bbox in enumerate(pred_dicts[0]["pred_boxes"]):
        if tracking_type_id == pred_dicts[0]["pred_labels"][idx]:
            if thres > pred_dicts[0]["pred_scores"][idx]:
                continue
            result = []
            result.extend(pred_bbox.tolist())
            result.append(pred_dicts[0]["pred_scores"][idx].tolist())
            result.append(update_confidence)
            results.append(result)
    return results




def main():
    args, detection_cfg, tracking_cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Swift 3DMOT-------------------------")
    logger.info(f"cuda available : {torch.cuda.is_available()}")
    demo_dataset = DemoDataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
        tracking_cfg = tracking_cfg,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    logger.info(f"Total number of samples: \t{len(demo_dataset)}")

    model = build_network(
        model_cfg=detection_cfg.MODEL,
        num_class=len(detection_cfg.CLASS_NAMES),
        dataset=demo_dataset,
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    tracking_type_id = tracking_cfg.class_list.index(tracking_cfg.tracking_type)+1
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            group,frame_id = data_dict["group"],data_dict["frame_id"]
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            convert_to_kittiforamt(postprocessing_det(
                pred_dicts,
                tracking_type_id,
                args.confidence,
                tracking_cfg.update_confidence
            ), group, frame_id, tracking_cfg)

if __name__ == "__main__":
    main()
