import argparse
import glob
import time
from pathlib import Path

import torch
import argparse
import time

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tracking_modules.model import Spb3DMOT
from tracking_modules.utils import Config


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
        default="../sample/1",
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


class TrackerDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
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
        data_file_list = (
            glob.glob(str(root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        input_dict = {"points": points, "frame_id": index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def _detection_postprocessing(pred_dicts):
    """
    x,y,z,dx,dy,dz,rot,score

    """
    tracking_info_data = {}
    for idx, pred_bbox in enumerate(pred_dicts[0]["pred_boxes"]):
        label = str(pred_dicts[0]["pred_labels"][idx].item())
        tracking_info_data.setdefault(label, [])
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

    with torch.no_grad():
        for data_dict in tracking_dataset:
            data_dict = tracking_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            detection_results_dict = _detection_postprocessing(pred_dicts)
            # TODO : nms
            for label, pred_bboxes in detection_results_dict.items():
                id_max = 0
                tracker = tracker_dict[detection_cfg.CLASS_NAMES[int(label) - 1]]
                tracking_result, _ = tracker.track(pred_bboxes)
                tracking_result = tracking_result[0].tolist()
                if len(tracking_result) != 0:
                    id_max = max(id_max, tracking_result[0][-1])
                ID_start_dict[detection_cfg.CLASS_NAMES[int(label) - 1]] = id_max + 1
                tracking_results_dict.setdefault(label, [])
                tracking_results_dict[label].append(tracking_result)
    logger.info(f"tracking_results_dict : {tracking_results_dict}")
    logger.info(f"tracking time : { time.time()-tracking_time}")
    logger.info("========= Finish =========")


if __name__ == "__main__":
    main()
