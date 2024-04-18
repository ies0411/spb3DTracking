import argparse
import glob
import time
from pathlib import Path

import torch
import argparse
import os
import time

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tracking_modules.io import (
    get_frame_det,
    get_saving_dir,
    load_detection,
    save_affinity,
    save_results,
)
from tracking_modules.model import SWIFT3DMOT

# AB3DMOT
from tracking_modules.utils import Config, createFolder, get_subfolder_seq

# from tracking_utils.tracker import PointcloudObjectTracker


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--dataset", type=str, default="KITTI", help="KITTI, nuScenes")
    parser.add_argument("--split", type=str, default="val", help="train, val, test")
    parser.add_argument("--det_name", type=str, default="pointrcnn", help="pointrcnn")

    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/pv_rcnn.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/pv_rcnn_8369.pth",
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


class DemoDataset(DatasetTemplate):
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


def apply_seperate(pred_dicts, tracking_info_data, num_label, thres):
    bbox = {}
    score = {}
    for idx in range(num_label):
        bbox[str(idx + 1)] = []
        score[str(idx + 1)] = []

    for idx, pred_bbox in enumerate(pred_dicts[0]["pred_boxes"]):
        if thres > pred_dicts[0]["pred_scores"][idx]:
            continue

        label = str(pred_dicts[0]["pred_labels"][idx].item())
        bbox[label].append(pred_bbox.tolist())
        score[label].append(pred_dicts[0]["pred_scores"][idx].tolist())

    for idx in range(num_label):
        tracking_info_data["bbox"][str(idx + 1)].append(bbox[str(idx + 1)])
        # tracking_info_data["bbox"][str(idx + 1)].append(score[str(idx + 1)])
        tracking_info_data["score"][str(idx + 1)].append(score[str(idx + 1)])


def main():
    args, detection_cfg, tracking_cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Superb 3D CAL-------------------------")
    logger.info(f"cuda available : {torch.cuda.is_available()}")
    demo_dataset = DemoDataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
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

    tracking_info_data = {}
    tracking_info_data["pcd"] = []
    tracking_info_data["bbox"] = {}
    tracking_info_data["score"] = {}

    for class_idx in range(len(detection_cfg.CLASS_NAMES)):
        tracking_info_data["bbox"][str(class_idx + 1)] = []
        tracking_info_data["score"][str(class_idx + 1)] = []

    # TODO : set batch size
    total_time = time.time()
    inference_time = time.time()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            tracking_info_data["pcd"].append(data_dict["points"][:, 1:])
            apply_seperate(
                pred_dicts,
                tracking_info_data,
                len(detection_cfg.CLASS_NAMES),
                args.confidence,
            )

    logger.info(f"detection inference time : {time.time()-inference_time}")

    tracking_time = time.time()
    tracking_results = []

    ID_start = 1
    id_max = 0

    ########
    #     result_sha = '%s_%s_%s' % (args.det_name, "Car", "val")
    #     det_root = os.path.join('/home/eslim/workspace/Swift3DMOT/data', args.dataset, 'detection', result_sha)

    #     subfolder, det_id2str, hw, seq_eval, data_root = get_subfolder_seq(args.dataset, args.split)
    #     print(f'det_id2str : {det_id2str}')
    #     print(f'seq_eval : {seq_eval}')
    #     print(f'hw : {hw}')
    #     # trk_root = os.path.join(data_root, 'tracking')
    #     # save_dir = os.path.join(cfg.save_root, result_sha + '_H%d' % cfg.num_hypo); mkdir_if_missing(save_dir)
    #     eval_dir_dict = dict()
    #     eval_dir_dict = os.path.join(args.tracking_output_dir, 'data_%d' % 0)

    # # det_id2str : {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    #     ########
    #     seq_count = 0
    #     ID_start = 1
    #     total_time, total_frames = 0.0, 0

    #     for seq_name in seq_eval:
    #         seq_file = os.path.join(det_root, seq_name+'.txt')
    #         seq_dets, flag = load_detection(seq_file)
    #         # print(f'seq_dets : {seq_dets}')
    #             # load detection
    #         if not flag: continue
    #         eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = \
    # 			get_saving_dir(eval_dir_dict, seq_name, args.tracking_output_dir, 1)

    #         tracker = SWIFT3DMOT(ID_init=ID_start)
    #         # min_frame, max_frame = int(frame_list[0]), int(frame_list[-1])

    #         # for
    #         dets_frame = get_frame_det(seq_dets, 0)
    #         # print(f'dets_frame : {dets_frame}')
    det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    for label, pred_bboxes in tracking_info_data["bbox"].items():
        converted_bbox = []
        tracker = SWIFT3DMOT(ID_init=ID_start)
        for idx, pred_bbox in enumerate(pred_bboxes):
            for order, bbox in enumerate(pred_bbox):
                bbox.append(tracking_info_data["score"][label][idx][order])
                bbox.append(0.5)
                converted_bbox.append(bbox)
            tracking_result, affi = tracker.track(converted_bbox)
            print(f"tracking_result : {tracking_result}")
            convertformat_det_to_track(tracking_result)

            save_trk_file = os.path.join(args.tracking_output_dir, "%06d.txt" % idx)
            save_trk_file = open(save_trk_file, "w")
            for result_tmp in tracking_result:  # N x 15
                save_results(
                    result_tmp, save_trk_file, args.mot_output_dir, 0, idx, 0.5
                )
            save_trk_file.close()
            tracking_result = tracking_result[0]
            if len(tracking_result) != 0:
                id_max = max(id_max, tracking_result[0][7])
        ID_start = id_max + 1
    tracking_results.append(tracking_result.tolist())

    for frame, tracking_result in enumerate(tracking_results):
        save_trk_file = os.path.join(args.tracking_output_dir, "%06d.txt" % frame)
        save_trk_file = open(save_trk_file, "w")
        for result_tmp in tracking_result:  # N x 15
            save_results(
                result_tmp,
                save_trk_file,
                eval_file_dict,
                det_id2str,
                frame,
                cfg.score_threshold,
            )
        save_trk_file.close()

    print(tracking_results)
    logger.info(f"tracking time : { time.time()-tracking_time}")
    logger.info(f"total time : {time.time()-total_time}")
    logger.info("========= Finish =========")


if __name__ == "__main__":
    main()


# def tracking(detection_results):
#     total_result = {}
#     ID_start = 1
#     id_max = 0
#     for label in detection_results[0].keys():
#         each_class_result = []
#         tracker = PointcloudObjectTracker(ID_init=ID_start)
#         for frame_id, result in detection_results.items():
#             tracking_result, affi = tracker.track(result[label])
#             tracking_result = tracking_result[0]
#             if len(tracking_result) != 0:
#                 id_max = max(id_max, tracking_result[0][7])
#             each_class_result.append(tracking_result.tolist())
#         total_result[label] = each_class_result
#         ID_start = id_max + 1
#     return total_result
