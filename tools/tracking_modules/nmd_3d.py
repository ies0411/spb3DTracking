# # import time

# import numpy as np

# from .nms import custom_nms


# # class score, x, y, z, dx,dy,dz)
# def iou(box_a, box_b):
#     """
#     deprecated function -> substituted to cpp
#     """
#     box_a_top_right_corner = [box_a[1] + (box_a[4] / 2.0), box_a[2] + (box_a[5] / 2.0)]
#     box_b_top_right_corner = [box_b[1] + (box_b[4] / 2.0), box_b[2] + (box_b[5] / 2.0)]

#     box_a_area = (box_a[4]) * (box_a[5])
#     box_b_area = (box_b[4]) * (box_b[5])

#     if box_a_top_right_corner[0] < box_b_top_right_corner[0]:
#         length_xi = box_a_top_right_corner[0] - (box_b[1] - (box_b[4] / 2.0))
#     else:
#         length_xi = box_b_top_right_corner[0] - (box_a[1] - (box_a[4] / 2.0))

#     if box_a_top_right_corner[1] < box_b_top_right_corner[1]:
#         length_yi = box_a_top_right_corner[1] - (box_b[2] - (box_b[5] / 2.0))
#     else:
#         length_yi = box_b_top_right_corner[1] - (box_a[2] - (box_a[5] / 2.0))

#     intersection_area = length_xi * length_yi

#     if length_xi <= 0 or length_yi <= 0:
#         iou = 0
#     else:
#         iou = intersection_area / (box_a_area + box_b_area - intersection_area)
#     return iou

# # def encoding_nms(pred):
# #     boxes = []
# #     for idx, pred in enumerate(preds):
# #         boxes.append(
# #             np.concatenate(
# #                 (np.array([scores[idx]]), np.array(pred), np.array([idx])), axis=0
# #             )
# #         )

# def nms(
#     original_boxes,
#     iou_thres_same_class,
#     iou_thres_different_class,
# ):
#     boxes_probability_sorted = original_boxes[np.flip(np.argsort(original_boxes[:, 0]))]
#     selected_boxes = []
#     for bbox in boxes_probability_sorted:
#         if bbox[0] > 0:
#             selected_boxes.append(bbox)
#             for other_box in boxes_probability_sorted:
#                 converted_iou_threshold = (
#                     iou_thres_same_class
#                     if bbox[-2] == other_box[-2]
#                     else iou_thres_different_class
#                 )
#                 if (
#                     other_box[-1] != bbox[-1]
#                     and custom_nms.iou(bbox[1:-2], other_box[1:-2])
#                     > converted_iou_threshold
#                 ):
#                     other_box[0] = 0
#     return selected_boxes
