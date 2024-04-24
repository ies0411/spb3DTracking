import copy
import logging

import numpy as np

from autolabel3d.utils.box3d import Box3D
from autolabel3d.utils.kalman_filter import KF, UKF, get_bbox_distance
from autolabel3d.utils.tracker_utils import (
    compute_affinity,
    data_association,
    greedy_matching,
    select_giou_thres,
)

logger = logging.getLogger("al.pointcloud.tracker")


# TODO : adaptive filter
class PointcloudObjectTracker(object):
    def __init__(self, ID_init=0):
        # counter
        self.trackers = []
        self.frame_count = 0
        self.ID_count = [ID_init]
        self.id_now_output = []

        # config
        self.affi_process = False

        # self.get_param(algm, metric, thres, min_hits, max_age)
        self.algm = None
        self.metric = None
        self.thres = None
        self.max_age = None
        self.min_hits = None
        # debug
        self.debug_id = None
        self.death_threshold = 0.1

    def get_param(  # "greedy"
        self, algm="hungar", metric="giou_3d", thres=-0.5, min_hits=1, max_age=2
    ):
        # get parameters for each dataset

        # add negative due to it is the cost
        if metric in ["dist_3d", "dist_2d", "m_dis"]:
            thres *= -1
        self.algm, self.metric, self.thres, self.max_age, self.min_hits = (
            algm,
            metric,
            thres,
            max_age,
            min_hits,
        )

        # define max/min values for the output affinity matrix
        if self.metric in ["dist_3d", "dist_2d", "m_dis"]:
            self.max_sim, self.min_sim = 0.0, -100.0
        elif self.metric in ["iou_2d", "iou_3d"]:
            self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ["giou_2d", "giou_3d"]:
            self.max_sim, self.min_sim = 1.0, -1.0

    def process_dets(self, dets):
        # convert each detection into the class Box3D
        # inputs:
        # 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
        dets_new = []
        for det in dets:
            det_tmp = Box3D.pcdet2bbox_raw(det)
            dets_new.append(det_tmp)

        return dets_new

    def within_range(self, theta):
        # make sure the orientation is within a proper range

        if theta >= np.pi:
            theta -= np.pi * 2  # make the theta still in the range
        if theta < -np.pi:
            theta += np.pi * 2

        return theta

    def orientation_correction(self, theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree

        # make the theta still in the range
        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

        # if the angle of two theta is not acute angle, then make it acute
        if (
            abs(theta_obs - theta_pre) > np.pi / 2.0
            and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0
        ):
            theta_pre += np.pi
            theta_pre = self.within_range(theta_pre)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0:
                theta_pre += np.pi * 2
            else:
                theta_pre -= np.pi * 2

        return theta_pre, theta_obs

    def prediction(self):
        # get predicted locations from existing tracks
        trks = []
        for t in range(len(self.trackers)):
            # propagate locations
            kf_tmp = self.trackers[t]
            if kf_tmp.id == self.debug_id:
                logger.info("\n before prediction")
                logger.info(kf_tmp.ukf.x.reshape((-1)))
                logger.info("\n current velocity")
                logger.info(kf_tmp.get_velocity())
            kf_tmp.ukf.predict()
            if kf_tmp.id == self.debug_id:
                logger.info("After prediction")
                logger.info(kf_tmp.ukf.x.reshape((-1)))
            kf_tmp.ukf.x[3] = self.within_range(kf_tmp.ukf.x[3])

            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.ukf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def update(self, matched, unmatched_trks, dets):
        # update matched trackers with assigned detections

        dets = copy.copy(dets)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                assert len(d) == 1, "error"

                # update statistics
                # trk.time_since_update = 0  # reset because just updated
                trk.confidence = max(trk.confidence, dets[d[0]].s)
                trk.hits = True

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.ukf.x[3], bbox3d[3] = self.orientation_correction(
                    trk.ukf.x[3], bbox3d[3]
                )
                trk.ukf.R[0:, 0:] *= np.clip(0.05 * (1 - trk.confidence), 0.001, 0.1)
                if trk.id == self.debug_id:
                    logger.info("After ego-compoensation")
                    logger.info(trk.ukf.x.reshape((-1)))
                    logger.info("matched measurement")
                    logger.info(bbox3d.reshape((-1)))
                    logger.info("uncertainty")
                    logger.info(trk.ukf.P)
                    logger.info("measurement noise")
                    logger.info(trk.ukf.R)

                # kalman filter update with observation
                trk.ukf.update(bbox3d[:-2])
                trk.distance = get_bbox_distance(trk.ukf.x[:3])
                if trk.id == self.debug_id:
                    logger.info("after matching")
                    logger.info(trk.ukf.x.reshape((-1)))
                    logger.info("\n current velocity")
                    logger.info(trk.get_velocity())

                trk.ukf.x[3] = self.within_range(trk.ukf.x[3])
            else:
                trk.confidence -= trk.distance
                trk.hits = False

    def birth(self, dets, unmatched_dets):
        new_id_list = list()  # new ID generated for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            # trk = KF(Box3D.bbox2array(dets[i]), self.ID_count[0])
            trk = UKF(Box3D.bbox2array(dets[i]), self.ID_count[0])
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            self.ID_count[0] += 1

        return new_id_list

    # TODO : ukf speed
    def output(self):
        num_trks = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            d = Box3D.array2bbox(trk.ukf.x[:7].reshape((7,)))  # bbox location self
            d = Box3D.bbox2array_raw(d)
            if (trk.confidence >= trk.threshold) and (
                ((trk.hits is True) or (self.frame_count == 1))
            ):
                results.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            num_trks -= 1

            if trk.confidence <= self.death_threshold:
                self.trackers.pop(num_trks)

        return results

    def process_affi(self, affi, matched, unmatched_dets, new_id_list):
        ###### determine the ID for each past track
        trk_id = self.id_past  # ID in the trks for matching

        ###### determine the ID for each current detection
        det_id = [-1 for _ in range(affi.shape[0])]  # initialization

        # assign ID to each detection if it is matched to a track
        for match_tmp in matched:
            det_id[match_tmp[0]] = trk_id[match_tmp[1]]

        # assign the new birth ID to each unmatched detection
        count = 0
        assert len(unmatched_dets) == len(new_id_list), "error"
        for unmatch_tmp in unmatched_dets:
            det_id[unmatch_tmp] = new_id_list[
                count
            ]  # new_id_list is in the same order as unmatched_dets
            count += 1
        assert not (-1 in det_id), "error, still have invalid ID in the detection list"

        ############################ update the affinity matrix based on the ID matching

        # transpose so that now row is past trks, col is current dets
        affi = affi.transpose()

        ###### compute the permutation for rows (past tracklets), possible to delete but not add new rows
        permute_row = list()
        for output_id_tmp in self.id_past_output:
            index = trk_id.index(output_id_tmp)
            permute_row.append(index)
        affi = affi[permute_row, :]
        assert affi.shape[0] == len(self.id_past_output), "error"

        ###### compute the permutation for columns (current tracklets), possible to delete and add new rows
        # addition can be because some tracklets propagated from previous frames with no detection matched
        # so they are not contained in the original detection columns of affinity matrix, deletion can happen
        # because some detections are not matched

        max_index = affi.shape[1]
        permute_col = list()
        to_fill_col, to_fill_id = (
            list(),
            list(),
        )  # append new columns at the end, also remember the ID for the added ones
        for output_id_tmp in self.id_now_output:
            try:
                index = det_id.index(output_id_tmp)
            except:  # some output ID does not exist in the detections but rather predicted by KF
                index = max_index
                max_index += 1
                to_fill_col.append(index)
                to_fill_id.append(output_id_tmp)
            permute_col.append(index)

        # expand the affinity matrix with newly added columns
        append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
        append.fill(self.min_sim)
        affi = np.concatenate([affi, append], axis=1)

        # find out the correct permutation for the newly added columns of ID
        for count in range(len(to_fill_col)):
            fill_col = to_fill_col[count]
            fill_id = to_fill_id[count]
            row_index = self.id_past_output.index(fill_id)

            # construct one hot vector because it is proapgated from previous tracks, so 100% matching
            affi[row_index, fill_col] = self.max_sim
        affi = affi[:, permute_col]

        return affi

    def track(self, dets):
        # TODO : kitti crop
        """
        Params:
                dets_all: dict
                        dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
                        info: a array of other info for each det
                frame:    str, frame number, used to query ego pose
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        thres = (
            select_giou_thres(dets[0], dets[1])
            if len(dets) >= 2
            else select_giou_thres(dets[0], None)
            if len(dets) == 1
            else 0
        )
        self.get_param(thres=thres)
        self.frame_count += 1

        # recall the last frames of outputs for computing ID correspondences during affinity processing
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

        # process detection format
        dets = self.process_dets(dets)

        # tracks propagation based on velocity
        trks = self.prediction()

        # matching
        trk_innovation_matrix = None

        matched, unmatched_dets, unmatched_trks, cost, affi = data_association(
            dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix
        )

        # update trks with matched detection measurement
        self.update(matched, unmatched_trks, dets)

        # create and initialise new trackers for unmatched detections
        new_id_list = self.birth(dets, unmatched_dets)
        # output existing valid tracks
        results = self.output()

        if len(results) > 0:
            results = [
                np.concatenate(results)
            ]  # h,w,l,x,y,z,theta, ID, other info, confidence
        else:
            results = [np.empty((0, 15))]
        self.id_now_output = results[0][
            :, 7
        ].tolist()  # only the active tracks that are outputed

        # post-processing affinity to convert to the affinity between resulting tracklets
        if self.affi_process:
            affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)

        return results, affi
