import copy

import numpy as np
from numba import jit

from .box import Box3D
from .kalman_filter import KF, UKF, get_bbox_distance
from .matching import data_association


@jit(nopython=True, cache=True)
def _within_range(theta):
    # make sure the orientation is within a proper range
    if theta >= np.pi:
        theta -= np.pi * 2  # make the theta stillf in the range
    if theta < -np.pi:
        theta += np.pi * 2

    return theta


@jit(nopython=True, cache=True)
def _select_giou_thres(bbox_a, bbox_b):
    # TODO : jit, set by delta T
    volume_size = (
        (bbox_a[3] * bbox_a[4] * bbox_a[5])
        if bbox_b is None
        else ((bbox_a[3] * bbox_a[4] * bbox_a[5]) + (bbox_b[3] * bbox_b[4] * bbox_b[5]))
        / 2.0
    )

    if volume_size > 15:
        return 0.1
    elif volume_size > 8:
        return -0.1

    elif volume_size > 5:
        return -0.5
    else:
        return -0.7


# TODO : adaptive filter
class Spb3DMOT(object):
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
        self, algm="hungar", metric="eiou", thres=-0.5, min_hits=1, max_age=2
    ):
        # if metric in ["dist_3d", "dist_2d", "m_dis"]:
        #     thres *= -1

        self.algm, self.metric, self.thres, self.max_age, self.min_hits = (
            algm,
            metric,
            thres,
            max_age,
            min_hits,
        )

        # define max/min values for the output affinity matrix
        # if self.metric in ["dist_3d", "dist_2d", "m_dis"]:
        #     self.max_sim, self.min_sim = 0.0, -100.0
        # elif self.metric in ["iou_2d", "iou_3d"]:
        #     self.max_sim, self.min_sim = 1.0, 0.0
        # elif self.metric in ["giou_2d", "giou_3d"]:
        self.max_sim, self.min_sim = 1.0, -1.0

    def process_dets(self, dets):
        # convert each detection into the class Box3D
        # inputs:
        # 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
        dets_new = []
        # dets : [[42.35746765136719, 6.294949531555176, -0.6724109649658203, 0.6947809457778931, 0.6723328232765198, 1.6847240924835205, 6.3626885414123535, 0.21080490946769714, 0.5],
        #         [-21.329811096191406, -1.3849732875823975, -1.0482765436172485, 0.6782370209693909, 0.6069652438163757, 1.665456771850586, 5.301192283630371, 0.14246301352977753, 0.5],
        # [44.61482620239258, 4.7019829750061035, -0.48246583342552185, 0.7044910788536072, 0.6307433843612671, 1.8210793733596802, 6.282752990722656, 0.12992656230926514, 0.5]]
        for det in dets:
            det_tmp = Box3D.pcdet2bbox_raw(det)
            dets_new.append(det_tmp)

        # dets_high = [det for det in dets if det[-2] > det[-1]]
        # dets_low = [det for det in dets if det[-2] < det[-1] and det[-2] > 0.1]
        return dets_new

    def orientation_correction(self, theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree

        # make the theta still in the range
        theta_pre = _within_range(theta_pre)
        theta_obs = _within_range(theta_obs)

        # if the angle of two theta is not acute angle, then make it acute
        if (
            abs(theta_obs - theta_pre) > np.pi / 2.0
            and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0
        ):
            theta_pre += np.pi
            theta_pre = _within_range(theta_pre)

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
        for tracker in self.trackers:
            # propagate locations
            kf_tmp = tracker
            kf_tmp.ukf.predict()
            kf_tmp.ukf.x[3] = _within_range(kf_tmp.ukf.x[3])
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

                # kalman filter update with observation
                trk.ukf.update(bbox3d[:-2])
                trk.distance = get_bbox_distance(trk.ukf.x[:3])

                trk.ukf.x[3] = _within_range(trk.ukf.x[3])
            else:
                trk.confidence -= trk.distance
                trk.hits = False

    def birth(self, dets, unmatched_dets):
        new_id_list = []  # new ID generated for unmatched detections
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
                results.append(
                    np.concatenate((d, [trk.confidence], [trk.id])).reshape(1, -1)
                )
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
        # TODO : association (theta - pcd, giou w size)
        # TODO : separate high score, low score w beyain
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
            _select_giou_thres(dets[0], dets[1])
            if len(dets) >= 2
            else _select_giou_thres(dets[0], None) if len(dets) == 1 else 0
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
        # print(f"results : {results}")
        # post-processing affinity to convert to the affinity between resulting tracklets
        # if self.affi_process:
        #     affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)

        return results, affi
