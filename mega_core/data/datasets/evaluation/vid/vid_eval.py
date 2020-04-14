from __future__ import division

import os
from collections import defaultdict
import numpy as np

import torch

from mega_core.structures.bounding_box import BoxList
from mega_core.structures.boxlist_ops import boxlist_iou


def do_vid_evaluation(dataset, predictions, output_folder, box_only, logger):
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)
    if box_only:
        result = eval_proposals_vid(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=0.5,
        )
        result_str = "Recall: {:.4f}".format(result["recall"])
        logger.info(result_str)
        if output_folder:
            with open(os.path.join(output_folder, "proposal_result.txt"), "w") as fid:
                fid.write(result_str)
        return

    result = eval_detection_vid(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=False,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    print(result_str)
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    return result


def eval_proposals_vid(pred_boxlists, gt_boxlists, iou_thresh=0.5, limit=300):
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."

    gt_overlaps = []
    num_pos = 0
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        inds = pred_boxlist.get_field("objectness").sort(descending=True)[1]
        pred_boxlist = pred_boxlist[inds]

        if len(pred_boxlist) > limit:
            pred_boxlist = pred_boxlist[:limit]

        num_pos += len(gt_boxlist)

        if len(gt_boxlist) == 0:
            continue

        if len(pred_boxlist) == 0:
            continue

        overlaps = boxlist_iou(pred_boxlist, gt_boxlist)

        _gt_overlaps = torch.zeros(len(gt_boxlist))
        for j in range(min(len(pred_boxlist), len(gt_boxlist))):
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0

            box_ind = argmax_overlaps[gt_ind]

            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr

            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    recall = (gt_overlaps >= iou_thresh).float().sum() / float(num_pos)

    return {
        "recall": recall
    }


def eval_detection_vid(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on vid dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_vid_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_vid_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_vid_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of VID.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):

            pred_mask_l = pred_label == l
            if not all(pred_mask_l):
                print("not make sence!!")
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            # index = pred_score_l >= 0.05
            #
            # pred_bbox_l = pred_bbox_l[index, :]
            # pred_score_l = pred_score_l[index]

            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]

            n_pos[l] += gt_bbox_l.shape[0]
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VID evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()

            num_obj, num_gt_obj = iou.shape

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for j in range(0, num_obj):
                iou_match = -1
                arg_match = -1
                for k in range(0, num_gt_obj):
                    if selec[k]:
                        continue
                    if iou[j, k] >= iou_thresh and iou[j, k] > iou_match:
                        iou_match = iou[j, k]
                        arg_match = k
                if arg_match >= 0:
                    match[l].append(1)
                    selec[arg_match] = True
                else:
                    match[l].append(0)

            # gt_index = iou.argmax(axis=1)
            # # set -1 if there is no matching ground truth
            # gt_index[iou.max(axis=1) < iou_thresh] = -1
            # del iou
            #
            # selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            # for gt_idx in gt_index:
            #     if gt_idx >= 0:
            #         if not selec[gt_idx]:
            #             match[l].append(1)
            #         else:
            #             match[l].append(0)
            #         selec[gt_idx] = True
            #     else:
            #         match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    print(n_pos)
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_vid_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of VID.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in VID.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
