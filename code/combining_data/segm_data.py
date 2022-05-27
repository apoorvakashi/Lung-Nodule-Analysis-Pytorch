import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from utill.disk import getCache
from utill.util import XyzTuple, xyz2irc
from utill.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('segm_data_raw')


MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateTuple = namedtuple('CandidateTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def getCandidatesList(reqOnDisk = True):
    mhdPath_list = glob.glob('../data/subset*/*.mhd')
    # print(mhdPath_list)
    presentOnDisk_set = {os.path.split(path)[-1][:-4] for path in mhdPath_list}
    # logging.info(presentOnDisk_set)

    candidates_list = []
    with open('../data/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and reqOnDisk:
                continue

            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidates_list.append(
                CandidateTuple(
                    True,
                    True,
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

    with open('../data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and reqOnDisk:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidates_list.append(
                    CandidateTuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    candidates_list.sort(reverse=True)
    return candidates_list


@functools.lru_cache(1)
def getCandidateDict(requireOnDisk_bool=True):

    candidates_list = getCandidatesList(requireOnDisk_bool)
    candidates_dict = {}

    for candidate_tuple in candidates_list:
        candidates_dict.setdefault(candidate_tuple.series_uid,
                                      []).append(candidate_tuple)

    return candidates_dict


class Ct:

    def __init__(self, series_uid):

        # print(glob.glob('../data/subset*/{}.mhd'.format(series_uid)))
        mhd_path = glob.glob('../data/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)

        self.hunits_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.voxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_arr = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidates_list = getCandidateDict()[self.series_uid]

        self.positiveCand_list = [
            candidate_tuple for candidate_tuple in candidates_list
            if candidate_tuple.isNodule_bool
        ]

        #Build a 3D mask of positive candidates with series_uid
        #Each mask conains flagged voxels which belong to a nodule
        #Voxel is only flagged if hu is above threshold
        self.positive_mask = self.buildAnnotationMask(self.positiveCand_list)

        # each slice of ct of series_uid is checked.
        # Index of each slice where sum is not zero is stored as array in a tuple
        # Array is convertd to list
        self.positive_indices = (self.positive_mask.sum(axis=(1,2))
                                 .nonzero()[0].tolist())


    def buildAnnotationMask(self, positiveCand_list, threshold_hu = -700):

        boundingBox_arr = np.zeros_like(self.hunits_arr, dtype=np.bool)

        for candidate_tuple in positiveCand_list:
            center_irc = xyz2irc(
                candidate_tuple.center_xyz,
                self.origin_xyz,
                self.voxSize_xyz,
                self.direction_arr,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hunits_arr[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hunits_arr[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hunits_arr[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hunits_arr[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hunits_arr[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hunits_arr[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            boundingBox_arr[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True

        mask_arr = boundingBox_arr & (self.hunits_arr > threshold_hu)

        return mask_arr


    def getCtChunk(self, center_xyz, width_irc):

        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.voxSize_xyz,
            self.direction_arr,
        )

        slices = []
        for axis, center_val in enumerate(center_irc):
            start_index = int(round(center_val - width_irc[axis]/2))
            end_index = int(start_index + width_irc[axis]) 

            assert center_val >= 0 and center_val < self.hunits_arr.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])   

            if start_index < 0:
                    start_index = 0
                    end_index = int(width_irc[axis])

            if end_index > self.hunits_arr.shape[axis]:
                end_index = self.hunits_arr.shape[axis]
                start_index = int(self.hunits_arr.shape[axis] - width_irc[axis])

            slices.append(slice(start_index, end_index))

        ct_chunk = self.hunits_arr[tuple(slices)]
        pos_chunk = self.positive_mask[tuple(slices)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


"""Generate raw candidate using cache

    Returns:
    3D Ct chunk containing nodule,
    Nodule Center Coordinate
"""
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk,pos_chunk, center_irc = ct.getCtChunk(center_xyz, width_irc)

    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hunits_arr.shape[0]), ct.positive_indices


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
            ):

        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list

        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indices = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indices]

        self.candidates_list = getCandidatesList()

        series_set = set(self.series_list)
        self.candidates_list = [cit for cit in self.candidates_list
                                   if cit.series_uid in series_set]

        self.pos_list = [nt for nt in self.candidates_list
                            if nt.isNodule_bool]

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_tensor = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hunits_arr.shape[0] - 1)
            ct_tensor[i] = torch.from_numpy(ct.hunits_arr[context_ndx].astype(np.float32))

        ct_tensor.clamp_(-1000, 1000)

        pos_tensor = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_tensor, pos_tensor, ct.series_uid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 50000

    def shuffleSamples(self):
        random.shuffle(self.candidates_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidate_tuple = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidate_tuple)

    def getitem_trainingCrop(self, candidate_tuple):
        ct_arr, pos_arr, center_irc = getCtRawCandidate(
            candidate_tuple.series_uid,
            candidate_tuple.center_xyz,
            (7, 96, 96),
        )
        pos_arr = pos_arr[3:4]

        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)
        ct_t = torch.from_numpy(ct_arr[:, row_offset:row_offset+64,
                                     col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_arr[:, row_offset:row_offset+64,
                                       col_offset:col_offset+64]).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidate_tuple.series_uid, slice_ndx


class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidatesList()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtRawCandidate(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96))

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1
