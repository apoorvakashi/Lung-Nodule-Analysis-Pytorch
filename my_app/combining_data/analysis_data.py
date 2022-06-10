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

raw_cache = getCache('noduleAnalysis_raw')

CandidateTuple = namedtuple(
    'CandidateTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz',
)
MaskTuple = namedtuple(
    'MaskTuple',
    'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask',
)

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
def getCandidateDict(reqOnDisk=True):

    candidates_list = getCandidatesList(reqOnDisk)
    candidates_dict = {}

    for candidate_tuple in candidates_list:
        candidates_dict.setdefault(candidate_tuple.series_uid, []).append(candidate_tuple)

    return candidates_dict


class Ct: 

    def __init__(self, series_uid):

        mhd_path = glob.glob('../data/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down

        ct_arr.clip(-1000, 1000, ct_arr)

        self.series_uid = series_uid
        self.hunits_arr = ct_arr

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.voxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_arr = np.array(ct_mhd.GetDirection()).reshape(3, 3)

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

        return ct_chunk, center_irc


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
    ct_chunk, center_irc = ct.getCtChunk(center_xyz, width_irc)
    return ct_chunk, center_irc


# @raw_cache.memoize(typed=True)
# def getCtSampleSize(series_uid):
#     ct = Ct(series_uid, buildMasks_bool=False)
#     return len(ct.negative_indexes)

# @raw_cache.memoize(typed=True)
# def getCtSampleSize(series_uid):
#     ct = Ct(series_uid)
#     return int(ct.hunits_arr.shape[0]), ct.positive_indices

def getAugmenCandidate(augmen_dict,
    series_uid,
    center_xyz,
    width_irc,
    use_cache = True
    ):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)

    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getCtChunk(center_xyz, width_irc)

    ct_tensor = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_tensor = torch.eye(4)

    for i in range(3):
        if 'flip' in augmen_dict:
            if random.random() > 0.5:
                transform_tensor[i,i] *= -1

        if 'offset' in augmen_dict:
            offset_val = augmen_dict['offset']
            random_val = (random.random() * 2 - 1)
            transform_tensor[i,3] = offset_val * random_val

        if 'scale' in augmen_dict:
            scale_val = augmen_dict['scale']
            random_val = (random.random() * 2 - 1)
            transform_tensor[i,i] *= 1.0 + scale_val*random_val

    if 'rotate' in augmen_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_tensor = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_tensor @= rotation_tensor

    affine_tensor = F.affine_grid(
        transform_tensor[:3].unsqueeze(0).to(torch.float32),
        ct_tensor.size(),
        align_corners=False
    )

    augmented_chunk = F.grid_sample(
        ct_tensor,
        affine_tensor,
        padding_mode='border',
        align_corners=False,
        ).to('cpu')

    if 'noise' in augmen_dict:
        noise_tensor = torch.randn_like(augmented_chunk)
        noise_tensor *= augmen_dict['noise']

        augmented_chunk += noise_tensor


    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self, 
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmen_dict=None,
                 candidates_list=None,
            ):
        self.ratio_int = ratio_int
        self.augmen_dict = augmen_dict

        if candidates_list:
            self.candidates_list = copy.copy(candidates_list)
            self.use_cache = False
        else:
            self.candidates_list = copy.copy(getCandidatesList())
            self.use_cache = True

        # print(len(self.candidates_list))

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(candidate_tuple.series_uid for candidate_tuple in self.candidates_list))

        # print(len(self.series_list))
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        series_set = set(self.series_list)
        self.candidates_list = [x for x in self.candidates_list if x.series_uid in series_set]
        # print(len(series_set))
        # print(len(self.candidates_list))

        if sortby_str == 'random':
            random.shuffle(self.candidates_list)
        elif sortby_str == 'series_uid':
            self.candidates_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.neg_list = [nt for nt in self.candidates_list if not nt.isNodule_bool]
        self.pos_list = [nt for nt in self.candidates_list if nt.isNodule_bool]
        self.ben_list = [nt for nt in self.pos_list if not nt.isMal_bool]
        self.mal_list = [nt for nt in self.pos_list if nt.isMal_bool]

        # print(len(self.neg_list))
        # print(len(self.pos_list))
        # print(len(self.mal_list))
        # print(len(self.ben_list))

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidates_list),
            "validation" if isValSet_bool else "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.candidates_list)
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    def __len__(self):
        if self.ratio_int:
            return 23000
        else:
            return len(self.candidates_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidate_tuple = self.neg_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidate_tuple = self.pos_list[pos_ndx]
        else:
            candidate_tuple = self.candidates_list[ndx]

        return self.sampleFromCandidate_tuple(
            candidate_tuple, candidate_tuple.isNodule_bool
        )

    def sampleFromCandidate_tuple(self, candidate_tuple, label_bool):
        width_irc = (32, 48, 48)

        if self.augmen_dict:
            candidate_t, center_irc = getAugmenCandidate(
                self.augmen_dict,
                candidate_tuple.series_uid,
                candidate_tuple.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidate_tuple.series_uid,
                candidate_tuple.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidate_tuple.series_uid)
            candidate_a, center_irc = ct.getCtChunk(
                candidate_tuple.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        label_t = torch.tensor([False, False], dtype=torch.long)

        if not label_bool:
            label_t[0] = True
            index_t = 0
        else:
            label_t[1] = True
            index_t = 1

        return candidate_t, label_t, index_t, candidate_tuple.series_uid, torch.tensor(center_irc)


class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio_int:
            return 23000
        else:
            return len(self.ben_list + self.mal_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            if ndx % 2 != 0:
                candidate_tuple = self.mal_list[(ndx // 2) % len(self.mal_list)]
            elif ndx % 4 == 0:
                candidate_tuple = self.ben_list[(ndx // 4) % len(self.ben_list)]
            else:
                candidate_tuple = self.neg_list[(ndx // 4) % len(self.neg_list)]
        else:
            if ndx >= len(self.ben_list):
                candidate_tuple = self.mal_list[ndx - len(self.ben_list)]
            else:
                candidate_tuple = self.ben_list[ndx]

        return self.sampleFromCandidate_tuple(
            candidate_tuple, candidate_tuple.isMal_bool
        )