import copy
import csv
import functools
import glob
import os
import random
import math
# from time import clock_settime
from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from my_app.utill.util import XyzTuple, xyz2irc
from my_app.utill.disk import getCache

from my_app.utill.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('combining_data_raw')

CandidateTuple = namedtuple('CandidateTuple', ['isNodule', 'diameter_mm', 'series_uid', 'center_xyz'])


@functools.lru_cache(1)
def getCandidatesList(reqOnDisk = True):

    mhdPath_list = glob.glob('../data/subset*/*.mhd')
    # print(mhdPath_list)
    presentOnDisk_set = {os.path.split(path)[-1][:-4] for path in mhdPath_list}

    diameter_dict = {}
    with open('../data/annotations.csv', 'r') as file:
        for row in list(csv.reader(file))[1:]:

            series_uid = row[0]
            annotationsCoord_xyz = tuple([float(x) for x in row[1:4]])
            annotationsDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationsCoord_xyz, annotationsDiameter_mm))

    candidates_list = []
    with open('../data/candidates.csv', 'r') as file:
        for row in list(csv.reader(file))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and reqOnDisk == True:
                continue

            isNodule = bool(int(row[4]))
            candidatesCoord_xyz = tuple([float(x) for x in row[1:4]])

            candidatesDiameter_mm = 0.0
            for annot_tuple in diameter_dict.get(series_uid, []):
                annotationsCoord_xyz, annotationsDiameter_mm = annot_tuple

                for i in range(3):
                    difference_mm = abs(candidatesCoord_xyz[i] - annotationsCoord_xyz[i])

                    if difference_mm > annotationsDiameter_mm/4:
                        break
                else:
                    candidatesDiameter_mm = annotationsDiameter_mm
                    break

            candidates_list.append(CandidateTuple(
                isNodule,
                candidatesDiameter_mm,
                series_uid,
                candidatesCoord_xyz
                ))

    candidates_list.sort(reverse=True)

    return candidates_list


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


"""Get the Augmented Candidate by applying
    mirroring,
    shifting by an offset,
    scaling,
    rotating(only along x and y axes),
    adding noise.

    Resample the candidate using affine_grid and grid_sample 

    Returns:
    Augmented Ct tensor and
    Nodule Center coordinate
"""
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

    def __init__(self, val_step=0,
                isValset=None,
                series_uid=None,
                sortby_str='random',
                ratio_int =0,
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


        if series_uid:
            self.candidates_list = [x for x in self.candidates_list if x.series_uid == series_uid]

        if isValset:
            assert val_step > 0, val_step
            self.candidates_list = self.candidates_list[::val_step]
            assert self.candidates_list

        elif val_step > 0:
            del self.candidates_list[::val_step]
            assert self.candidates_list

        if sortby_str == 'random':
            random.shuffle(self.candidates_list)
        elif sortby_str == 'series_uid':
            self.candidates_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [
            nt for nt in self.candidates_list if not nt.isNodule
        ]
        self.pos_list = [
            nt for nt in self.candidates_list if nt.isNodule
        ]

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidates_list),
            "validation" if isValset else "training",
        ))

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
        self,
        len(self.candidates_list),
        "validation" if isValset else "training",
        len(self.negative_list),
        len(self.pos_list),
        '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
    ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.ratio_int:
            return 23000
        else:
            return len(self.candidates_list)

    def __getitem__(self, index):
        print(1)
        if self.ratio_int:
            pos_index = index // (self.ratio_int + 1)

            if index % (self.ratio_int + 1):
                neg_index = index - 1 - pos_index
                neg_index %= len(self.negative_list)
                candidate_tuple = self.negative_list[neg_index]

            else:
                pos_index %= len(self.pos_list)
                candidate_tuple = self.pos_list[pos_index]

        else:
            candidate_tuple = self.candidates_list[index]
            
        width_irc = (32, 48, 48)

        if self.augmen_dict:
            candidate_tensor, center_irc = getAugmenCandidate(
                self.augmen_dict,
                candidate_tuple.series_uid,
                candidate_tuple.center_xyz,
                width_irc,
                self.use_cache
                )

        elif self.use_cache:
            candidate_arr, center_irc = getCtRawCandidate(
                candidate_tuple.series_uid,
                candidate_tuple.center_xyz,
                width_irc,
            )

            candidate_tensor = torch.from_numpy(candidate_arr)
            candidate_tensor = candidate_tensor.to(torch.float32)
            candidate_tensor = candidate_tensor.unsqueeze(0)

        else:
            ct = getCt(candidate_tuple.series_uid)
            candidate_arr, center_irc = ct.getCtChunk(
                candidate_tuple.center_xyz,
                width_irc,
            )

            candidate_tensor = torch.from_numpy(candidate_arr)
            candidate_tensor = candidate_tensor.to(torch.float32)
            candidate_tensor = candidate_tensor.unsqueeze(0)            

        isNodule_tensor = torch.tensor([not candidate_tuple.isNodule, candidate_tuple.isNodule], dtype=torch.long)

        return (
            candidate_tensor,
            isNodule_tensor,
            candidate_tuple.series_uid,
            torch.tensor(center_irc)
        )

# LunaDataset()[0]

getCandidatesList()