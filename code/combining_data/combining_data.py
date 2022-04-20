import copy
import csv
import functools
import glob
import os
# from time import clock_settime
from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from utill.util import XyzTuple, xyz2irc


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


# @raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getCtChunk(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):

    def __init__(self, val_step=0, isValset=None, series_uid=None):
        self.candidates_list = copy.copy(getCandidatesList())

        if series_uid:
            self.candidates_list = [x for x in self.candidates_list if x.series_uid == series_uid]

        if isValset:
            assert val_step > 0, val_step
            self.candidates_list = self.candidates_list[::val_step]
            assert self.candidates_list

        elif val_step > 0:
            del self.candidates_list[::val_step]
            assert self.candidates_list

    def __len__(self):
        return len(self.candidates_list)

    def __getitem__(self, index):
        candidate_tuple = self.candidates_list[index]
        width_irc = (32, 48, 48)

        candidate_arr, center_irc = getCtRawCandidate(
            candidate_tuple.series_uid,
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

