import glob
import numpy as np
import os

from my_app.combining_data.combining_data import CandidateTuple, Ct, getCandidatesList, getCt
from my_app.utill.util import XyzTuple

def test_getCandidatesList():
  candidates_list = getCandidatesList()

  assert isinstance(candidates_list, list)
  assert len(candidates_list) > 0

  candidate = candidates_list[0]
  center_xyz = getattr(candidate, 'center_xyz')

  assert isinstance(candidate, CandidateTuple)
  assert isinstance(getattr(candidate, 'isNodule'), bool)
  assert isinstance(getattr(candidate, 'diameter_mm'), float)
  assert isinstance(getattr(candidate, 'series_uid'), str)
  
  assert isinstance(center_xyz, tuple)
  assert len(center_xyz) == 3
  assert isinstance(center_xyz[0], float)
  assert isinstance(center_xyz[1], float)
  assert isinstance(center_xyz[2], float)

def test_getCt():
  mhdPath_list = glob.glob('../data/subset*/*.mhd')
  series_uuid_list = [os.path.split(path)[-1][:-4] for path in mhdPath_list]

  for series_uuid in series_uuid_list[:5]:
    ct = getCt(series_uuid)

    assert isinstance(ct, Ct)

    assert isinstance(ct.hunits_arr, np.ndarray)
    assert ct.hunits_arr.shape[1:] == (512, 512)

    assert isinstance(ct.origin_xyz, XyzTuple)
    assert len(ct.origin_xyz) == 3
    assert isinstance(ct.origin_xyz[0], float)
    assert isinstance(ct.origin_xyz[1], float)
    assert isinstance(ct.origin_xyz[2], float)

    assert isinstance(ct.voxSize_xyz, XyzTuple)
    assert len(ct.voxSize_xyz) == 3
    assert isinstance(ct.voxSize_xyz[0], float)
    assert isinstance(ct.voxSize_xyz[1], float)
    assert isinstance(ct.voxSize_xyz[2], float)

    assert isinstance(ct.direction_arr, np.ndarray)
    assert ct.direction_arr.shape == (3, 3)
