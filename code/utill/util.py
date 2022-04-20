import collections
import copy
import datetime
import gc
import time

# import torch
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, voxSize_xyz, direction_arr):
    cri_arr = np.array(coord_irc)[::-1]
    origin_arr = np.array(origin_xyz)
    voxSize_arr = np.array(voxSize_xyz)
    coords_xyz = (direction_arr @ (cri_arr * voxSize_arr)) + origin_arr
    # coords_xyz = (direction_arr @ (idx * voxSize_arr)) + origin_arr
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, voxSize_xyz, direction_arr):
    origin_arr = np.array(origin_xyz)
    voxSize_arr = np.array(voxSize_xyz)
    coord_arr = np.array(coord_xyz)
    cri_arr = ((coord_arr - origin_arr) @ np.linalg.inv(direction_arr)) / voxSize_arr
    cri_arr = np.round(cri_arr)
    return IrcTuple(int(cri_arr[2]), int(cri_arr[1]), int(cri_arr[0]))