import collections
import copy
import datetime
import gc
import time

# import torch
import numpy as np

from utill.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

"""
To go from voxel indices to coordinates, we need to follow these four steps in order:
1 Flip the coordinates from IRC to CRI, to align with XYZ.
2 Scale the indices with the voxel sizes.
3 Matrix-multiply with the directions matrix, using @ in Python.
4 Add the offset for the origin.

"""

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


def importstr(module_str, from_=None):
    
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))

    return module


def enumerateWithEstimate(
        iter,
        desc_str,
        start_index=0,
        print_index=4,
        backoff=None,
        iter_len=None,
):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).
    However, the side effects (logging, specifically) are what make the
    function interesting.
    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.
    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.
    :param start_index: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.
        NOTE: Using `start_index` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.
        This parameter defaults to `0`.
    :param print_index: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_index` not be less than `start_index` times
        `backoff`, since `start_index` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.
        `print_index` defaults to `4`.
    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.
        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.
    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.
    :return:

    """
    
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_index < start_index * backoff:
        print_index *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))

    start_ts = time.time()


    for (current_index, item) in enumerate(iter):
        yield (current_index, item)

        if current_index == print_index:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_index - start_index + 1)
                            * (iter_len-start_index)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_index,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_index *= backoff

        if current_index + 1 == start_index:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))

def prhist(ary, prefix_str=None, **kwargs):
    if prefix_str is None:
        prefix_str = ''
    else:
        prefix_str += ' '

    count_ary, bins_ary = np.histogram(ary, **kwargs)
    for i in range(count_ary.shape[0]):
        print("{}{:-8.2f}".format(prefix_str, bins_ary[i]), "{:-10}".format(count_ary[i]))
    print("{}{:-8.2f}".format(prefix_str, bins_ary[-1]))
