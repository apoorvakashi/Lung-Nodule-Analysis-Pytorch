import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from combining_data.combining_data import Ct, LunaDataset

def findPositiveSamples(start_index = 0, limit = 100):
    ds = LunaDataset()

    positiveSample_list = []
    for sample in ds.candidates_list:
        if sample.isNodule:
            # print(len(positiveSample_list), sample)
            positiveSample_list.append(sample)

        if len(positiveSample_list) > limit:
            break

    return positiveSample_list

def showCandidate(series_uid, batch_index = None, **kwargs):
    ds = LunaDataset(series_uid = series_uid, **kwargs)

    positive_index_list = [i for i, x in enumerate(ds.candidates_list) if x.isNodule]

    if batch_index is None:
        if positive_index_list:
            batch_index = positive_index_list[0]
        
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_index = 0

    ct = Ct(series_uid)
    ct_tensor, isNodule_tensor, series_uid, center_irc = ds[batch_index]
    ct_arr = ct_tensor[0].numpy()

    clim=(-1000.0, 300)
    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hunits_arr[int(center_irc[0])], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hunits_arr[:,int(center_irc[1])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hunits_arr[:,:,int(center_irc[2])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_arr[ct_arr.shape[0]//2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_arr[:,ct_arr.shape[1]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_arr[:,:,ct_arr.shape[2]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_arr[index], clim=clim, cmap='gray')


    print(series_uid, batch_index, bool(isNodule_tensor[1]), positive_index_list)