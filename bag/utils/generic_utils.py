#  -*- coding: utf-8 -*-
'''
author: xuechao.wang@ugent.be
'''
import numpy as np


def segment_fn(data):
    '''
    Mask the data to prepare for the subsequent generation of neighbor data.
    Args:
        data: original signal

    Returns:
        fudged_data: Split the original data into equal segments, calculate the mean of each segment, and copy the mean to the corresponding position of the segment in the original data. Its length is the same as the original data.
        mask: It is a list that stores the index of the split segments. Its length is the same as the original data.
    '''

    m, _ = data.shape
    idxs = np.arange(0, m, int(m * 0.1)).tolist()
    mask = []
    fudged_data = np.ones(data.shape, dtype=np.float64)
    for i in np.arange(len(idxs)):
        if idxs[i] == idxs[-1]:
            patch_fudged_data = data[idxs[i]:, :]
            # np.tile(): Copies the array in the specified direction
            fudged_data[idxs[i]:, :] = np.tile(np.average(patch_fudged_data, axis=0),
                                                          (patch_fudged_data.shape[0], 1))
            n = m - idxs[i]
            patch_data = (np.ones(n) * i).tolist()
            mask += patch_data
        else:
            patch_fudged_data = data[idxs[i]:idxs[i+1], :]
            fudged_data[idxs[i]:idxs[i+1], :] = np.tile(np.average(patch_fudged_data, axis=0), (patch_fudged_data.shape[0], 1))
            n = idxs[i + 1] - idxs[i]
            patch_data = (np.ones(n) * i).tolist()
            mask += patch_data
    mask = np.array(mask, dtype=np.int64)
    return mask, fudged_data

