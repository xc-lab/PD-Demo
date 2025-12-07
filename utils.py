#  -*- coding: utf-8 -*-
'''
author: xuechao.wang@ugent.be
'''
import numpy as np
import torch
import math


def curvature(x, y):
    # Calculating curvature
    dx_dt = np.gradient(x)  # The first derivative of x
    dy_dt = np.gradient(y)  # The first derivative of y
    d2x_dt2 = np.gradient(dx_dt)  # The second derivative of x
    d2y_dt2 = np.gradient(dy_dt)  # The second derivative of y

    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** (3 / 2)+(1e-6))
    return curvature


def derived_feature(data, dataset='ParkinsonHW'):
    '''
    Calculate derived features such as velocity, acceleration, jerk, etc.
    :param data:
    :return:
    '''
    x_coord = data[:, 0] - np.mean(data[:, 0])
    y_coord = data[:, 1] - np.mean(data[:, 1])
    z_coord = data[:, 2] - np.mean(data[:, 2])
    p_coord = data[:, 3]
    g_coord = data[:, 4]
    time_coord = data[:, 5]

    x_coord_diff = np.diff(x_coord, n=1, axis=-1, append=0)
    y_coord_diff = np.diff(y_coord, n=1, axis=-1, append=0)
    z_coord_diff = np.diff(z_coord, n=1, axis=-1, append=0)
    p_coord_diff = np.diff(p_coord, n=1, axis=-1, append=0)
    g_coord_diff = np.diff(g_coord, n=1, axis=-1, append=0)
    time_coord_diff = np.diff(time_coord, n=1, axis=-1, append=0)

    radius_coord = np.array(
        [pow(pow((x_coord[idx]), 2) + pow((y_coord[idx]), 2), 0.5) for idx in range(len(x_coord_diff))])
    angle_coord = np.array([math.atan(y_coord[idx] / x_coord[idx]) for idx in range(len(x_coord_diff))])
    curva_coord = curvature(x_coord, y_coord)

    if dataset == 'ParkinsonHW':
        location_diff_list = np.array([pow(pow((x_coord_diff[idx]), 2) + pow((y_coord_diff[idx]), 2), 0.5) for idx in range(len(x_coord_diff))])
        velocity_list = location_diff_list
        velocity_diff_list = np.array(np.diff(velocity_list, n=1, axis=-1, append=0))
        acceleration_list = velocity_diff_list
        acceleration_diff_list = np.array(np.diff(acceleration_list, n=1, axis=-1, append=0))
        jerk_list = acceleration_diff_list
    else:
        print('%s is a new dataste!' % (dataset))

    data = np.hstack((data,
                      velocity_list.reshape((len(jerk_list), -1)),
                      acceleration_list.reshape((len(jerk_list), -1)),
                      jerk_list.reshape((len(jerk_list), -1)),
                      x_coord_diff.reshape(len(jerk_list), -1),
                      y_coord_diff.reshape(len(jerk_list), -1),
                      z_coord_diff.reshape(len(jerk_list), -1),
                      p_coord_diff.reshape(len(jerk_list), -1),
                      g_coord_diff.reshape(len(jerk_list), -1),
                      time_coord_diff.reshape(len(jerk_list), -1),
                      radius_coord.reshape(len(jerk_list), -1),
                      angle_coord.reshape(len(jerk_list), -1),
                      curva_coord.reshape(len(jerk_list), -1))
                     )
    return data[50:-50, :]



def data_reading(path, dataset, idx):
    '''
    Derived features (such as speed, acceleration, jerk, etc.) are calculated on each individual stroke and then concatenated to the x, y, a, l, p, t features, and the last column represents the index of the stroke.
    :param path:
    :param dataset:
    :return:

    '''
    index = ['x', 'y', 'z', 'p', 'g', 't', 'id']

    if dataset == 'ParkinsonHW':
        # load data
        test_data = np.loadtxt(path, dtype=np.float32, delimiter=';')
        # Intercept the drawing shapes that meet the plot pattern
        pattern_data = test_data[np.where(test_data[:, -1] == idx), :-1][0]
        L, _ = pattern_data.shape
        if L > 0:
            data = derived_feature(pattern_data, dataset)  # Add derived features such as velocity, acceleration, and jerk
            m, n = data.shape
        else:
            data = np.array([])
            m = 0
            print('%s do not include %d pattern data.' % (path, idx))

    return data, m


def pc_normalize(data):
    '''    Normalize the point cloud fragments
    '''
    temp_data = data.copy()

    xy_point = data[:, :2]
    xy_point = (xy_point - np.mean(xy_point, axis=0)) / np.max(
        np.sqrt(np.sum(np.power((xy_point - np.mean(xy_point, axis=0)), 2), axis=1)))
    temp_data[:, :2] = xy_point

    feature_point = data[:, 2:]
    temp_data[:, 2:] = (feature_point - np.mean(feature_point, axis=0)) / np.std(feature_point, axis=0)

    return temp_data

def get_testing_patches(data, window_size, stride_size):
    patch_dataset = []
    if data.shape[0] > window_size:
        data_idxs = np.arange(0, data.shape[0] - window_size, stride_size)
        for p in data_idxs:
            patch = data[p:p + window_size, :]
            nor_patch = pc_normalize(patch)  # normalize
            patch_dataset.append(nor_patch)
    return patch_dataset




def pc_normalize_all(data):
    '''
    Normalize the point cloud
    '''
    temp_data = data.copy()

    xy_point = data[:, :2]
    xy_point = (xy_point - np.mean(xy_point, axis=0)) / np.max(
        np.sqrt(np.sum(np.power((xy_point - np.mean(xy_point, axis=0)), 2), axis=1)))
    temp_data[:, :2] = xy_point

    feature_point = data[:, 2:]
    temp_data[:, 2:] = (feature_point - np.min(feature_point, axis=0)) / (np.max(feature_point, axis=0) - np.min(feature_point, axis=0))

    return temp_data




def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__=='__main__':
    '''
    test code
    '''

    # path = '../data\ParkinsonHW/raw_data/KT/KT4/KT-4_F4A1ADA3_spiral_2017-11-22_10_59_23___43b1be2b11c6427aae4d838609e73450.json'
    # data = data_reading(path, "DraWritePD")
    # generate_img(data[:-1200,:], 'a', 'aa')



