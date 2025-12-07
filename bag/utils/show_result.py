#  -*- coding: utf-8 -*-
'''
author: xuechao.wang@ugent.be
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
import plotly.graph_objects as go



def moving_average(data, window_size):
    '''
        Moving average is used to smooth the data between each segment to prevent sudden changes.
        '''
    pad_size = window_size // 2
    if window_size % 2 == 0:
        padded_data = np.pad(data, (pad_size, pad_size-1), mode='edge')
    else:
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data



def saliency_map(mask):
    '''
        Map the weight or shap value to the corresponding position in the colorbar, ranging from 0 to 1
        Args:
            mask represents weight or shap value.
        '''
    weight_signal = [i[0] for i in mask]

    impact = moving_average(weight_signal, int((len(mask)*0.1)*0.5))

    # # 绘制原始组合信号和平滑后的信号
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(impact)), weight_signal, label='Combined Signal', alpha=0.5)
    # plt.plot(range(len(impact)), impact, label='Smoothed Signal', linewidth=2)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Smoothing Combined Step Signals with Wavelet Transform')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    colors_map = []
    min_value = np.min(impact)
    max_value = np.max(impact)
    if max_value > 0 and min_value < 0:
        for i in np.arange(len(impact)):
            weight = impact[i]
            if weight >= 0:
                new_weight = 0.5 + (weight / max_value)*0.5
                colors_map.append(new_weight)
            else:
                new_weight = 0.5 - (weight / min_value)*0.5
                colors_map.append(new_weight)
    elif max_value <= 0 and min_value < 0:
        for i in np.arange(len(impact)):
            weight = impact[i]
            new_weight = 0.5 - (weight / min_value)*0.5
            colors_map.append(new_weight)
        colors_map[0] = 1
    elif min_value >= 0 and max_value > 0:
        for i in np.arange(len(impact)):
            weight = impact[i]
            new_weight = 0.5 + (weight / max_value)*0.5
            colors_map.append(new_weight)
        colors_map[0] = 0
    elif min_value==0 and max_value==0:
        for i in np.arange(len(impact)):
            new_weight = 0.5
            colors_map.append(new_weight)
        colors_map[0] = 1
        colors_map[1] = 0

    return colors_map



def plot_result(normalized_pc, colors_map, mask, out_path, file_name, data_pattern, if_save=False):
    '''
    Display and save the results, where the colors represent the weights
    '''
    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.3
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T

    # Using the coolwarm colormap
    cmap = cm.get_cmap('coolwarm')

    colors = cmap(colors_map)

    PD_colors_map = [0.5 if i < 0.5 else i for i in colors_map]
    PD_colors = cmap(PD_colors_map)

    HC_colors_map = [0.5 if i > 0.5 else i for i in colors_map]
    HC_colors = cmap(HC_colors_map)

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(14, 7))
    plt.axis('off')

    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

    # 3D Plot (Left side)
    ax3d = fig.add_subplot(gs[:, :2], projection='3d')
    sc = ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=40)
    ax3d.scatter(points[:, 0], points[:, 1], np.zeros_like(points[:, 2]), c=colors, marker='o', s=5, alpha=0.5)
    ax3d.view_init(elev=15, azim=-45)
    ax3d.set_xlabel('X', fontsize=20)
    ax3d.set_ylabel('Y', fontsize=20)
    ax3d.set_zlabel('Z', fontsize=20)
    ax3d.tick_params(axis='x', labelsize=12)
    ax3d.tick_params(axis='y', labelsize=12)
    ax3d.tick_params(axis='z', labelsize=12)

    # 在3D子图下方添加一行文字说明
    ax3d.text2D(0.5, -0.05,
                'Note: Red indicates a tendency towards patients, blue towards healthy individuals;\n color intensity reflects the degree of inclination.',
                horizontalalignment='center', verticalalignment='center', transform=ax3d.transAxes, fontsize=18)

    # PD 2D Plot (Top-right)
    ax_pd = fig.add_subplot(gs[0, 2])
    ax_pd.scatter(points[:, 0], points[:, 1], c=PD_colors, marker='o', s=20)
    ax_pd.grid(True, color='gray', alpha=0.5)
    ax_pd.set_xlabel('X', fontsize=20)
    ax_pd.set_ylabel('Y', fontsize=20)
    ax_pd.set_aspect('equal', 'box')
    ax_pd.set_xticklabels([])
    ax_pd.set_yticklabels([])

    # HC 2D Plot (Bottom-right)
    ax_hc = fig.add_subplot(gs[1, 2])
    ax_hc.scatter(points[:, 0], points[:, 1], c=HC_colors, marker='o', s=20)
    ax_hc.grid(True, color='gray', alpha=0.5)
    ax_hc.set_xlabel('X', fontsize=20)
    ax_hc.set_ylabel('Y', fontsize=20)
    ax_hc.set_aspect('equal', 'box')

    ax_hc.set_xticklabels([])
    ax_hc.set_yticklabels([])
    # 在整体图片下方添加文字说明


    if if_save:
        plt.savefig(os.path.join(out_path, f'{file_name}-{str(data_pattern)}-explain.jpg'), dpi=500)
        np.save(os.path.join(out_path, f'{file_name}-{str(data_pattern)}-weight.npy'), mask)
    else:
        plt.show()
    plt.close()



