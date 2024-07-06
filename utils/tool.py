# -*- coding:utf-8 -*-
# @time: 2022/11/2 22:05
# @author: Metalope
import math
import random
import numpy as np
import torch


def seg_rect_intersect(rect, seg):
    # rect top-left, bottom-right
    rect_x_range = [min(rect[0][0], rect[1][0]), max(rect[0][0], rect[1][0])]
    rect_y_range = [min(rect[1][1], rect[0][1]), max(rect[1][1], rect[0][1])]

    # Is point in rect?
    for point in seg:
        if rect_x_range[0] <= point[0] <= rect_x_range[1] and rect_y_range[0] <= point[1] <= rect_y_range[1]:
            return True

    # two sides cross?
    y = []
    for x in rect_x_range:
        if seg[0][0]-seg[1][0] != 0:
            if min(seg[0][0], seg[1][0]) <= x <= max(seg[0][0], seg[1][0]) or \
                    (rect_x_range[0] <= seg[0][0] <= rect_x_range[1] and
                     rect_x_range[0] <= seg[1][0] <= rect_x_range[1]):
                y.append((x-seg[1][0])/(seg[0][0]-seg[1][0])*(seg[0][1]-seg[1][1])+seg[1][1])
        else:
            if rect_x_range[0] <= seg[0][0] <= rect_x_range[1] \
                    and max(seg[0][1], seg[1][1]) >= rect_y_range[0] and min(seg[0][1], seg[1][1]) <= rect_y_range[1]:
                return True
            else:
                return False

    for item in y:
        if rect_y_range[0] <= item <= rect_y_range[1]:
            return True
    if y:
        if max(y) >= rect_y_range[1] and min(y) <= rect_y_range[0]:
            return True

    return False


def set_seed(seed):
    print(f"Using seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def normalize(data, from_range, to_range):
    """
    normalize or denormalize

    Args:
        data: data needs to be mapped
        from_range: origin scale
        to_range: target scale

    Returns:
        mapped data

    """
    data_trans = (data - (from_range[0] + from_range[1]) * 0.5) / (from_range[1] - from_range[0]) * \
                 (to_range[1] - to_range[0]) + (to_range[0] + to_range[1]) / 2

    if data_trans < to_range[0]:
        data_trans = to_range[0]
    elif data_trans > to_range[1]:
        data_trans = to_range[1]

    return data_trans