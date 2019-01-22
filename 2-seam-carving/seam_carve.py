import numpy as np
import skimage.io as io
from itertools import product


def get_energy_img(img, mask=None):
    weights = np.array([0.299, 0.587, 0.114])
    brightness = np.dot(img, weights)
    padded_img = np.pad(brightness, 2, 'edge')
    y_grads = (padded_img[2:] - padded_img[:-2])[1:-1, 2:-2]
    x_grads = (padded_img[:, 2:] - padded_img[:, :-2])[2:-2, 1:-1]
    energy = np.sqrt(y_grads ** 2 + x_grads ** 2)
    if mask is not None:
        energy += mask * energy.size * 256
    return energy


def build_dinamic_table(energy_img):
    dinamic_table = np.zeros([*energy_img.shape, 2])
    
    for j in range(energy_img.shape[1]):
        dinamic_table[0, j] = [energy_img[0, j], 0]
    for i, j in product(range(1, energy_img.shape[0]), range(0, energy_img.shape[1])):
        min_j = j + np.argmin(dinamic_table[i - 1, max(0, j - 1): min(energy_img.shape[1], j + 2), 0])
        if j != 0:
            min_j -= 1
        dinamic_table[i, j, 0] = energy_img[i, j] + dinamic_table[i - 1, min_j, 0]
        dinamic_table[i, j, 1] = min_j - j
    return dinamic_table


def find_min_curve(img, mask=None):
    dinamic_table = build_dinamic_table(get_energy_img(img, mask))
    min_curve = np.empty(img.shape[0], dtype=int)
    min_curve[-1] = np.argmin(dinamic_table[-1, :, 0])
    for i in range(img.shape[0] - 2, -1, -1):
        min_curve[i] = min_curve[i + 1] + dinamic_table[i + 1, min_curve[i + 1], 1]
    return min_curve


def delete_curve(img, curve):
    shape = list(img.shape)
    shape[1] -= 1
    result = np.empty(shape, dtype=img.dtype)
    for i, j in enumerate(curve):
        result[i] = np.concatenate([img[i,:j], img[i, j+1:]], axis=0)
    return result


def add_curve(img, curve):
    shape = list(img.shape)
    shape[1] += 1
    result = np.empty(shape, dtype=img.dtype)
    for i, j in enumerate(curve):
        result[i] = np.concatenate([img[i,:j], [(img[i, j] + img[i, min(j + 1, img.shape[1] - 1)]) // 2], img[i, j:]], axis=0)
    return result


def seam_carve(img, mode, mask=None):
    if 'vertical' in mode:
        img = img.transpose(1, 0, 2)
        if mask is not None:
            mask = mask.transpose(1, 0)
    curve = find_min_curve(img, mask)
    
    result_curve = np.zeros([img.shape[0], img.shape[1]])
    for i, j in enumerate(curve):
        result_curve[i, j] = 1
        
    if 'shrink' in mode:
        result_img = delete_curve(img, curve)
        result_mask = delete_curve(mask, curve) if mask is not None else None
    else:
        result_img = add_curve(img, curve)
        result_mask = add_curve(mask, curve) if mask is not None else None
    
    if 'vertical' in mode:
        img = img.transpose(1, 0, 2)
        if mask is not None:
            mask = mask.transpose(1, 0)
            result_mask = result_mask.transpose(1, 0)
        result_img = result_img.transpose(1, 0, 2)
        result_curve = result_curve.transpose(1, 0)
    return result_img, result_mask, result_curve
