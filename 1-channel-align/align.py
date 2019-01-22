import numpy as np
import skimage.io as io
from itertools import product
from skimage.transform import rescale


def img_preprocess(image):
    height = image.shape[0] // 3
    cut_borders = lambda size: slice(size // 20, size // 20 * 19)
    return [
        img[cut_borders(img.shape[0]), cut_borders(img.shape[1])] 
        for img in (image[:height], image[height: 2 * height], image[2 * height: 3 * height])
    ]


def mse(image_1, image_2):
    return np.sum((image_1 - image_2) ** 2) / np.prod(image_1.shape)


def cross_cor(image_1, image_2):
    return np.sum(image_1 * image_2) / np.sqrt(np.sum(image_1 ** 2) * np.sum(image_2 ** 2))


def shift_images(image_1, image_2, shift):
    
    slices = np.array([
        [slice(None), slice(None)],
        [slice(None), slice(None)]
    ])
    
    for i in range(2):
        if shift[i] > 0:
            slices[0, i] = slice(-shift[i])
            slices[1, i] = slice(shift[i], None)
        elif shift[i] < 0:
            slices[0, i] = slice(-shift[i], None)
            slices[1, i] = slice(shift[i])
    
    return image_1[slices[0, 0], slices[0, 1]], image_2[slices[1, 0], slices[1, 1]]


def find_best_shift(image_1, image_2, metrics=None, max_shift=15, start_shift=None):
    
    if start_shift is None:
        start_shift = (0, 0)
    if metrics is None:
        metrics = mse
    
    best_metrics=np.inf
    y_range, x_range = [range(-max_shift + start, max_shift + 1 + start) for start in start_shift]
    
    for shift in product(y_range, x_range):
        shifted_1, shifted_2 = shift_images(image_1, image_2, shift)
        current_metrics = metrics(shifted_1, shifted_2)
        if current_metrics < best_metrics:
                best_shift = shift
                best_metrics = current_metrics
    
    return best_shift


def pyramid_recursive(images, max_size, **kwargs):
    if max(images[0].shape) > max_size:
        compresed_images = [rescale(img, 0.5) for img in images]
        approximate_shifts = 2 * pyramid_recursive(compresed_images, max_size, **kwargs)
        return np.array([
            find_best_shift(
                images[0], 
                images[1], 
                metrics=kwargs.get('metrics'), 
                max_shift=2, 
                start_shift=approximate_shifts[0]
            ),
            find_best_shift(
                images[2], 
                images[1], 
                metrics=kwargs.get('metrics'), 
                max_shift=2, 
                start_shift=approximate_shifts[1]
            ),
        ])        
    else:
        return np.array([
            find_best_shift(
                images[0], 
                images[1], 
                **kwargs
            ),
            find_best_shift(
                images[2], 
                images[1], 
                **kwargs
            ),
        ]) 

    
def join_images(images, shift_0, shift_2):
    
    top = max(0, shift_0[0], shift_2[0])
    bot = images[0].shape[0] + min(0, shift_0[0], shift_2[0])
    left = max(0, shift_0[1], shift_2[1])
    right = images[0].shape[1] + min(0, shift_0[1], shift_2[1])
    
    return np.array([
        images[2][top - shift_2[0] : bot - shift_2[0], left - shift_2[1]: right - shift_2[1]],
        images[1][top: bot, left: right],
        images[0][top - shift_0[0] : bot - shift_0[0], left - shift_0[1]: right - shift_0[1]],
    ]).transpose(1, 2, 0)
    
    
def align(image, g_coord, max_size=500, **kwargs):

    images = img_preprocess(image)
    shifts = pyramid_recursive(images, max_size)
    colored_img = join_images(images, shifts[0], shifts[1])
    b_coord = g_coord - shifts[0] - (image.shape[0] // 3, 0)
    r_coord = g_coord - shifts[1] + (image.shape[0] // 3, 0)
    return colored_img, b_coord, r_coord
