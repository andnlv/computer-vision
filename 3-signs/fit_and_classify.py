
import numpy as np
import skimage.io as io

from skimage.filters import sobel_h, sobel_v
from skimage.transform import resize

from sklearn.svm import LinearSVC
from itertools import product


def get_grads(img):
    weights = np.array([0.299, 0.587, 0.114])
    brightness = img @ weights
    grad_y = sobel_h(brightness)
    grad_x = sobel_v(brightness)
    return np.sqrt(grad_y ** 2 + grad_x ** 2), np.arctan2(grad_y, grad_x)


def make_hist(grad_norms, grad_angles, bins):
    limits = np.linspace(-np.pi, np.pi, bins + 1).reshape(bins + 1, 1, 1) + np.pi / bins
    return np.sum(((grad_angles >= limits[:-1]) & (grad_angles < limits[1:])) * grad_norms, axis=(1, 2))


def build_cells(grad_norms, grad_angles, cell_shape=(8, 8), bins=8):
    hists = np.empty([*np.array(grad_norms.shape) // cell_shape, bins])
    cell_slice = lambda i, ax: slice(cell_shape[ax] * i, cell_shape[ax] * (i + 1))
    for i, j in product(range(hists.shape[0]), range(hists.shape[1])):
        hists[i, j] = make_hist(
            grad_norms[cell_slice(i, 0), cell_slice(j, 1)], 
            grad_angles[cell_slice(i, 0), cell_slice(j, 1)],
            bins,
        )
    return hists


def build_blocks(cells, block_shape=(4, 4), step=(2, 2)):
    if step is None:
        step = block_shape
    result_shape = (np.array(cells.shape[:2]) - block_shape) // step + 1
    result = np.empty([*result_shape, np.prod(block_shape) * cells.shape[-1]])
    for i, j in product(range(result.shape[0]), range(result.shape[1])):
        result[i, j] = cells[
            i * step[0]: i * step[0] + block_shape[0], 
            j * step[1]: j * step[1] + block_shape[0]
        ].reshape(-1)
        result[i, j] /= np.sqrt((result[i, j] ** 2).sum() + 1e-5)
    return result.reshape(-1)


def extract_hog(image):
    image = resize(image, (64, 64))
    grad_norms, grad_angles = get_grads(image)
    cells = build_cells(grad_norms, grad_angles)
    return build_blocks(cells)


def fit_and_classify(X_train, y_train, X_test):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model.predict(X_test)
