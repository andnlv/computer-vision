import os
from time import time

import numpy as np

from keras.models import load_model

from skimage.transform import rescale
from skimage import io


def adjust_size(img, size=224):
    scaled_img = rescale(img, size/max(img.shape))
    if len(scaled_img.shape) == 2:
        scaled_img = np.stack([scaled_img]*3).transpose(1, 2, 0)
    for i in [0, 1]:
        if scaled_img.shape[i] < size:
            result = np.zeros([size, size, 3])
            gap = (size - scaled_img.shape[i]) // 2
            subslice = [slice(gap, gap + scaled_img.shape[i]), slice(None)]
            result[subslice[i], subslice[1 - i]] = scaled_img
            return result
    return scaled_img


def load_images(path, names, size=224):
    result = np.empty([len(names), size, size, 3])
    for i, name in enumerate(names):
        result[i, :, :, :] = adjust_size(io.imread(os.path.join(path, name)), size=size)
    return result


def train_classifier(train_gt, train_img_dir, fast_train=False):
    return
    time_begin = time()
    np.random.seed(325)
    names = os.listdir(train_img_dir)
    
    batch_size = 10
    max_time = 5 if fast_train else 20
    max_epochs = 50
    
    model = load_model('birds_model.hdf5')

    for epoch_id in range(max_epochs):
        np.random.shuffle(names)
        for batch_id, batch_start in enumerate(range(0, len(names), batch_size)):
            print('epoch_id: {}, batch_id: {}'.format(epoch_id, batch_id))
            images = load_images(train_img_dir, names[batch_start: batch_start + batch_size])
            targets = [train_gt[name] for name in names[batch_start: batch_start + batch_size]]
            model.train_on_batch(images, targets)
            if time() - time_begin > max_time * 60:
                model.save('birds_model.hdf5')
                return
    model.save('birds_model.hdf5')


def classify(model, test_img_dir):
    names = os.listdir(test_img_dir)
    images = load_images(test_img_dir, names)
    predicts = model.predict(images).argmax(axis=1)
    return dict(zip(names, predicts))
