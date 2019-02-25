import os

import numpy as np
import pandas as pd
import skimage.io as io
from skimage.transform import resize
import keras
import keras.layers as L
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

img_shape = (100, 100)


def train_detector(gt, img_dir, fast_train=True):
    
    filenames = os.listdir(img_dir)
    
    shapes = []
    for filename in filenames:
        shapes.append(io.imread(os.path.join(img_dir, filename)).shape[:2])
    shapes = np.array(shapes)
    
    df = pd.DataFrame(columns=['filename'] + sum([['x' + str(i), 'y' + str(i)] for i in range(1, 15)], []))
    df['filename'] = filenames
    df[sum([['x' + str(i), 'y' + str(i)] for i in range(1, 15)], [])] = np.array([*map(lambda name: gt[name], df['filename'])])
    
    df[['x' + str(i) for i in range(1, 15)]] = (df[['x' + str(i) for i in range(1, 15)]].values / shapes[:, 1].reshape(-1, 1)) * 100
    df[['y' + str(i) for i in range(1, 15)]] = (df[['y' + str(i) for i in range(1, 15)]].values / shapes[:, 0].reshape(-1, 1)) * 100
    df[[*df][1:]] = df[[*df][1:]].astype(float)
    batch_size = 50
    train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=df, 
        directory=img_dir,
        x_col='filename',
        y_col= [*df][1:],
        target_size=img_shape,
        batch_size=batch_size,
        class_mode='other'
    )

    model = keras.models.Sequential([
        L.Convolution2D(filters=48, kernel_size=5, activation='relu', input_shape=(*img_shape, 3)),
        L.Convolution2D(filters=64, kernel_size=5),
        L.BatchNormalization(),
        L.Activation('relu'),
        L.MaxPooling2D(),
        L.Convolution2D(filters=96, kernel_size=3, activation='relu'),
        L.Convolution2D(filters=128, kernel_size=3),
        L.BatchNormalization(),
        L.Activation('relu'),
        L.MaxPooling2D(),
        L.Convolution2D(filters=192, kernel_size=3, activation='relu'),
        L.Convolution2D(filters=256, kernel_size=3),
        L.BatchNormalization(),
        L.Activation('relu'),
        L.MaxPooling2D(),
        L.Flatten(),
        L.Dense(units=128, activation='relu'),
        L.Dense(units=64, activation='relu'),
        L.Dense(units=28),
    ])
    
    model.compile('adam', loss='mse')
    model.fit_generator(train_generator, epochs=1, steps_per_epoch=10)
    
    
def detect(model, test_img_dir):
    results = {}
    for filename in tqdm(os.listdir(test_img_dir)):
        img = io.imread(os.path.join(test_img_dir, filename))
        result = model.predict(resize(img, [*img_shape, 3])[np.newaxis])[0]
        result[0::2] *= img.shape[1] / 100
        result[1::2] *= img.shape[0] / 100
        results[filename] = result
    return results
 
    
 