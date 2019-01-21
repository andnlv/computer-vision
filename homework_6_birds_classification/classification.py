import os
import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications import Xception
import keras.layers as L


def train_classifier(train_gt, train_img_dir, fast_train=False):    
    names = os.listdir(train_img_dir)
    df = pd.DataFrame({'filename':names})
    df['class'] = [*map(lambda name: train_gt[name], df['filename'])]
    batch_size=10
    train_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.2,
        horizontal_flip=True
    ).flow_from_dataframe(
        directory=train_img_dir,
        dataframe=df,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        x_col='filename',
        y_col='class',
    )

    num_classes = 50
    xception = Xception()
    for layer in xception.layers: # first we optimize new layers only
        layer.trainable = False

    activation = xception.get_layer('block14_sepconv2_act').output
    pool = L.GlobalMaxPooling2D()(activation)
    dropout = L.Dropout(0.5)(pool)
    dense = L.Dense(200, activation='relu')(dropout)
    dense = L.Dense(num_classes, activation='softmax')(dense)
    model = keras.models.Model(inputs=xception.inputs, outputs=dense)
    model.compile(
        optimizer=Adam(lr=0.001), 
        loss='sparse_categorical_crossentropy', 
        metrics=['sparse_categorical_accuracy']
    )
    # localy this was done till convergence
    model.fit_generator(train_generator, steps_per_epoch= 10, verbose=1)
    
    # now optimize whole model
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=Adam(lr=0.0001), 
        loss='sparse_categorical_crossentropy', 
        metrics=['sparse_categorical_accuracy']
    )
    
    # localy this was done till convergence
    model.fit_generator(train_generator, steps_per_epoch= 10, verbose=1)
                      
def classify(model, test_img_dir, batch_size=25):
    names = os.listdir(test_img_dir)
    df = pd.DataFrame({'names': names})
    generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=df,
        x_col='names',
        y_col=None,
        directory=test_img_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        class_mode=None,
    )
    predicts = model.predict_generator(generator, steps=len(df.names) / batch_size, verbose=1)
    return dict(zip(generator.filenames, map(np.argmax, predicts)))
    