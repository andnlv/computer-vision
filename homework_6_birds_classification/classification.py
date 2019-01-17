import os
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator


def train_classifier(train_gt, train_img_dir, fast_train=False):
    return

                      
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
    steps = len(df.names) / batch_size
    predicts = model.predict_generator(generator, steps=steps, verbose=1)
    return dict(zip(names, map(np.argmax, predicts)))
    