import os
import random
from glob import glob

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from keras.utils import to_categorical

"""
Trains best with H, W, C = 225, 225, 3
"""
H, W, C = 128, 128, 3
DATA_DIR = os.sep.join(["D:", "dev", "datasets", "17810_23812_bundle_archive", "chest_xray"])
TRAIN_DIR = DATA_DIR + os.sep + "train"
TEST_DIR = DATA_DIR + os.sep + "test"
VAL_DIR = DATA_DIR + os.sep + "val"

NORMAL_DIR = TRAIN_DIR + os.sep + "NORMAL"
PNEUMONIA_DIR = TRAIN_DIR + os.sep + "PNEUMONIA"
transformations = {
    "rotate": iaa.Affine(rotate=(-50, 30)),
    "contrast": iaa.GammaContrast(),
    "shear": iaa.Affine(shear=(0, 40)),
    "flipr": iaa.Fliplr(p=1.0)
}


def augment_dataframe():
    # To balance out data for Normal labels, augment with random transformations
    aug_train_data = list()
    for file in glob(NORMAL_DIR + os.sep + "*"):
        aug_train_data.append((file, 0, random.sample(list(transformations.keys()), k=2)))
        aug_train_data.append((file, 0, random.sample(list(transformations.keys()), k=2)))
    return pd.DataFrame(aug_train_data, columns=["image_path", "label", "aug"])


def get_training_df():
    # Step 1: Read training data for both labels
    train_data = list()
    for file in glob(NORMAL_DIR + os.sep + "*"):
        train_data.append((file, 0, None))

    for file in glob(PNEUMONIA_DIR + os.sep + "*"):
        train_data.append((file, 1, None))

    train_df = pd.DataFrame(train_data, columns=["image_path", "label", "aug"], index=None)
    labels_to_count = train_df.groupby(by="label").agg("count")
    print(labels_to_count)
    aug_train_data = augment_dataframe()
    train_df = pd.concat([train_df, aug_train_data])

    labels_to_count = train_df.groupby(by="label").agg("count")
    print(labels_to_count)

    train_df = train_df.sample(frac=1.).reset_index(drop=True)
    return train_df


def augment_image(img, aug_list):
    if aug_list is None:
        return img
    for aug in aug_list:
        img = transformations[aug].augment_image(img)
    return img


def batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def train_data_gen(batch_size):
    train_df = get_training_df()
    X_batch = np.zeros([batch_size, H, W, C])
    Y_batch = np.zeros([batch_size, 2])
    indices = np.arange(train_df.shape[0])

    while True:
        np.random.shuffle(indices)
        for b in batch(indices, batch_size):
            for i, j in enumerate(b):
                img_path, label, aug_list = train_df.iloc[j].values
                img = cv2.imread(img_path)
                img = cv2.resize(img, (H, W))
                img = img / 255
                img = augment_image(img, aug_list)
                X_batch[i] = img
                Y_batch[i] = to_categorical(label, num_classes=2)

            yield X_batch, Y_batch


def validation_data():
    val_data = list()
    for file in glob(VAL_DIR + os.sep + "NORMAL" + os.sep + "*"):
        val_data.append((file, 0))

    for file in glob(VAL_DIR + os.sep + "PNEUMONIA" + os.sep + "*"):
        val_data.append((file, 1))

    df = pd.DataFrame(val_data, columns=["image_path", "label"])
    val_img, val_label = list(), list()

    for idx, row in df.iterrows():
        img_path, label = row.image_path, row.label
        img = cv2.imread(img_path)
        img = cv2.resize(img, (H, W))
        img = img / 255
        label = to_categorical(label, num_classes=2)
        val_img.append(img)
        val_label.append(label)

    X, Y = np.array(val_img), np.array(val_label)
    print(f"Valid Images Shape : {X.shape}")
    print(f"Valid Labels Shape : {Y.shape}")
    return X, Y
