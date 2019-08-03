#coding: utf-8
#http://hikm.hatenablog.com/entry/20090206/1233950923
#TODO: http://aidiary.hatenablog.com/entry/20161030/1477830597
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import os
from sklearn.model_selection import train_test_split
#import autokeras as ak
from sklearn.utils import shuffle
import cv2
#from keras.preprocessing import image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta, Adam
from keras import optimizers
from tqdm import tqdm
from sklearn import preprocessing
import glob

print(os.environ['HOME'])
common_data_dir = os.environ['HOME']+"/DATA"

IMAGE_DIR_PATH = "/home/ueno/DATA/TestDataset/MyImageDataset/*/"

IMAGE_DIRS = glob.glob(IMAGE_DIR_PATH)

img = mpimg.imread('/home/ueno/DATA/TestDataset/MyImageDataset/22kohm/000000.png')

orgHeight, orgWidth = img.shape[:2]
size = (int(orgHeight/6), int(orgWidth/6))
print(size)
import numpy as np

def load_dataset():
    dataset_img = []
    labels = []
    idx=0
    label_idx = 0

    print(IMAGE_DIRS)
    for dir_path in IMAGE_DIRS:
        for img_name in tqdm(os.listdir(dir_path), unit='actions', ascii=True):
            print(img_name)
            img = cv2.imread("{imagedir}/{imgname}".format(imagedir=dir_path, imgname=img_name))
            img = cv2.resize(img, size)
            dataset_img.append(img)
            labels.append([1 if label_idx == list_idx else 0 for list_idx in range(0,len(IMAGE_DIRS))])
        label_idx +=1

    print("DATASET LOADED")
    print(labels)
    return np.array(dataset_img), np.array(labels)

dataset, label = load_dataset()

dataset = dataset / 255.0

dataset, label = shuffle(dataset,  label, random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(
    dataset, label, test_size=0.15)

del dataset
del label

model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
#model.add(Flatten())
GlobalAveragePooling2D()
model.add(Dense(100))
model.add(Dense(5, activation='softmax'))

class_weight={0:1.00,1:1.00}

opt = optimizers.rmsprop(lr=0.000001, decay=1e-6)#TODO 学習率大事！！大きすぎると学習がうまく行かないバグになる。
model.compile(loss="categorical_crossentropy",
                   optimizer=opt, #optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
                   metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model.fit(np.array(X_train), np.array(Y_train),
          batch_size=32,
          epochs=100,
          verbose=1,
          validation_split=0.05,
          class_weight = class_weight,
          shuffle=True,
          callbacks=[early_stopping])

result=model.predict(np.array(X_test))

sib=0
denominator=0
for pred, teacher in zip(result.tolist(),Y_test):#TODO resの中身がリストになっているので、添字0を指定。[[1.0], [1.0], [1.0]]
    denominator += 1
    print("pred:"+str(pred)+"   teacher:"+str(teacher))
    if round(pred[0]) == teacher:
        print(round(pred[0]))
        sib+=1


print("sib:"+str(sib)+"  denominator"+str(denominator))
print("ACCURACY:"+str(float(sib/denominator)))
