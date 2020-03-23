# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:56:02 2020

@author: HP
"""

def extract_features(file_name):
   max_pad_len=182
   try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width=max_pad_len-mfccs.shape[1]
        mfccs=np.pad(mfccs,pad_width=((0,0),(0,pad_width)),mode='constant')
       
   except Exception as e:
        print("Error encountered while parsing file: ", file_name,e)
        return None
     
   return mfccs
   
   
# Load various imports
import numpy as np
import pandas as pd
import os
import librosa
import soundfile
import glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

fulldatasetpath = 'E:/dl projects/datasets/actorsss/'
features = []
dir= os.listdir(fulldatasetpath)
dir
path = []
actor = []

for i in dir:
    file_name =os.listdir(fulldatasetpath+i)
    for f in file_name:
        if i == 'Actor_01':
            actor.append('Actor_1')
        elif i == 'Actor_02':
            actor.append('Actor_2')
        elif i == 'Actor_03':
            actor.append('Actor_3')
        elif i == 'Actor_04':
            actor.append('Actor_4')

        else:
            actor.append('Unknown')
        path.append(fulldatasetpath + i + "/" + f)
df = pd.DataFrame(actor,columns=['name'])
labels = list(df['name'].unique())
for index, row in df.iterrows():
    class_label = row["name"]
    for p in path:
        data = extract_features(p)
    features.append([data, class_label])
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# split the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

num_rows = 40
num_columns = 182
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs =1
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='weights.best.basic_cnn.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)