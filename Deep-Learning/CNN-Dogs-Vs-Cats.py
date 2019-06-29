import numpy as np 
import os
import cv2
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


# set the directory to the images of the kaggle dataset
datadir = "PetImages/"
# declare the categories to be dogs or cats
categories = ["Dog", "Cat"]
# size to resize image before passing to ConvNN
IMG_SIZE = 50


# create the training dataset
def create_training_data():
    training_data = []
    for category in categories:
        # create path to cats and dogs
        path = os.path.join(datadir, category)
        # create labels for pictures
        class_num = categories.index(category)

        # iterate over image in each path
        for img in tqdm(os.listdir(path)): 
            try: 
                # read image as grayscale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #resize image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # add data to training data as list to shuffle
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    return training_data

# create and shuffle dataset
training_data = create_training_data()
random.shuffle(training_data)

# create our features and labels
X = [feature for feature, _ in training_data]
y = [label for _, label in training_data]

# reshape list to appropiate array size
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X / 255.0 # normalize pixel values

# create log file for tensorboard for model analysis
NAME = "Cats-vs-dogs-CNN"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# create model
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# convert 3D features to a 1D array
model.add(Flatten()) 
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])

# fit the model to the dataset
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3,
          callbacks=[tensorboard])



