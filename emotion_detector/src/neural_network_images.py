from keras.applications import InceptionV3
from keras import layers
from keras import models
from keras import optimizers
import os
import cv2
import csv
import numpy as np

# initialize neural network base for classifying images

convBaseImages = InceptionV3(include_top = False, weights = 'imagenet', input_shape = (320, 240, 3))

# define model architecture

model = models.Sequential()

convBaseImages.trainable = False
model.add(convBaseImages)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation = 'relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation = 'sigmoid'))

# compile model

rmsprop = optimizers.RMSprop(lr = 2e-5)
model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])

# import all images

cwd = os.getcwd() + '/data'
train_directory = cwd + '/train'
validation_directory = cwd + '/validation'
train_image_names = os.listdir(train_directory)
validation_image_names = os.listdir(validation_directory)

train_images = []
validation_images = []

print("Reading train images...\n")

for i in range(0, len(train_image_names)):
    img = cv2.imread(train_directory + '/' + train_image_names[i])

    if img.shape != (320, 240, 3):
        img = cv2.resize(img, (320, 240))

    train_images.append(img)

print("Reading validation images...\n")

for i in range(0, len(validation_image_names)):
    img = cv2.imread(validation_directory + '/' + validation_image_names[i])

    if img.shape != (320, 240, 3):
        img = cv2.resize(img, (320, 240))

    validation_images.append(img)

# import all labels

train_labels = []
validation_labels = []

print("Reading image labels...\n")

with open(cwd + "/labels.csv", newline = '') as csvFile:
    csvReader = csv.reader(csvFile, delimiter = ',')

    next(csvReader)

    counter = 0

    # decode classes from labels

    for row in csvReader:

        sum = 1e-5
        for j in range(0, len(row)):
            sum += float(row[j])

        labelPercentage = []

        for j in range(0, len(row)):
            labelPercentage.append(float(row[j]) / sum)

        label = np.argmax(labelPercentage)

        if counter < len(train_image_names):
            train_labels.append(int(label))
        elif counter < len(train_image_names) + len(validation_image_names):
            validation_labels.append(int(label))
        else:
            break
        counter += 1

# reshape data

train_images = np.asarray(train_images)
validation_images = np.asarray(validation_images)

train_labels = np.asarray(train_labels)
validation_labels = np.asarray(validation_labels)

train_images = np.reshape(train_images, (train_images.shape[0], 320, 240, 3))
validation_images = np.reshape(validation_images, (validation_images.shape[0], 320, 240, 3))

# train model

print("Beginning model training...\n")

history = model.fit(train_images, train_labels, epochs = 30, validation_data = (validation_images, validation_labels),
                    batch_size = 20, verbose = 1)

print("Model training successfully completed !\n")

# save model for future use

model.save('model-images.h5')