from keras.applications import InceptionV3
from keras import layers
from keras import models
from keras import optimizers
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

# define model architecture

model = models.Sequential()

model.add(layers.Dense(64, activation = 'relu', input_shape = (68 * 2,)))

model.add(layers.Dense(64, activation = 'relu'))

model.add(layers.Dense(1))

# compile model

model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])

# import all features

cwd = os.getcwd() + '/data'
train_directory = cwd + '/train'
validation_directory = cwd + '/validation'
test_directory = cwd + '/test'
train_image_names = os.listdir(train_directory)
validation_image_names = os.listdir(validation_directory)
test_image_names = os.listdir(test_directory)

train_features = []
validation_features = []
test_features = []

print("Reading features...\n")

with open(cwd + "/landmarks.csv", newline = '') as csvFile:
    csvReader = csv.reader(csvFile, delimiter = ',')

    next(csvReader)

    counter = 0

    # append labels to different rows

    for row in csvReader:
        if counter < len(train_image_names):
            train_features.append(row)
        elif counter < len(train_image_names) + len(validation_image_names):
            validation_features.append(row)
        else:
            test_features.append(row)
        counter += 1

# import all labels

train_labels = []
validation_labels = []
test_labels = []

print("Reading feature labels...\n")

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
            test_labels.append(int(label))
        counter += 1

# reshape data

train_features = np.asarray(train_features).astype(np.float)
validation_features = np.asarray(validation_features).astype(np.float)
test_features = np.asarray(test_features).astype(np.float)

train_labels = np.asarray(train_labels)
validation_labels = np.asarray(validation_labels)
test_labels = np.asarray(test_labels)

mean = np.mean(train_features, axis = 0)
deviation = np.std(train_features, axis = 0)
train_features = train_features - mean
train_features = train_features / deviation

mean = np.mean(validation_features, axis = 0)
deviation = np.std(validation_features, axis = 0)
validation_features = validation_features - mean
validation_features = validation_features / deviation

mean = np.mean(test_features, axis = 0)
deviation = np.std(test_features, axis = 0)
test_features = test_features - mean
test_features = test_features / deviation

train_features = np.reshape(train_features, (train_features.shape[0], 68 * 2))
validation_features = np.reshape(validation_features, (validation_features.shape[0], 68 * 2))
test_features = np.reshape(test_features, (test_features.shape[0], 68 * 2))

# train model

print("Beginning model training...\n")

history = model.fit(train_features, train_labels, epochs = 100, validation_data = (validation_features, validation_labels),
                    batch_size = 1, verbose = 1)

print("Model training successfully completed !\n")

# save model for future use

model.save('model-features.h5')

# testing the network predictions

prediction = model.predict(test_features)

correct1 = 0
correct2 = 0

for i in range(0, len(prediction)):
    if round(prediction[i][0]) == test_labels[i]:
        correct1 += 1
    elif int(prediction[i][0]) == test_labels[i]:
        correct2 += 1


print("Accuracy of prediction on test subset: " + str(correct1 / len(test_features) * 100) + " % (round), " + str(correct2 / len(test_features) * 100) + ' % (int)')