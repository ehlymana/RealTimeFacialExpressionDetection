import cv2
import os
from moviepy.editor import *
import csv
import dlib

# define the detectors for facial landmark points

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# get all images from the dataset

datasetDirectory = os.getcwd() + "/data/images"
allImages = os.listdir(datasetDirectory)

# keep the order of images the same as when extracting the labels

allImages.sort(key = len)

# detect landmarks for all images

allLandmarks = []

for i in range(0, len(allImages)):

    print("Reading image no. " + str(i + 1) + "...\n")

    # read image

    img = cv2.imread(datasetDirectory + '/' + allImages[i])

    # convert RGB image to GS

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face on the image

    landmarks_list = []

    faces = detector(img_gray, 0)

    # no face found on image - all locations default to zero

    if len(faces) < 1:

        for j in range(0, 68 * 2):
            landmarks_list.append(0)

        allLandmarks.append(landmarks_list)
        continue

    face = faces[0]

    # detect facial landmarks

    landmarks = predictor(img_gray, face)

    # add facial landmarks (X, Y) to the list of landmarks

    for j in range(0, landmarks.num_parts):
        landmarks_list.append(landmarks.part(j).x)
        landmarks_list.append(landmarks.part(j).y)

    # append the overall landmark list

    allLandmarks.append(landmarks_list)

print("\nWriting all landmarks to a CSV file...")

# write all labels to a CSV file

with open(os.getcwd() + '/data/landmarks.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # write landmarks header

    header = []

    counter = 0
    for i in range(0, 68 * 2):
        if i % 2 == 0:
            header.append('f' + str(counter + 1) + 'x')
        else:
            header.append('f' + str(counter + 1) + 'y')
            counter += 1

    writer.writerow(header)

    # write labels

    writer.writerows(allLandmarks)

print("\nFinished writing all landmarks to a CSV file!")