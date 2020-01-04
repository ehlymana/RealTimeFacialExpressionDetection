import cv2
import os
from moviepy.editor import *
import csv

# get all images from dataset

datasetDirectory = os.getcwd() + "/AMFEDPLUS/AMFEDPLUS_Distribution/Videos - FLV (labeled)"
allVideos = os.listdir(datasetDirectory)

# initialize list of labels

labelsDirectory = os.getcwd() + "/AMFEDPLUS/AMFEDPLUS_Distribution/Baseline Classification"
labelsList = []

# initialize the number of images

number = 1

# extract frames of interest from videos

for i in range(0, len(allVideos)):

    print("\nProcessing video no. " + str(i + 1) + "/" + str(len(allVideos)) + "...\n")

    # get frames of interest from dataset

    framesOfInterest = []

    # get labels of different emotions from dataset

    labelsVideo = []

    # get focal points for all frames of interest from dataset

    focalPoints = []

    # iterate through all label rows

    with open(labelsDirectory + "/" + allVideos[i][:-4] + "_METRICS.csv", newline = '') as csvFile:

        csvReader = csv.reader(csvFile, delimiter=',')

        # skip header row

        next(csvReader)

        # add frame and label data for images

        for row in csvReader:

            labelsFrame = []

            # skip frames that have almost the exactly same labels

            if len(labelsVideo) > 0 and len(framesOfInterest) < 20:

                sameRows = 0

                for j in range(1, len(row)):

                    # no face on image

                    if (row[j]) == "TF":
                        sameRows = len(row) - 1
                        break

                    if abs(float(row[j]) - float(labelsVideo[len(labelsVideo) - 1][j - 1])) < 0.1:
                        sameRows += 1

                 # process the frame only if at least four labels differ by 0.1

                if sameRows < len(row) - 4 :

                    framesOfInterest.append(row[0])

                    for j in range(1, len(row)):

                        labelsFrame.append(row[j])

            # first frame - add it immediately

            elif len(framesOfInterest) < 20:
                framesOfInterest.append(row[0])

                for j in range(1, len(row)):

                    # no face on image - no activation

                    if row[j] == "TF":
                        labelsFrame.append(0)

                    else:
                        labelsFrame.append(row[j])

            # append label list

            if len(labelsFrame) > 0:
                labelsVideo.append(labelsFrame)

    # read FLV video from dataset

    video = VideoFileClip(datasetDirectory + "/" + allVideos[i])

    print("Found " + str(len(framesOfInterest)) + " images!")

    # turn all frames of interest into pictures

    for j in range(0, len(framesOfInterest)):
        try:
            # reading from frame

            image = video.get_frame(float(framesOfInterest[j]) / video.fps)
            # check if video directory exists

            directory = os.getcwd() + '/data/images'
            if not os.path.exists(directory):
                os.makedirs(directory)

            # create image from frame

            name = directory + '/image' + str(number) + '.jpg'
            number += 1

            # uncomment if you want to get information about every photo

            # print('Creating...' + name)

            cv2.imwrite(name, image)

        # frame does not exist - end image extraction for video

        except:
            labelsVideo = labelsVideo[:-(len(framesOfInterest) - j)]

            break

    labelsList.append(labelsVideo)

print("\nWriting all labels to a CSV file...")

# write all labels to a CSV file

with open(os.getcwd() + '/data/labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # write labels header

    writer.writerow(["AU2", "AU4", "AU5", "AU17", "AU18", "AU26", "Smile", "Smirk"])

    # write labels

    for i in range(0, len(labelsList)):
        writer.writerows(labelsList[i])

print("\nFinished writing all labels to a CSV file!")