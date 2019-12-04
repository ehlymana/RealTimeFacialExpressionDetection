import cv2
import os
from moviepy.editor import *
import csv

# get all video names from dataset
datasetDirectory = os.getcwd() + "/AMFEDPLUS/AMFEDPLUS_Distribution/Videos - FLV (labeled)"
allVideos = os.listdir(datasetDirectory)

# initialize list of labels
labelsDirectory = os.getcwd() + "/AMFEDPLUS/AMFEDPLUS_Distribution/AU Labels"
labelsList = []

#initialize list of focal points

# extract frames of interest from videos
for i in range(0, len(allVideos)):
    # get frames of interest from dataset
    framesOfInterest = []

    # get labels of different emotions from dataset
    labelsVideo = []

    # iterate through all label rows
    with open(labelsDirectory + "/" + allVideos[i][:-4] + "-label.csv", newline = '') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')

        # skip header row
        next(csvReader)

        labelsFrame = []

        # add frame and label data for images
        for row in csvReader:
            framesOfInterest.append(row[0])

            for j in range(1, len(row)):
                labelsFrame.append(row[j])

            # append label list
            labelsVideo.append(labelsFrame)

    # read FLV video from dataset
    video = VideoFileClip(datasetDirectory + "/" + allVideos[i])

    # turn all frames of interest into pictures
    for j in range(0, len(framesOfInterest)):
        try:
            # reading from frame
            image = video.get_frame(float(framesOfInterest[j]))

            # check if video directory exists
            directory = os.getcwd() + '/data/video' + str(i)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # create image from frame
            name = directory + '/frame' + str(framesOfInterest[j]) + '.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, image)

        # frame does not exist - end image extraction for video
        except:
            labelsVideo = labelsVideo[:-(len(framesOfInterest) - j)]

            break

    labelsList.append(labelsVideo)

# write all labels to a CSV file
with open(os.getcwd() + 'labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # write labels header
    writer.writerow(["Smile", "AU04", "AU02", "AU15", "Trackerfail",
                     "AU18", "AU09", "negAU12", "AU10", "Expressive",
                     "Unilateral_LAU12", "Unilateral_RAU12","AU14", "Unilateral_LAU14", "Unilateral_RAU14",
                     "AU05", "AU17", "AU26", "Forward", "Backward"])

    # write labels
    writer.writerows(labelsList)