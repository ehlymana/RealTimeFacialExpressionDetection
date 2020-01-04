import os
import shutil

# the directory of the full dataset

base_dir = os.getcwd() + '/data'

original_dataset_dir = base_dir + '/images'

# create the directories for the train, validation and test subsets

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

allImages = os.listdir(original_dataset_dir)

print('Starting image copying...')

for i in range(0, len(allImages)):

    src = os.path.join(original_dataset_dir, allImages[i])

    # define different destination directories for different parts of the dataset

    if i < 0.75 * len(allImages):
        dst = os.path.join(train_dir, allImages[i])

    elif i < 0.9 * len(allImages):
        dst = os.path.join(validation_dir, allImages[i])

    else:
        dst = os.path.join(test_dir, allImages[i])

    shutil.copyfile(src, dst)

print('\nFinished copying images!')