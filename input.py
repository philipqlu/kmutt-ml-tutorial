import numpy as np
import os

################################################################################
# TEAM A Task: Load and preprocess the images from the directory on the disk.  #
################################################################################

def load_data(train_folder, test_folder):
    X_train, y_train, X_test, y_test = [], [], [], []

    # Associate images and their labels using a dictionary

    data = []
    with open('labels.csv', 'r+') as label_file:
        next(label_file)
        for line in label_file:
            temp_data = line.split(',')
            data.append([temp_data[0],int(temp_data[1])])
    data_dicts = dict({d[0]:d[1] for d in data})

    # We loop through the images in the training folder

    for image_file in os.listdir(train_folder):
        image_path = os.path.join(train_folder, image_file)  # get the path of each image so we can open it
        image_raw = Image.open(image_path).convert('L').resize((32,32), Image.ANTIALIAS) # resize and grayscale
        image_flat = np.asarray(image_raw.getdata())
        X_train.append(image_flat)
        y_train.append(data_dicts[image_file])

    ################################################################################
    # TODO:                                                                        #
    # Write code that loads the images and labels from the test folder like above  #
    # Append the image array to X_test and the label to y_test                     #
    # Hint: You only have to change four lines from the loop above!                #
    ################################################################################


    ################################################################################
    # TODO:                                                                        #
    # Compute the mean image of X and subtract that from train and test images     #
    ################################################################################


    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################

    return X_train, y_train, X_test, y_test
