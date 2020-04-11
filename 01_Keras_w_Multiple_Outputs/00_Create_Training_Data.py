import numpy as np
import os
from PIL import Image
import cv2
import pandas as pd


"""
This script takes images/labels from disk and writes to numpy files for training.
The input images are frames from a car camera.
The first output is another image that only contains the labeled lanes.
The second output are lane types read from a csv file (labels.csv).
The script produces 3 files: 
    x_data.npy - Input images stored as numpy arrays
    y_data.npy - Output images/labels containing only the lanes (1st output)
    y_label_data.npy - Additional output labels of lane types (2nd output)
"""

# Helper
def load_image_file(path_to_image):
    img = Image.open(path_to_image).convert('RGB')
    w, h = img.size
    return img, w, h


# Helper
def _get_bottom_roi(img):
    # Crop bottom half of image
    height = np.size(img, 0)
    width = np.size(img, 1)
    return img[int(height / float(2)):height, 0:width]


# Main function
def create_numpy_lane_dataset(path_to_input_images, path_to_labeled_images, path_to_output_data):
    r"""
    This function creates a numpy dataset for lanes using pre labeled images and corresponding input images.
    The output will be cropped images saved as x & y numpy files.
    """

    x_data_path = os.path.join(path_to_output_data, 'x_data.npy')
    y_data_path = os.path.join(path_to_output_data, 'y_data.npy')
    y_label_data_path = os.path.join(path_to_output_data, 'y_label_data.npy')

    x_data = np.array([])
    y_data = np.array([])
    y_label_data = np.array([])
    image_labels = np.array([])

    # Open additional label file
    path_to_label_file = os.path.join(path_to_input_images, 'labels.csv')
    df_labels = pd.read_csv(path_to_label_file)
    df_labels.set_index('image', inplace=True)
    print(df_labels.head())

    # Convert labels to one hot
    df_labels1 = pd.get_dummies(df_labels['label_left_desc'], prefix='left_')
    df_labels2 = pd.get_dummies(df_labels['label_right_desc'], prefix='right_')

    # Concat all one hot labels into single dataframe
    df_labels = pd.concat([df_labels1, df_labels2], axis=1, sort=False)
    print(df_labels.head())

    # Create training data for input images (aka 'x')
    print('Converting input images to numpy training data...')
    index = 0
    for subdir, dirs, files in os.walk(path_to_input_images):
        for file in files:
            print(os.path.join(subdir, file))
            filepath = subdir + os.sep + file

            if (filepath.lower().endswith(".png") or filepath.lower().endswith(
                    ".jpg")):
                index += 1

                img, width, height = load_image_file(filepath)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

                # Crop image
                bottom_roi_img = _get_bottom_roi(np.array(img))

                # Add to numpy array
                num_rows = bottom_roi_img.shape[0]
                num_cols = bottom_roi_img.shape[1]
                num_channels = bottom_roi_img.shape[2]

                # Get additional label data
                df_record = df_labels.loc[str(file)]

                # Convert additional label to numpy
                np_one_hot = np.array(df_record)
                np_one_hot = np.reshape(np_one_hot, (1, np_one_hot.shape[0]))
                print(file)
                print(np_one_hot[0, :])

                # Append to training data
                if index == 1:
                    x_data = bottom_roi_img.reshape(1, num_rows, num_cols, num_channels)
                    image_labels = np_one_hot
                else:
                    x_data = np.vstack(
                        (x_data, bottom_roi_img.reshape(1, num_rows, num_cols, num_channels)))
                    image_labels = np.vstack((image_labels, np_one_hot))

                # Show image
                #cv2.imshow('Example', bottom_roi_img)
                #cv2.waitKey(100)

    print('Saving x data...')
    np.save(x_data_path, x_data)

    print('Saving y label data...')
    np.save(y_label_data_path, image_labels)

    # Create y data from labeled images
    print('Converting labeled images to numpy y data...')
    index = 0
    for subdir, dirs, files in os.walk(path_to_labeled_images):
        for file in files:
            print(os.path.join(subdir, file))
            filepath = subdir + os.sep + file

            if (filepath.lower().endswith(".png") or filepath.lower().endswith(
                    ".jpg")):
                index += 1

                img, width, height = load_image_file(filepath)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

                # Crop image
                bottom_roi_img = _get_bottom_roi(np.array(img))

                # Only preserve lane points == [1,1,1]
                lower_black = np.array([1, 1, 1], dtype="uint16")
                upper_black = np.array([1, 1, 1], dtype="uint16")
                bottom_roi_img = cv2.inRange(bottom_roi_img, lower_black, upper_black)
                bottom_roi_img = np.where(bottom_roi_img == 255, 1, bottom_roi_img)

                # Add to numpy array
                num_rows = bottom_roi_img.shape[0]
                num_cols = bottom_roi_img.shape[1]
                #num_channels = bottom_roi_img.shape[2]

                # Append to training data
                if index == 1:
                    # With color channel - DEPRECATED
                    #y_data = bottom_roi_img.reshape(1, num_rows, num_cols, num_channels)

                    # With normal shape - DEPRECATED
                    #y_data = bottom_roi_img.reshape(1, num_rows, num_cols)

                    # With flat shape
                    # Flatten images to single row before adding to training data
                    y_data = np.array(bottom_roi_img).flatten().reshape((1, (num_rows * num_cols)))
                else:
                    #y_data = np.vstack(
                        #(y_data, bottom_roi_img.reshape(1, num_rows, num_cols, num_channels)))
                    #y_data = np.vstack(
                        #(y_data, bottom_roi_img.reshape(1, num_rows, num_cols)))
                    temp_y_data = np.array(bottom_roi_img).flatten().reshape((1, (num_rows * num_cols)))

                    y_data = np.vstack((y_data, temp_y_data))

                # Show image
                #print(bottom_roi_img[50, :])
                #cv2.imshow('Example', bottom_roi_img)
                #cv2.waitKey(100)

    print('Saving y data...')
    np.save(y_data_path, y_data)

    print('Process Complete')


################
##### Main #####
################
# Set input data paths
base_path_of_image_data = r'.\Input_Data'   # Base path to input data
path_to_image_data_input = os.path.join(base_path_of_image_data, 'img') # Input images from car camera
path_to_image_data_labeled = os.path.join(base_path_of_image_data, 'masks_machine') # Labeled images of lanes

# Set output paths
training_data_output_path = r'.\Training_Data'  # Where the training data will be saved

# Create training data
create_numpy_lane_dataset(path_to_image_data_input, path_to_image_data_labeled, training_data_output_path)
print('Process Complete')
