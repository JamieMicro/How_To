import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model, load_model
import os
import cv2
from PIL import Image
import pandas as pd


"""
This script trains and saves an Keras model with multiple outputs.
The final model will take input images from a car camera and predict the lanes and lane types.
The script will train the model and then run inference on the training data.
During inference the original input image will be displayed along with the lane types overlayed on the image.
The script will also display the predicted lanes which will display in another window that shows white lanes
on a black background.
"""

# Params
TRAIN_NEW_MODEL = False
training_data_base_path = './Data/Training_Data'
image_x_file = 'x_data.npy'
image_y_file = 'y_data.npy'
aux_y_file = 'y_label_data.npy'

x_train = np.load(os.path.join(training_data_base_path, image_x_file))
y_train = np.load(os.path.join(training_data_base_path, image_y_file))
y_aux_train = np.load(os.path.join(training_data_base_path, aux_y_file))

# Create training data columns and headers
num_batch = x_train.shape[0]
num_rows = x_train.shape[1]
num_cols = x_train.shape[2]
num_channels = x_train.shape[3]

print('Input Image Shape:')
print(x_train[0].shape)
print('Image Label Shape:')
print(y_train[0].shape)
print('Aux Label Shape:')
print(y_aux_train[0].shape)

image_input = Input(shape=(num_rows, num_cols, num_channels), name='image_input')

# Main layers
conv2d_1 = Conv2D(16, padding='SAME', kernel_size=2, strides=2, activation='relu', name='main_conv2d_1')(image_input)
conv2d_2 = Conv2D(32, padding='SAME', kernel_size=2, strides=2, activation='relu', name='main_conv2d_2')(conv2d_1)
conv2d_3 = Conv2D(64, padding='SAME', kernel_size=2, strides=2, activation='relu', name='main_conv2d_3')(conv2d_2)
flatten = Flatten()(conv2d_3)

# Lane line prediction
dense_1 = Dense(256, activation='relu', name='line_dense_1')(flatten)
dense_2 = Dense(256, activation='relu', name='line_dense_2')(dense_1)
predictions_img = Dense(y_train.shape[1], activation='sigmoid', name='predictions_img')(dense_2)

# Lane type prediction
dense_type_1 = Dense(64, activation='relu', name='dense_type_1')(flatten)
dense_type_2 = Dense(64, activation='relu', name='dense_type_2')(dense_type_1)
predictions_type = Dense(y_aux_train.shape[1], activation='sigmoid', name='predictions_type')(dense_type_2)

model = Model(inputs=[image_input], outputs=[predictions_img, predictions_type])

model.compile(optimizer='adam', loss={'predictions_img': 'binary_crossentropy', 'predictions_type': 'binary_crossentropy'}, metrics=['accuracy'])
print(model.summary())

save_model_file_path = './Models/model'
if TRAIN_NEW_MODEL:
    # Train model
    model.fit({'image_input': x_train}, {'predictions_img': y_train, 'predictions_type': y_aux_train}, epochs=50, batch_size=4)

    # Save model
    model.save(save_model_file_path + '.hdf5', overwrite=True)
############################################################################


############################################################################
# Make predictions with model
############################################################################
pred_model = load_model(save_model_file_path + '.hdf5')

def _get_bottom_roi(img):
    # Crop bottom half of image
    height = np.size(img, 0)
    width = np.size(img, 1)
    return img[int(height / float(2)):height, 0:width]


def _get_aux_label_data(file):
    aux_y_label_path = './Data/Input_Data'

    # Open additional label file
    path_to_label_file = os.path.join(aux_y_label_path, 'labels.csv')
    df_labels = pd.read_csv(path_to_label_file)
    df_labels.set_index('image', inplace=True)

    # Convert labels to one hot
    df_labels1 = pd.get_dummies(df_labels['label_left_desc'], prefix='left_')
    df_labels2 = pd.get_dummies(df_labels['label_right_desc'], prefix='right_')

    # Concat all one hot labels into single dataframe
    df_labels = pd.concat([df_labels1, df_labels2], axis=1, sort=False)

    # Get additional label data
    df_record = df_labels.loc[str(file)]

    # Convert additional label to numpy
    np_one_hot = np.array(df_record)
    np_one_hot = np.reshape(np_one_hot, (1, np_one_hot.shape[0]))
    print(file)
    print(np_one_hot[0, :])
    image_labels = np_one_hot

    return image_labels


def _get_aux_label(pred_label):
    # ['left__broken', 'left__double_solid', 'left__single_solid', 'right__broken', 'right__single_solid']
    found_index = np.argpartition(pred_label, -2)[-2:]	# Find 2 indexes with largest values
    found_index = np.sort(found_index)
    print('Aux Index:')
    print(found_index)

    aux_label = ''
    if found_index[0] == 1 and found_index[1] == 4:
        aux_label = 'L: Double | R: Solid'
    elif found_index[0] == 0 and found_index[1] == 3:
        aux_label = 'L: Broken | R: Broken'
    elif found_index[0] == 2 and found_index[1] == 4:
        aux_label = 'L: Solid | R: Solid'
    elif found_index[0] == 1 and found_index[1] == 3:
        aux_label = 'L: Double | R: Broken'
    else:
        aux_label = 'Unknown'

    return  aux_label


# Make predictions using the training data
print('Predicting input images...')
index = 0
path_to_image_data_input = './Data/Input_Data/img'
for subdir, dirs, files in os.walk(path_to_image_data_input):
    for file in files:
        print(os.path.join(subdir, file))
        filepath = subdir + os.sep + file

        if (filepath.lower().endswith(".png") or filepath.lower().endswith(
                ".jpg")):
            index += 1

            # Load Image
            test_img_path = filepath
            test_img = Image.open(test_img_path)
            test_img = cv2.cvtColor(np.array(test_img), cv2.COLOR_BGR2RGB)

            # Convert to numpy array
            test_img_np = np.array(test_img)

            # Crop image
            test_img_np = _get_bottom_roi(test_img_np)
            test_img_np_copy = test_img_np.copy()

            # Get dimensions of image
            num_rows = test_img_np.shape[0]
            num_cols = test_img_np.shape[1]
            num_channels = test_img_np.shape[2]

            # Reshape for CNN model
            test_img_np = np.array(test_img_np).reshape(1, num_rows, num_cols, num_channels)

            additional_data = _get_aux_label_data(file)

            # Make prediction
            pred = pred_model.predict([test_img_np.astype('float16'), additional_data])
            predicted_lanes = pred[0][0]

            predicted_type = pred[1][0]

            # Get lane type display description
            aux_label = _get_aux_label(predicted_type)
            print('Lane Type:', aux_label)

            # Make predicted lanes visible for displaying to screen
            predicted_lanes = np.array(predicted_lanes).round()
            predicted_lanes = np.array(predicted_lanes).reshape(num_rows, num_cols)
            predicted_lanes = np.where(predicted_lanes == 1, 255, predicted_lanes)

            # Add lane type display text to image
            test_img_np_copy = cv2.putText(test_img_np_copy, aux_label, (int(num_cols/4), int(num_rows/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            #cv2.imshow('Original Lanes', np.array(test_img))
            cv2.imshow('Input Image', test_img_np_copy)
            cv2.imshow('Predicted Lanes', predicted_lanes)
            cv2.waitKey(1000)
