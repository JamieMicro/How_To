import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv3D, Conv2D, Conv1D, Dropout, MaxPooling2D, LeakyReLU, BatchNormalization, Reshape, InputLayer, Input, ZeroPadding3D, ZeroPadding2D, UpSampling2D, UpSampling3D, MaxPool3D, MaxPooling3D
from tensorflow.keras.optimizers import Adam
import os, time
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


dtype='float16'
#K.set_floatx(dtype)
# default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
#K.set_epsilon(1e-4)


class GAN():
    def __init__(self):
        self.CWD = os.getcwd()  # Set current working directory

        # Video params
        self.img_rows = 48  # Frame height: 240  480  120  96  960 | 48
        self.img_cols = 85  # Frame width: 320  640  160 128  1280 | 64
        self.shrink_by = 10 # Shrink frames by a factor of...
        self.channels = 30  # Depth but can also be used for color channels
        self.frames_per_batch = 30   # Frames per training example (s/b sequential)
        self.NOISE_VALUES = 100     # Noise input array size
        self.USE_LABEL_FLIP = False  # Throws off discriminator by swapping labels with incorrect values
        self.LABEL_FLIP_EPS = .10   # Works with random value to flip labels (not needed)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  # If not color channels = # frames per examaple

        # Input paths
        self.INPUT_IMAGES_PATH = os.path.join(self.CWD, r'Data\Input_Video')
        self.INPUT_IMAGES_FAKE_PATH = os.path.join(self.CWD, r'Data\Input_Images_Fake')
        self.INPUT_VIDEO_FILE_NAME = 'robot_2.mp4'
        self.SAVED_OUTPUT_PATH = r'Data\Saved_Output'

        # Training data paths
        self.TRAINING_DATA_PATH = os.path.join(self.CWD, r'Data\Training_Data_Video')
        self.X_TRAINING_DATA_PATH = os.path.join(self.CWD, r'Data\Training_Data_Video\x_training_data.npy')
        self.y_TRAINING_DATA_PATH = os.path.join(self.CWD, r'Data\Training_Data_Video\y_training_data.npy')
        self.X_y_TRAINING_DATA_PATH = os.path.join(self.CWD, r'Data\Training_Data_Video\x_y_training_data.npy')

        # Misc params
        self.MODEL_SAVE_FOLDER = r'Models\Saved_GAN'      # Path where models will be saved
        self.CREATE_TRAINING_DATA = True    # True = new numpy file. False will use existing file @ X_TRAINING_DATA_PATH
        self.IMAGE_COUNTER = 0

        # Flags for training existing Models (False will create new models)
        self.USE_EXISTING_DISC = False
        self.EXISTING_DISCRIMINATOR = os.path.join(self.MODEL_SAVE_FOLDER, 'discriminator.hdf5')

        self.USE_EXISTING_GEN = False
        self.EXISTING_GENERATOR = os.path.join(self.MODEL_SAVE_FOLDER, 'generator.hdf5')

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        if self.USE_EXISTING_DISC and os.path.exists(self.EXISTING_DISCRIMINATOR):
            self.discriminator = load_model(self.EXISTING_DISCRIMINATOR)
        else:
            self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        if self.USE_EXISTING_GEN and os.path.exists(self.EXISTING_GENERATOR):
            self.generator = load_model(self.EXISTING_GENERATOR)
        else:
            self.generator = self.build_generator()

        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generates video frames
        z = Input(shape=(self.NOISE_VALUES,))
        img = self.generator(z)

        # For the combined model, only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_image_file(self, path_to_image):
        img = Image.open(path_to_image).convert('RGB')
        w, h = img.size
        return img, w, h

    def create_numpy_dataset_from_video(self):
        input_video_file_path = os.path.join(self.INPUT_IMAGES_PATH, self.INPUT_VIDEO_FILE_NAME)
        input_video_file_path_fake = self.INPUT_IMAGES_FAKE_PATH

        x_data = np.array([])
        y_data = np.array([])
        x_data_batch = np.array([])
        x_y_data = np.array([])

        # Read images frames
        vidcap = cv2.VideoCapture(input_video_file_path)
        success, img = vidcap.read()
        count = 0
        batch_count = 0
        total_batches = 1
        success = True

        width = vidcap.get(3)  # float
        height = vidcap.get(4)  # float
        print('Frame dims:', height, width)
        new_width = int(width / self.shrink_by)
        new_height = int(height / self.shrink_by)
        print('New frame dims:', new_height, new_width)

        use_color_channel = False

        # This loop converts real input video frames in to sets of N frames and finally to numpy array
        while success:
            count += 1
            batch_count += 1
            #cv2.imwrite("frame%d.jpg" % count, img)  # save frame as JPEG file

            img = cv2.resize(img, (new_width, new_height))

            if use_color_channel:
                # w/ color channel
                temp_x_data = img.reshape(new_height, new_width, self.channels)
            else:
                # w/o color channel
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                temp_x_data = img.reshape(new_height, new_width)

            cv2.imshow('Preview', img)
            cv2.waitKey(1)

            # If first frame in batch
            if batch_count == 1:
                x_data_batch = temp_x_data
                x_data_batch = np.reshape(x_data_batch, (x_data_batch.shape[0], x_data_batch.shape[1], 1))
            else:
                temp_x_data = np.reshape(temp_x_data, (temp_x_data.shape[0], temp_x_data.shape[1], 1))
                x_data_batch = np.dstack([x_data_batch, temp_x_data])

            x_data_batch_v = np.reshape(x_data_batch, (1, x_data_batch.shape[0], x_data_batch.shape[1], x_data_batch.shape[2]))

            random_value = round(np.random.random(), 2)
            flip = False
            if random_value < self.LABEL_FLIP_EPS and self.USE_LABEL_FLIP:
                flip = True

            if batch_count == self.frames_per_batch and total_batches == 1:
                if flip:
                    temp_y_data = np.array([0])     # Set to incorrect label
                else:
                    temp_y_data = np.array([1])

                y_data = temp_y_data

                x_data = x_data_batch_v
                batch_count = 0
                total_batches += 1
            elif batch_count == self.frames_per_batch and total_batches != 1:
                if flip:
                    temp_y_data = np.array([0])     # Set to incorrect label
                else:
                    temp_y_data = np.array([1])

                y_data = np.append(y_data, temp_y_data)

                x_data = np.vstack((x_data, x_data_batch_v))
                total_batches += 1
                batch_count = 0

            # y data
            if count == 1:
                if flip:
                    temp_y_data = np.array([0])     # Set to incorrect label
                else:
                    temp_y_data = np.array([1])
                x_y_data = np.append(x_data_batch_v.flatten(), temp_y_data)
            else:
                if flip:
                    temp_y_data = np.array([0])     # Set to incorrect label
                else:
                    temp_y_data = np.array([1])
                temp_x_y_data = np.append(x_data_batch_v.flatten(), temp_y_data)
                #x_y_data = np.vstack((x_y_data, temp_x_y_data))

            success, img = vidcap.read()
            print('Read frame:', count, success)

        # This loop converts fake input images in to sets of N frames and finally to numpy array
        # Create training data for fake input images (aka 'x')
        print('Converting fake input images to numpy training data...')
        index = 0
        batch_count = 0
        for subdir, dirs, files in os.walk(input_video_file_path_fake):
            for file in files:
                filepath = subdir + os.sep + file
                print(filepath)

                if (filepath.lower().endswith(".png") or filepath.lower().endswith(".jpg")):
                    index += 1
                    batch_count += 1

                    img, width, height = self.load_image_file(filepath)
                    #img = cv2.resize(img, (new_width, new_height))
                    img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
                    temp_x_data = np.array(img)

                    # If first frame in batch
                    if batch_count == 1:
                        x_data_batch = temp_x_data
                        x_data_batch = np.reshape(x_data_batch, (temp_x_data.shape[0], temp_x_data.shape[1], 1))
                    else:
                        temp_x_data = np.reshape(temp_x_data, (temp_x_data.shape[0], temp_x_data.shape[1], 1))
                        x_data_batch = np.dstack([x_data_batch, temp_x_data])

                    # Append to training data
                    x_data_batch_v = np.reshape(x_data_batch, (
                    1, x_data_batch.shape[0], x_data_batch.shape[1], x_data_batch.shape[2]))

                    if batch_count == self.frames_per_batch and total_batches == 1:
                        x_data = x_data_batch_v
                        batch_count = 0
                        total_batches += 1

                        if flip:
                            temp_y_data = np.array([1])  # Set to incorrect label
                        else:
                            temp_y_data = np.array([0])

                        y_data = temp_y_data
                    elif batch_count == self.frames_per_batch and total_batches != 1:
                        x_data = np.vstack((x_data, x_data_batch_v))
                        total_batches += 1
                        batch_count = 0

                        if flip:
                            temp_y_data = np.array([1])  # Set to incorrect label
                        else:
                            temp_y_data = np.array([0])

                        y_data = np.append(y_data, temp_y_data)

                    # y data
                    if count == 1:
                        x_y_data = np.append(x_data_batch_v.flatten(), y_data)
                    else:
                        if flip:
                            temp_y_data = np.array([1])  # Set to incorrect label
                        else:
                            temp_y_data = np.array([0])
                        temp_x_y_data = np.append(x_data_batch_v.flatten(), temp_y_data)
                        #x_y_data = np.vstack((x_y_data, temp_x_y_data))

        # Save to numpy files
        print('Saving x data...')
        x_data_path = os.path.join(self.TRAINING_DATA_PATH, 'x_training_data.npy')
        np.save(x_data_path, x_data)
        print('Saving y data...')
        y_data_path = os.path.join(self.TRAINING_DATA_PATH, 'y_training_data.npy')
        np.save(y_data_path, y_data)
        print('Saving x y data...')
        x_y_data_path = os.path.join(self.TRAINING_DATA_PATH, 'x_y_training_data.npy')
        #np.save(x_y_data, x_data)

    def create_numpy_dataset(self, path_to_input_images, path_to_output_data):
        r"""
        This function creates a numpy dataset using pre labeled images and corresponding input images.
        The output will be cropped (_get_bottom_roi) images saved as x & y numpy files.
        """

        MAX_IMAGES = 500

        x_data = np.array([])
        y_data = np.array([])
        x_y_data = np.array([])

        # Create training data for input images (aka 'x')
        print('Converting input images to numpy training data...')
        index = 0
        for subdir, dirs, files in os.walk(path_to_input_images):
            for file in files:
                filepath = subdir + os.sep + file
                print(filepath)

                if (filepath.lower().endswith(".png") or filepath.lower().endswith(
                        ".jpg")):
                    index += 1

                    if index >= MAX_IMAGES and MAX_IMAGES > 0:
                        break

                    img, width, height = self.load_image_file(filepath)
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (96, 96))

                    # Add to numpy array
                    num_rows = img.shape[0]
                    num_cols = img.shape[1]
                    num_channels = img.shape[2]

                    # Append to training data
                    if index == 1:
                        x_data = img.reshape(1, num_rows, num_cols, num_channels)

                        if str(subdir).endswith('fake'):
                            y_data = np.array([0])
                        else:
                            y_data = np.array([1])

                        x_y_data = np.append(x_data.flatten(), y_data)
                        # print(x_y_data.shape)
                    else:
                        x_data = np.vstack(
                            (x_data, img.reshape(1, num_rows, num_cols, num_channels)))

                        if str(subdir).endswith('fake'):
                            y_data_tmp = np.array([0])
                        else:
                            y_data_tmp = np.array([1])
                        y_data = np.vstack((y_data, y_data_tmp))

                        x_y_data_tmp = np.append(img.reshape(1, num_rows, num_cols, num_channels).flatten(), y_data_tmp)
                        x_y_data = np.vstack((x_y_data, x_y_data_tmp))


        print('Saving x data...')
        x_data_path = os.path.join(path_to_output_data, 'x_training_data.npy')
        np.save(x_data_path, x_data)
        print('Saving y data...')
        y_data_path = os.path.join(path_to_output_data, 'y_training_data.npy')
        np.save(y_data_path, y_data)
        print('Saving x y data...')
        x_y_data_path = os.path.join(path_to_output_data, 'x_y_training_data.npy')
        np.save(x_y_data_path, x_y_data)
        print('Process Complete')

    ######## GENERATOR ########

    def build_generator(self):

        noise_shape = (self.NOISE_VALUES,)

        model = Sequential()

        model.add(InputLayer(input_shape=noise_shape, name='Input_1'))
        model.add(Dense(5 * 4 * 5 * 1, activation="relu", name='Dense_1'))
        model.add(Reshape((5, 4, 5, 1), name='Reshape_1'))
        model.add(UpSampling3D())  # Transforms small input to a large image output
        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', name='Conv3D_1'))
        model.add(BatchNormalization(momentum=0.8))  # Improves performance/stability and helps generalize
        model.add(UpSampling3D())
        model.add(Conv3D(64, kernel_size=(2, 2, 2), activation='relu', name='Conv3D_2'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform', name='Dense_2'))
        model.add(Dense(np.prod(self.img_shape), activation='tanh', name='Dense_Output'))  # Size of samples
        model.add(Reshape(self.img_shape))  # Reshape to our video sample dimensions
        model.summary()

        noise = Input(shape=noise_shape)  # Create Input for Model
        img = model(noise)  # Create Output for Model

        return Model(noise, img)

    ###############################

    ######## DISCRIMINATOR ########

    def build_discriminator(self):
        model = Sequential()

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model.add(Conv2D(128, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    ###############################

    def train_my_gan(self, epochs, batch_size=128, save_interval=50, save_model_epoch=5000):
        # Create training data
        if self.CREATE_TRAINING_DATA:
            self.create_numpy_dataset_from_video()

        # Load the dataset
        X_train = np.load(self.X_TRAINING_DATA_PATH)
        y_train = np.load(self.y_TRAINING_DATA_PATH)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            #print(idx)
            imgs = X_train[idx]
            imgs_y = y_train[idx]
            #print('imgs.shape', imgs.shape)

            noise = np.random.normal(0, 1, (half_batch, self.NOISE_VALUES))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            #d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_real = self.discriminator.train_on_batch(imgs, imgs_y)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.NOISE_VALUES))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_my_imgs(epoch)

            if save_model_epoch == epoch and save_model_epoch > 0:
                self.save_my_gan()

    def save_my_imgs(self, epoch):
        self.IMAGE_COUNTER += 1
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.NOISE_VALUES))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Display/save sample image from generator
        file_name = 'generator_sample_' + str(self.IMAGE_COUNTER) + '.png'
        file_path = os.path.join(self.SAVED_OUTPUT_PATH, file_name)
        save_img = gen_imgs[0, :, :, 0].copy()
        save_img = save_img * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite(file_path, save_img)

        temp_img = gen_imgs[0, :, :, 0].copy()
        temp_img = cv2.resize(temp_img, (temp_img.shape[1] * 4, temp_img.shape[0] * 4))
        cv2.imshow('Sample Img', temp_img)
        cv2.waitKey(1)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                #axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        sample_frame_file_name = 'training_sample_frames_epoch_' + str(epoch) + '.png'
        sample_frame_file_path = os.path.join(self.SAVED_OUTPUT_PATH, sample_frame_file_name)
        fig.savefig(sample_frame_file_path)
        plt.close()

    def save_my_gan(self):
        print('Saving GAN models to:', self.MODEL_SAVE_FOLDER)
        save_model(self.discriminator, os.path.join(self.MODEL_SAVE_FOLDER, 'discriminator.hdf5'), overwrite=True)
        print('Saved discriminator.h5...')
        save_model(self.generator, os.path.join(self.MODEL_SAVE_FOLDER, 'generator.hdf5'), overwrite=True)
        print('Saved generator.h5...')
        save_model(self.combined, os.path.join(self.MODEL_SAVE_FOLDER, 'combined.hdf5'), overwrite=True)
        print('Saved combined.h5...')


# Main program
if __name__ == '__main__':

    # Helper function that saves images for review and visibility into progress
    def save_image(img, suffix):
        # Display/save sample image from generator
        file_name = 'new_fake_video_' + str(suffix) + '.png'
        file_path = os.path.join('Data/Saved_Output', file_name)
        save_img = img.copy()
        save_img = save_img * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite(file_path, save_img)

    # GAN Class
    gan = GAN()

    # Control Params
    train_gan = False       # Train a new GAN
    gen_gan = True          # Run the GAN
    save_all_frames = False # Save generated frames for review (warning: creates a lot of files)

    # Run GAN Model Params
    generator = None
    MODEL_SAVE_FOLDER = r'Models\Saved_GAN'     # Load GAN model from here
    NOISE_VALUES = 100                          # Input noise latent vector size
    TOTAL_FRAMES = 500                          # Stop after creating n videos
    FRAMES = 30                                 # Frames per example

    # Train a new GAN
    if train_gan:
        print(86 * '=')
        gan.train_my_gan(epochs=2000, batch_size=32, save_interval=100, save_model_epoch=500)
        print(86 * '=')
        gan.save_my_gan()
        print(86 * '=')

    # Run the trained GAN
    if gen_gan:
        print(86 * '=')
        print('Loading saved generator model...')
        generator = load_model(os.path.join(MODEL_SAVE_FOLDER, 'generator.hdf5'))

        # For specified number of videos
        for f in range(TOTAL_FRAMES):
            print('Generating noise for GAN...', f)
            # Create random noise as input
            noise = np.random.normal(0, 1, (FRAMES, NOISE_VALUES))

            print('Running inference to generate frames...')
            # Generate video output
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5

            # Display/Save Output
            for i in range(FRAMES):
                temp_img = gen_imgs[i, :, :, 0].copy()
                if save_all_frames:
                    save_image(temp_img, str(f) + '_' + str(i))
                temp_img = cv2.resize(temp_img, (temp_img.shape[1] * 8, temp_img.shape[0] * 8))
                cv2.imshow('Generated Video', temp_img)
                cv2.waitKey(200)

            # Slow down for viewing
            time.sleep(.2)
print('Done')
