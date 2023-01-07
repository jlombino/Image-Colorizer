import os
import cv2
import math
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Generator to load the images into a tensorflow dataset with transformations
class image_loader(keras.utils.Sequence):

    # Default directory to load image is outside repository
    # training value determines whether to apply random rotations to images
    def __init__(self,directory='../images/train_images/',batch_size=256,training=True,sort_files=False):

        if not sort_files:
            self.file_paths = [directory + file_name for file_name in os.listdir(directory)]
            
        else:
            self.file_paths = [directory + file_name for file_name in sorted(os.listdir(directory))]

        self.batch_size = batch_size
        self.training = training

    def __len__(self):

        return math.ceil(len(self.file_paths)/self.batch_size)

    def __getitem__(self,idx):

        # List of file names in the batch
        file_batch = self.file_paths[idx * self.batch_size: (idx+1) * self.batch_size]

        # List of numpy arrays representing images in the batch
        image_batch = [load_and_resize(image_filepath,self.training) for image_filepath in file_batch]

        # Creates a list of tuples representing images
        # Index 0 contains the l (grayscale) channel for each image
        # Index 1 contains the a and b (color) channels for each image
        lab_converted_batch = [rgb_to_lab_split_channels(image) for image in image_batch]

        l_channel = np.array([channels[0] for channels in lab_converted_batch])
        ab_channels = np.array([channels[1] for channels in lab_converted_batch])

        return l_channel, ab_channels

def load_and_resize(image_filepath,transform):

    image = cv2.cvtColor(cv2.imread(image_filepath),cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))

    # Apply random rotations and flips to images for training
    if transform:

        rotation = random.randint(0,3)
        if rotation == 0:
            pass
        elif rotation == 1:
            image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 2:
            image = cv2.rotate(image,cv2.ROTATE_180)
        elif rotation == 3:
            image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        flip = random.randint(0,1)
        if flip:
            image = cv2.flip(image,1)

    # Returns numpy array of image
    return image

def rgb_to_lab_split_channels(image):

    # Converts image to LAB color space and sets pixel values
    # between 0 and 1
    lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    lab_image_adj = np.add(lab_image,-128)/128

    # Splits the image into L (grayscale) and A/B (color) channels
    l_channel = np.expand_dims(lab_image_adj[:,:,0],2)
    ab_channels = lab_image_adj[:,:,1:]

    # Returns 2 numpy arrays representing channels of the image
    return l_channel, ab_channels

def lab_to_rgb_combine_channels(l_channel,ab_channels):

    # Split the a and b channels to work with numpy dstack
    a_channel = ab_channels[:,:,0]
    b_channel = ab_channels[:,:,1]

    # Recombines channels and converts image back to rgb format
    recombined = np.dstack([l_channel,a_channel,b_channel])
    recombined = np.add(recombined*128,128)
    converted = cv2.cvtColor(recombined.astype(np.uint8),cv2.COLOR_LAB2RGB)

    # Returns numpy array of RGB image
    return converted

def display_images(gray_channel,color_channels,generator1,gen1_title,generator2=None,gen2_title=None):

    # Predict colors using generator1
    generator1_predictions = generator1(gray_channel,training=False)

    # Smaller figure if only one generator
    if generator2 is None:
        figure, axes = plt.subplots(8,3,figsize=(10, 32))

    # Predict colors and make larger figure if generator2 is present
    else:
        generator2_predictions = generator2(gray_channel,training=False)
        figure,axes = plt.subplots(8,4,figsize=(14,32))

    for index,row in enumerate(axes):

        # Display the grayscale image
        row[0].imshow(gray_channel[index],cmap='gray')
        row[0].set_title('Gray Input',fontsize=20)

        # Display the image made using generator1
        row[1].imshow(lab_to_rgb_combine_channels(gray_channel[index],generator1_predictions[index]))
        row[1].set_title(gen1_title,fontsize=20)

        # Display the original color image if generator2 is note present        
        if generator2 is None:
            row[2].imshow(lab_to_rgb_combine_channels(gray_channel[index],color_channels[index]))
            row[2].set_title('Ground Truth',fontsize=20)

        # Display the image made using generator2 and the original image
        else:
            row[2].imshow(lab_to_rgb_combine_channels(gray_channel[index],generator2_predictions[index]))
            row[2].set_title(gen2_title,fontsize=20)

            row[3].imshow(lab_to_rgb_combine_channels(gray_channel[index],color_channels[index]))
            row[3].set_title('Ground Truth',fontsize=20)

        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.set_facecolor('#FFFFFF')
        figure.subplots_adjust(wspace=0.02, hspace=0)

    return(figure)

def make_gifs(image_folder,batch_size,batch_number,generator,save_folder):

    # Empty lists for the frames for 3 gifs
    grayscale_frames = []
    colorized_frames = []
    original_frames = []

    # Load the batch of sorted images starting at frame batch size * batch number
    image_batch = image_loader(directory=image_folder,
    batch_size=batch_size,training=False,sort_files=True).__getitem__(batch_number)

    # Generate colored frames using generator
    generator_output = generator(image_batch[0],training=False)

    for frame_number in range(batch_size):

        # Fill a and b channels in the grayscale image with zeros
        grayscale_dummy_color = np.dstack([np.zeros_like(image_batch[0][frame_number]),
            np.zeros_like(image_batch[0][frame_number])])

        # Need to recombine channels and convert back to rgb
        # Add each grayscale frame to grayscale list
        grayscale_frames.append(lab_to_rgb_combine_channels(l_channel=image_batch[0][frame_number],
            ab_channels=grayscale_dummy_color))

        # Add each colorized frame made using generator to colorized list
        colorized_frames.append(lab_to_rgb_combine_channels(l_channel=image_batch[0][frame_number],
            ab_channels=generator_output[frame_number]))

        # Add each original frame to original list without modifying
        original_frames.append(lab_to_rgb_combine_channels(l_channel=image_batch[0][frame_number],
            ab_channels=image_batch[1][frame_number]))

    # Create gifs from each list and save to save folder prefixed with batch number
    with imageio.get_writer(save_folder + f'{batch_number}_grayscale.gif',mode='I') as writer:
        for frame in grayscale_frames:
            writer.append_data(frame)

    with imageio.get_writer(save_folder + f'{batch_number}_colorized.gif',mode='I') as writer:
        for frame in colorized_frames:
            writer.append_data(frame)

    with imageio.get_writer(save_folder + f'{batch_number}_original.gif',mode='I') as writer:
        for frame in original_frames:
            writer.append_data(frame)