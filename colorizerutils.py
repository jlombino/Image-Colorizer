import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

class image_loader(keras.utils.Sequence):

    def __init__(self,directory='../images/train_images/',batch_size=256,training=True):

        self.file_paths = [directory + file_name for file_name in os.listdir(directory)]
        self.batch_size = batch_size
        self.training = training

    def __len__(self):

        return math.ceil(len(self.file_paths)/self.batch_size)

    def __getitem__(self,idx):

        file_batch = self.file_paths[idx * self.batch_size: (idx+1) * self.batch_size]

        image_batch = [load_and_resize(image_filepath,self.training) for image_filepath in file_batch]

        lab_converted_batch = [rgb_to_lab_split_channels(image) for image in image_batch]

        l_channel = np.array([channels[0] for channels in lab_converted_batch])
        ab_channels = np.array([channels[1] for channels in lab_converted_batch])

        return l_channel, ab_channels

def load_and_resize(image_filepath,transform):

    image = cv2.imread(image_filepath)
    image = cv2.resize(image,(64,64))

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

    return image

def rgb_to_lab_split_channels(image):

    lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    lab_image_adj = np.add(lab_image,-128)/128

    l_channel = np.expand_dims(lab_image_adj[:,:,0],2)
    ab_channels = lab_image_adj[:,:,1:]

    return l_channel, ab_channels

def lab_to_rgb_combine_channels(l_channel,ab_channels):

    a_channel = ab_channels[:,:,0]
    b_channel = ab_channels[:,:,1]

    recombined = np.dstack([l_channel,a_channel,b_channel])
    recombined = np.add(recombined*128,128)
    converted = cv2.cvtColor(recombined.astype(np.uint8),cv2.COLOR_LAB2RGB)

    return converted

def display_images(generator,gray_channel,color_channels):

    generator_predictions = generator(gray_channel,training=False)

    figure, axes = plt.subplots(8,3,figsize=(10, 32))

    for index,row in enumerate(axes):
        row[0].imshow(gray_channel[index],cmap='gray')
        row[0].set_title('Gray Input',fontsize=20)

        row[1].imshow(lab_to_rgb_combine_channels(gray_channel[index],generator_predictions[index]))
        row[1].set_title('Colorized',fontsize=20)

        row[2].imshow(lab_to_rgb_combine_channels(gray_channel[index],color_channels[index]))
        row[2].set_title('Ground Truth',fontsize=20)

        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.set_facecolor('#FFFFFF')
        figure.subplots_adjust(wspace=0.02, hspace=0)

    return(figure)

def downsampling(filters,stride,prev_layer):

    init = initializers.RandomNormal()

    block = layers.Conv2D(filters,strides=stride,kernel_size=4,padding='same',kernel_initializer=init,use_bias=False)(prev_layer)
    block = layers.BatchNormalization()(block)
    block = layers.LeakyReLU(0.2)(block)

    return block

def upsampling(filters,stride,prev_layer,skip_layer):

    init = initializers.RandomNormal()

    block = layers.Conv2DTranspose(
        filters,strides=stride,kernel_size=4,padding='same',kernel_initializer=init,use_bias=False)(prev_layer)
    block = layers.BatchNormalization()(block)
    block = layers.Concatenate()([block,skip_layer])
    block = layers.LeakyReLU(0.2)(block)
    block = layers.Dropout(0.3)(block)

    return block