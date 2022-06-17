import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_images(count=-1,image_size=64,image_dir='images/'):
    if count == -1:
        count = len(os.listdir(image_dir))

    image_array = []
    for image_file in os.listdir(image_dir):
        if len(image_array) >= count:
            break
        else:
            image = cv2.imread(image_dir+image_file)
            image = cv2.resize(image,(image_size,image_size))
            image_array.append(image)

    return np.array(image_array)

def rgb_to_lab_split_channels(image_set):
    l_array = []
    ab_array = []

    for image in image_set:
        lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        l_channel = np.add(l_channel,-128)/128
        a_channel = np.add(a_channel,-128)/128
        b_channel = np.add(b_channel,-128)/128
        ab_channels = np.dstack((a_channel,b_channel))

        l_array.append(np.expand_dims(l_channel,axis=2))
        ab_array.append(ab_channels)

    return np.array(l_array), np.array(ab_array)

def lab_to_rgb_combine_channels(gray_array,color_array):
    l_channel = np.add(gray_array*128,128)
    a_channel = np.add(color_array[:,:,0]*128,128)
    b_channel = np.add(color_array[:,:,1]*128,128)

    recombined = np.dstack((l_channel,a_channel,b_channel))
    converted = cv2.cvtColor(recombined.astype(np.uint8),cv2.COLOR_LAB2RGB)

    return converted

def make_dataset(X,y,batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X,y)).batch(batch_size)

    return dataset

def display_images(generator,X,y,count=5):
    y_pred = generator(X[:count]).numpy()

    fig, ax = plt.subplots(count,3,figsize=(12,4*count))
    for idx, row in enumerate(ax):
        row[0].imshow(X[idx],cmap='gray')
        row[1].imshow(lab_to_rgb_combine_channels(X[idx],y_pred[idx]))
        row[2].imshow(lab_to_rgb_combine_channels(X[idx],y[idx]))

        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.set_facecolor('#FFFFFF');
    ax[0][0].set_title('Input',fontsize=32);
    ax[0][1].set_title('Colorized',fontsize=32);
    ax[0][2].set_title('True',fontsize=32);

    return fig