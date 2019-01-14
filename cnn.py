# __author__  = "Umar Ayub"
# __email__ = "umar.ayub@duke.edu"

from keras import layers
from keras import models
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.applications import ResNet50
from keras.regularizers import l2


# Video Dimensions

total_frames = 5
frame_height = 224
frame_width = 224
frame_channels = 3
num_bodyparts = 6
weight_regularization = 0.0001

conv_base = ResNet50(weights = 'imagenet', 
                      include_top = False, 
                      input_shape = (frame_height, frame_width, frame_channels))




# Network Parameters

# Define Feature Extraction Network
def convolution_network(x):
    x = conv_base(x)
#     x = layers.Flatten()(x)
#     x = layers.Reshape((112,112,8))(x)
    return x


# Define Deconvolution network to extract heatmaps for each joint position

def deconvolution_network(x):
    x = layers.Conv2DTranspose(num_bodyparts*5,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts*4,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts*3,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts*2,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization),
                                activation= 'sigmoid',
                                name = "deconvolution_network_output")(x)

    return x


# Spatial Fusion layers that learn dependencies between the joint locations

def spatial_fusion_network(x1, x2):
    nb_channels = 32
    _strides = (1, 1)

    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        return y
    
    x1 = layers.Flatten()(x1)
    x = layers.concatenate([x1, x2])
    x = layers.Reshape((224,224,8))(x)
    
    x = layers.Conv2D(nb_channels, kernel_size=(7, 7), strides=_strides, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(nb_channels*2,  kernel_size=(13, 13),strides=_strides, padding='same')(x)
    x = add_common_layers(x)
    x = layers.Conv2D(nb_channels*4,  kernel_size=(13, 13),strides=_strides, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(nb_channels*8,  kernel_size=(1, 1),strides=_strides, padding='same')(x)
    x = add_common_layers(x)
    x = layers.Conv2D(1, kernel_size=(1, 1), strides=_strides, padding='same', name = 'spatial_fusion_output',  activation = 'sigmoid')(x)
    return x

# Function to compute optical flow calculations on a list of frames
def optical_flow(frame_list):    
    def calculate_optical_flow(one, two):
        one = np.squeeze(one, axis=0)
        two = np.squeeze(two, axis=0)
        flow = cv2.calcOpticalFlowFarneback(one, 
                                            two, 
                                            flow=None,
                                            pyr_scale=0.5, levels=1, winsize=15,
                                            iterations=2,
                                            poly_n=5, poly_sigma=1.1, flags=0)

        return flow

    def move_image_based_on_flow(prev_img, flow):
        # Generate a cartesian grid
        #         print('prev_img',prev_img.shape)
        height, width = flow.shape[0], flow.shape[1]
        R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))

        # desired mapping is simply the addition of this grid with the flow
        pixel_map = R2 + flow
        pixel_map = pixel_map.astype(np.float32)
        
        #perform the remapping of each pixel in the original image
        warped_frame = cv2.remap(prev_img, pixel_map[:, :, 0], pixel_map[:, :, 1], cv2.INTER_LINEAR)
        warped_frame = np.expand_dims(warped_frame, axis = -1)
        #         print('warped_frame',warped_frame.shape)
        return warped_frame
    
    mid = 2
    base_frame = frame_list[:, mid]
    new_list = tf.identity(frame_list)
    new_list = tf.concat((new_list[:,:mid], new_list[:,mid+1:]), axis = 1)
    
    warped_frames = []
    for i in range(total_frames-1):
        warped_bps_in_frame = []
        for k in range(num_bodyparts):
            bp_heatmap = new_list[:, i , :, : , k, np.newaxis]
            reference_frame = base_frame[:, : , :, k, np.newaxis]
            flow = tf.py_func(calculate_optical_flow, [reference_frame, bp_heatmap], tf.float32)
            new_frame = tf.py_func(move_image_based_on_flow, [bp_heatmap, flow], tf.float32)
            warped_bps_in_frame.append(new_frame)   
        warped_frames.append(warped_bps_in_frame)

    base_frame = tf.reshape(base_frame, (6, 224, 224, 1))
    warped_frames.insert(mid, base_frame)
    warped_frames = tf.stack(warped_frames, axis=0)
    warped_frames = tf.transpose(warped_frames, [4, 1, 2, 3, 0])
    warped_frames = tf.reshape(warped_frames, shape = (1, num_bodyparts,  frame_height, frame_width, total_frames))
    return warped_frames

def expand_dims(x):
    return K.expand_dims(x, axis = 0)

def repeat_target(x):
    return K.repeat_elements(x, total_frames, axis = 0)

def optical_flow_network(x):
    x = layers.Lambda(expand_dims, output_shape=(total_frames, frame_height, frame_width, num_bodyparts))(x)
    #     x = layers.Input(batch_shape=(1, total_frames, frame_height, frame_width, num_bodyparts))
    x = layers.Lambda(optical_flow, output_shape=(num_bodyparts, frame_height, frame_width, total_frames))(x)
    x = layers.Conv3D(32, kernel_size=(1, 1, 1), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv3D(1, kernel_size=(1, 1, 1), padding='same')(x)
    x = layers.Lambda(repeat_target, name = 'optical_flow_output')(x)
    return x



def deconvolution_fusion_network(x):
    x = layers.Conv2DTranspose(num_bodyparts*5,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts*4,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts*3,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts*2,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization))(x)
    x = layers.Conv2DTranspose(num_bodyparts,
                                kernel_size=3, 
                                strides = 2,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization),
                                activation= 'sigmoid',
                                name = "same")(x)
    x = layers.Conv2DTranspose(1,
                                kernel_size=3, 
                                strides = 1,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization),
                                activation= 'sigmoid',
                                name = "deconvolution_fusion_network_output")(x)

    return x


def separation_network(x):
    x = layers.Conv2DTranspose(num_bodyparts,
                                kernel_size=3, 
                                strides = 1,
                                padding = 'same',
                                kernel_regularizer=l2(weight_regularization),
                                bias_regularizer=l2(weight_regularization),
                                activation= 'sigmoid',
                                name = "deconvolution_network_output")(x)
    return x
