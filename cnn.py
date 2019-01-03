# __author__  = "Umar Ayub"
# __email__ = "umar.ayub@duke.edu"

from keras import layers
from keras import models
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K


# Video Dimensions

total_frames = 5
frame_height = 224
frame_width = 224
frame_channels = 3
num_bodyparts = 6


# Network Parameters

cardinality = 1
# Define Deconvolution network to extract heatmaps for each joint position

def deconvolution_network(x):
    
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        return y
    
    def grouped_convolution(y, nb_channels, _strides):
        if cardinality == 1:
            return layers.Conv2DTranspose(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
          
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality
        
        groups = []
        for j in range(cardinality):
            group =  layers.Lambda(lambda z: z[:,:,:, j*_d: j*_d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group)) 
            
        y = layers.concatenate(groups)
        
        return y
    
    
    def deconv_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1)):
        
        y = layers.Conv2DTranspose(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)
        
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)
        
        y = layers.Conv2DTranspose(nb_channels_out, kernel_size=(3,3), strides=(1, 1), padding='valid')(y)
        y = add_common_layers(y)
        
        return y
    
    
    # conv1
    x = layers.Conv2DTranspose(1024, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = add_common_layers(x)
    
    # conv2
    for i in range(1):
        strides = (2, 2)
        x = deconv_block(x, 1024, 512, _strides=strides)
        
    # conv3
    for i in range(2):
        strides = (2, 2)
        x = deconv_block(x, 512, 256, _strides=strides)
        
    # conv3
    for i in range(1):
        strides = (2, 2)
        x = deconv_block(x, 256, 18, _strides=strides)
        
    # Refactor into spatial probability maps for each bp
    x = layers.Flatten()(x)
    x = layers.Reshape((60492, num_bodyparts))(x)
    x = layers.Cropping1D((5158,5158))(x)
    x = layers.Reshape((frame_height, frame_width, num_bodyparts), name = 'deconvolution_network_output')(x)
    
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
    x = layers.Reshape((463304, 1))(x)
    x = layers.Cropping1D((6027,6027))(x)
    x = layers.Reshape((475,475,2))(x)
    
    x = layers.Conv2D(nb_channels, kernel_size=(7, 7), strides=_strides, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(nb_channels*2,  kernel_size=(13, 13),strides=(2,2), padding='same')(x)
    x = add_common_layers(x)
    x = layers.Conv2D(nb_channels*4,  kernel_size=(13, 13),strides=_strides, padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Cropping2D(cropping=((7, 7), (7, 7)))(x)
    
    x = layers.Conv2D(nb_channels*8,  kernel_size=(1, 1),strides=_strides, padding='valid')(x)
    x = add_common_layers(x)
    x = layers.Conv2D(1, kernel_size=(1, 1), strides=_strides, padding='same', name = 'spatial_fusion_output')(x)
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




