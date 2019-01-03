# __author__  = "Umar Ayub"
# __email__ = "umar.ayub@duke.edu"

import tensorflow as tf
import sys
sys.path.append(r'C:\Users\ua12\Desktop\deep_pose_estimation_in_videos')
import cnn
from build_dataset import Dataset
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
import keras.backend as K
from keras.applications import ResNet50
from keras.callbacks import TensorBoard, ModelCheckpoint 
from time import time
import cv2


'''Load and Preprocess Training Data'''
frames_path = r"C:\Users\ua12\Desktop\deep_pose_estimation_in_videos\Videos\181023_1_6_video1\FrameByFrame"
df_path = r"C:\Users\ua12\Desktop\deep_pose_estimation_in_videos\Videos\181023_1_6_video1\181023_1_6_video1DeepCut_resnet50_digitsOct24shuffle1_675000.h5"
data = Dataset(frames_path, df_path)
frames = data.dataset['x'][1:]
heatmaps = data.dataset['y']
heatmaps = [np.swapaxes(i, 1, 3) for i in heatmaps][1:]

combined_heatmaps = data.dataset['y1']
combined_heatmaps = [np.expand_dims(i, axis=-1) for i in combined_heatmaps][1:]

central_heatmaps = data.second_dataset['y']
central_heatmaps = [np.expand_dims(i, axis = -1) for i in central_heatmaps][1:]



''''Define Functional Model'''
# Some Hyperparameters
total_frames = 5
frame_height = 224
frame_width = 224
frame_channels = 3
num_bodyparts = 6



conv_base = ResNet50(weights = 'imagenet', 
                      include_top = False, 
                      input_shape = (frame_height, frame_width, frame_channels))

for layer in conv_base.layers:
    layer.trainable = False

def build_model():
    frame_tensor = conv_base.input
    resnet_output = conv_base(frame_tensor)
    model_resnet = models.Model(inputs=[frame_tensor], outputs=[resnet_output])

    deconv_output = cnn.deconvolution_network(conv_base.layers[-1].output)
    model_deconv = models.Model(inputs=[frame_tensor], outputs=[deconv_output])

    sfn_input1 =  conv_base.layers[-1].output
    sfn_input2 = model_deconv.layers[-4].output
    sfn_output = cnn.spatial_fusion_network(sfn_input1, sfn_input2)

    optical_flow_input = model_deconv.layers[-1].output
    optical_flow_output = cnn.optical_flow_network(optical_flow_input)
    model_master = models.Model(inputs=[frame_tensor], outputs=[deconv_output, sfn_output, optical_flow_output])
    
    # compile model with multiple loss functions passed as dictionaries where the key is the name of the layer.
    
    # define two dictionaries: one that specifies the loss method for each output of the network along with a second dictionary that
    # specifies the weight per loss
    losses = {
        "deconvolution_network_output": "mean_squared_error",
        "spatial_fusion_output": "mean_squared_error",
        "optical_flow_output": "mean_squared_error"
    }
    lossWeights = {"deconvolution_network_output": 1.0,
                   "spatial_fusion_output": 1.0,
                   "optical_flow_output": 1.0}
    
    
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model_master.compile( optimizer=opt, loss=losses, loss_weights=lossWeights, metrics = ["accuracy"])

    return model_master

checkpointer = ModelCheckpoint(filepath=r"C:\Users\ua12\Desktop\deep_pose_estimation_in_videos\pretrained models\model.h5",
                               verbose=0,
                               save_best_only=True)

tensorboard = TensorBoard(log_dir=r"C:\Users\ua12\Desktop\deep_pose_estimation_in_videos\logs\{}".format(time()))

model_master = build_model()


'''Define generator to load data in batches'''
def generator(frames, central_heatmaps, heatmaps, combined_heatmaps):
    while(1):
        for j,k,l,m in zip(frames, central_heatmaps, heatmaps, combined_heatmaps):
            # include batch dimension and repeat the matrix at axis 0 num_frames times to match the target dimensionality
            yield (j, {"optical_flow_output" : np.repeat(np.expand_dims(k, axis=0), 5, 0), 
                       "deconvolution_network_output": l, 
                       "spatial_fusion_output": m}) 

'''Train Model'''
model_master.fit_generator(generator(frames, 
                                     central_heatmaps,
                                     heatmaps, 
                                     combined_heatmaps),
                           steps_per_epoch=386, 
                           verbose = 2,
                           epochs = 100,
                           callbacks=[checkpointer, tensorboard])
