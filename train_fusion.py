import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
import sys
sys.path.append(r'~/Pose Estimation Keras')
import cnn
from cnn import conv_base
from build_dataset import Dataset
import numpy as np
# from sklearn.preprocessing import normalize
from scipy import stats

from keras import layers
from keras import models
from keras import optimizers
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from time import time
import cv2
import os 


# Some Hyperparameters
total_frames = 5
frame_height = 224
frame_width = 224
frame_channels = 3
num_bodyparts = 6



def build_deconv_model():
    frame_tensor = conv_base.layers[0].get_input_at(0)
    resnet_output = cnn.convolution_network(frame_tensor)
    model_resnet = models.Model(inputs=[frame_tensor], outputs=[resnet_output])
    
    deconv_output = cnn.deconvolution_fusion_network(model_resnet.output)
    model_deconv = models.Model(inputs=[frame_tensor], outputs=[deconv_output])
    
    losses = {
        "deconvolution_fusion_network_output": "binary_crossentropy",
#         "spatial_fusion_output": "binary_crossentropy",
#         "optical_flow_output": "binary_crossentropy"
    }
    lossWeights = {"deconvolution_fusion_network_output": 1.0,}
#                    "spatial_fusion_output": 1.0,
#                    "optical_flow_output": 1.0}
    
#     opt = optimizers.SGD(lr = 0.0, momentum=0.9, nesterov=True)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model_deconv.compile( optimizer=opt, loss=losses, loss_weights=lossWeights, metrics = ["accuracy"])
    
    return model_deconv

model = build_deconv_model()
model.summary()


import csv
def parse_csv(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        indices = []
        for line in csv_reader:
            indices.extend([int(f) for f in line if f!=''])
    return indices

indices = parse_csv('dataset_indices.csv')
print('Following Indices will be trained from dataset.h5: \n', indices)


import h5py
with h5py.File('../dataset.h5', 'r') as hf:
    # central_heatmap = hf['central_heatmap'][indices]
    frames = hf['frames'][indices]
    # heatmaps_individual = hf['heatmaps_individual'][indices]
    heatmaps_combined = hf['heatmaps_combined'][indices]


home_dir = os.getcwd()
heatmaps_combined = np.expand_dims(heatmaps_combined, axis=-1)

print('Shape of Training Data',heatmaps_combined.shape)


def step_decay(epoch):
    try:
        if epoch < 30:
            lrate = 0.0001
        elif epoch < 70:
            lrate = 0.00005
        elif epoch < 100:
            lrate = 0.00001
        return lrate
    except:
        print('Error in setting learning rate')
        
lrate = LearningRateScheduler(step_decay, verbose=1)

checkpointer = ModelCheckpoint(filepath= os.path.join(home_dir, "pretrained_models/model011019_spatial.h5"),
                               verbose=1,
                               monitor='loss')

tensorboard = TensorBoard(log_dir=os.path.join(home_dir, "logs/{}".format(time())))



def normalize(x):
    x = np.array(x, dtype = np.float64)
    x -= np.mean(x, dtype = np.float64)
    x /= np.std(x, dtype = np.float64)
    return x


def generator(frames, heatmaps_combined):
    while(1):
        for j,l in zip(frames, heatmaps_combined):
#             print(l.shape[0])
            for channel in range(l.shape[0]):
                l[channel, :, :, :] = normalize(l[channel, :, :, :]) 
            # include batch dimension and repeat the matrix at axis 0 num_frames times to match the target dimensionality
            yield (j, {
#                 "optical_flow_output" : np.repeat(np.expand_dims(k, axis=0), 5, 0), 
                       "deconvolution_fusion_network_output": l}) 
            
model.fit_generator(generator(frames,
                                     heatmaps_combined),
                           steps_per_epoch=100, 
                           verbose = 2,
                           epochs = 150,
                           callbacks=[lrate, checkpointer, tensorboard])