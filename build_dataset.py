import moviepy.editor as mpy
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, frames_path, df_path):
        self.base_path = frames_path
        self.frame_names= self.sort_frames(frames_path)
        self.df_rows = self.df_list(pd.read_hdf(df_path))
        self.frames = self.get_frames(self.base_path, self.frame_names)
        #self.heatmaps = [self.to_heatmap(i) for i in self.df_rows]
        self.heatmaps_ndarr = [self.dict_to_4Darray(self.to_heatmap(i)) for i in self.df_rows]
        self.combined_heatmaps = [self.to_combined_heatmap(i) for i in self.df_rows]

        '''
        Heirachy of frames = arr_x[sample_number][frame_number]
        Heirarchy of heatmaps = arr_y[sample_number][frame_number][bodypart_number]['heat_map']
        '''
        self.dataset = self.make_dataset(self.frames, self.heatmaps_ndarr, self.combined_heatmaps)
        self.central_heatmap = self.extract_heatmap_of_middle_frame(self.dataset['y'])
        self.second_dataset = self.make_second_dataset(self.dataset['y'], self.central_heatmap)
    
    def sort_frames(self, frames_path):
        lister = os.listdir(frames_path)
        bunch_of_frames = []
        for num in range(len(lister)):
            fname = [i for i in lister if int(i.split('.')[0].split('image')[-1]) == num]
            bunch_of_frames.append(fname[0])
        return bunch_of_frames

    def to_heatmap(self, Dataframe,  r = 3, pcutoff = 0):
        scorer= Dataframe[1].index.get_level_values(0)[0]
        bodyparts2plot = list(np.unique(Dataframe[1].index.get_level_values(1)))
        images = [np.zeros([550, 560, 3], dtype=np.uint8)]*len(bodyparts2plot)
        for bpindex, bp in enumerate(bodyparts2plot):
            image = Image.fromarray(images[bpindex])
            draw = ImageDraw.Draw(image)
            if Dataframe[1][scorer][bp]['likelihood'] > pcutoff:
                xc = int(Dataframe[1][scorer][bp]['x'])
                yc = int(Dataframe[1][scorer][bp]['y'])
                draw.ellipse((xc-r, yc-r, xc+r, yc+r), fill=(255,255,255,255))
            image = image.filter(ImageFilter.GaussianBlur())
            image = image.resize((224,224))
            images[bpindex] = {'heatmap': np.asanyarray(image.convert('L'), dtype='uint8'), 'id':bp}
        return images

    def to_combined_heatmap(self, Dataframe, r = 3, pcutoff = 0):
        scorer= Dataframe[1].index.get_level_values(0)[0]
        bodyparts2plot = list(np.unique(Dataframe[1].index.get_level_values(1)))
        image = np.zeros([550, 560, 3], dtype=np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for bpindex, bp in enumerate(bodyparts2plot):
            if Dataframe[1][scorer][bp]['likelihood'] > pcutoff:
                xc = int(Dataframe[1][scorer][bp]['x'])
                yc = int(Dataframe[1][scorer][bp]['y'])
                draw.ellipse((xc-r, yc-r, xc+r, yc+r), fill=(255,255,255,255))
        image = image.filter(ImageFilter.GaussianBlur())
        image = image.resize((224,224))
        return  np.asanyarray(image.convert('L'), dtype='uint8')


    def df_list(self, Dataframe):
        return [i for i in Dataframe.iterrows()]

    def get_frames(self, base_path, frame_names):
        return [np.asanyarray(Image.open(os.path.join(base_path, i)).resize((224,224)), dtype='uint8') for i in frame_names]

    def plot_datapoint(self, frame, heatmaps):
        def example_plot(ax, image,  fontsize=12):
            try:
                ax.imshow(image['heatmap'])
                ax.set_title(image['id'])
                ax.axis('off')
            except:
                ax.imshow(image)
                ax.axis('off')
        plt.close('all')
        plt.figure(figsize=(10,10))
        
        ax1 = plt.subplot2grid((5,4), (0,0))
        ax2 = plt.subplot2grid((5,4), (2,0))
        ax3 = plt.subplot2grid((5,4), (4,0))
        ax4 = plt.subplot2grid((5,4), (0,3))
        ax5 = plt.subplot2grid((5,4), (2,3))
        ax6 = plt.subplot2grid((5,4), (4,3))
        ax7 = plt.subplot2grid((5,4), (1, 1), rowspan = 3, colspan = 2)
        
        example_plot(ax1, heatmaps[3])
        example_plot(ax2, heatmaps[4])
        example_plot(ax3, heatmaps[5])
        example_plot(ax4, heatmaps[0])
        example_plot(ax5, heatmaps[1])
        example_plot(ax6, heatmaps[2])
        example_plot(ax7, frame)

        plt.tight_layout()

    def make_dataset(self, frames, heatmaps, combined_heatmaps, size=387):
        frames_chopped = np.array_split(frames, size)
        heatmaps_chopped = np.array_split(heatmaps, size)
        comb_heatmaps_chopped = np.array_split(combined_heatmaps, size)
        return {'x' : frames_chopped, 'y': heatmaps_chopped, 'y1': comb_heatmaps_chopped}

    def dict_to_4Darray(self, dictionary_list):
        frame_list = []
        for d in dictionary_list:
            frame_list.append(d['heatmap'])
        fourD_array = np.asarray(frame_list, dtype = 'uint8')
        return fourD_array

    def extract_heatmap_of_middle_frame(self, heatmaps):
        mid = heatmaps[-1].shape[0]//2
        return [heatmap_set[mid] for heatmap_set in heatmaps]   

    def make_second_dataset(self, heatmaps, central_heatmaps):
        return{'x': heatmaps, 'y': central_heatmaps}