import os
from os.path import isfile, join
import numpy as np
import glob
import cv2



class ChromaKeyingToVid:
    '''
        input: images
        output: video generated from input image sequence
    '''
    def __init__(self, input_dir,output_dir,vid, fps):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fps = fps
        self.vid = vid
    
    def generateVid(self):
        array = []
        files = os.listdir(self.input_dir)
        files.sort(key = lambda x: int(x.split(".")[0]))
        # print(files)
        for i in files:
            filename = self.input_dir + "/" + i
            image = cv2.imread(filename)
            h, w, l = image.shape
            size = (w, h)
            array.append(image)
        
        video = cv2.VideoWriter(os.path.join(self.output_dir, self.vid), cv2.VideoWriter_fourcc(*'DIVX'), self.fps,size)
        for i in range(len(array)):
            video.write(array[i])
        video.release()

# folders = [0]
folders = os.listdir('../image_infer')
for i in folders:
    print(i)
    chromaKeyingToVid = ChromaKeyingToVid('../image_infer/' + str(i), 'videos/', 'output_'+ str(i) +'.avi', 10)
    chromaKeyingToVid.generateVid()
