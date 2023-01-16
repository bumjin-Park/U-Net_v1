################################
##### U-net_Dataloader_V1 ######
################################



import os
import numpy as np
from torch.utils import data
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform_img=None,  transform_seg=None):
        self.data_dir = data_dir
        lst_label = os.listdir(data_dir + "masks")
        lst_input = os.listdir(data_dir + "images") 
        
        lst_label.sort()
        lst_input.sort()

        self.lst_label =  lst_label
        self.lst_input = lst_input
        self.transform_img = transform_img
        self.transform_seg = transform_seg

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        seg_file = self.lst_label[index]
        img_file = self.lst_input[index]

        seg = Image.open(self.data_dir + "masks/" + seg_file)
        img = Image.open(self.data_dir + "images/" + img_file)

        if self.transform_img:
            img = self.transform_img(img)
            # print('hi')
        if self.transform_seg:
            seg = self.transform_seg(seg)
            # print('hi')

        data = {'input': img, 'label': seg}

        return data
        

