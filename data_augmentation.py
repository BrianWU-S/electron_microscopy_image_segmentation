#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,save_img

def data_augmentation(img_path,label_path):
    """
    do the image augmentation for training data set
    :param img_path: file path of training image
    :param label_path:  file path of training label
    """
    image_list=glob.glob(img_path+"*.png")
    label_list=glob.glob(label_path+"*.npy")
    # define data augmentor
    datagen = ImageDataGenerator(rotation_range=20,shear_range=0.2,zoom_range=0.2,
                                 vertical_flip=True,horizontal_flip=True,fill_mode='constant',cval=0)
    assert len(image_list)==len(label_list)
    for i in range(len(image_list)):
        img = load_img(img_path+str(i)+".png",grayscale=True)
        label= load_img(label_path+str(i)+".png",grayscale=True)
        x = img_to_array(img,data_format='channels_first')
        y = img_to_array(label,data_format='channels_first')
        pack=np.zeros(shape=(3,x.shape[1],x.shape[2]),dtype=np.float)
        pack[0,:,:]=np.squeeze(x,0)
        pack[1,:,:]=np.squeeze(y,0)
        #add a batch_size to the first dim
        pack = pack.reshape((1,) + pack.shape)
        # generate 40 augmentations for every image
        image_num = 40
        for i in range(image_num):
            a=datagen.flow(pack, batch_size=1)
            X,Y=next(a)
            X=np.squeeze(X,axis=0)
            im=array_to_img(np.unsqueeze(X[0,:,:],0),data_format='channels_first')
            lb=array_to_img(np.unsqueeze(X[1,:,:],0),data_format='channels_first')
            save_img("dataset/aug",im,file_format="png")
            save_img("dataset/aug_lb",lb,file_format="png")


if __name__ == "main":
    img_path="dataset/train_img/"
    label_path="dataset/train_label/"
    data_augmentation(img_path,label_path)
