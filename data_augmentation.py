#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def data_augmentation(img_path,label_path):
    """
    do the image augmentation for training data set
    :param img_path: file path of training image
    :param label_path:  file path of training label
    """
    #define data augmentor
    image_list=glob.glob(img_path+"*.png")
    label_list=glob.glob(label_path+"*.npy")
    datagen = ImageDataGenerator(rotation_range=20,shear_range=0.2,zoom_range=0.2,
                                 vertical_flip=True,horizontal_flip=True,fill_mode='constant',cval=0)
    num=len(image_list)
    for i in range(num):
        img = load_img(img_path+str(i)+".png",grayscale=True)
        label= load_img(label_path+str(i)+".png",grayscale=True)
        x = img_to_array(img)
        y = img_to_array(label)
        #add a batch_size to the first dim
        x = x.reshape((1,) + x.shape)
        y = y.reshape((1,) + y.shape)
        # generate 40 augmentations for every image
        image_id = 0
        for batch in datagen.flow(x, batch_size=1,save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
            image_id += 1
            if image_id > 40:
                break


if __name__ == "main":
    img_path="dataset/train_img/"
    label_path="dataset/train_label/"
    data_augmentation(img_path,label_path)
