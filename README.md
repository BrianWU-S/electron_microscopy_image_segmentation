# Electron_Microscopy_Image_Segmentation
Image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain visual characteristics. In this project, we try to solve the problem in ISBI challenge.  In this challenge, a full stack of EM slices will be used to train machine learning algorithms for the purpose of automatic segmentation of neural structures.

The images are representative of actual images in the real-world, containing some noise and small image alignment errors. None of these problems led to any difficulties in the manual labeling of each element in the image stack by an expert human neuroanatomist. The aim of the challenge is to compare and rank the different competing methods based on their pixel and object classification accuracy.

# Dataset
The data description is same with ISBI Challenge except that we split the raw train data set (consist of 30 samples) into two parts: the new train set and new
test set. The downloaded data set consists of 30 samples, 25 for train and 5 for test. We simply train our model on the newly split data sets and did not use pre-training models. 

Here are one example for training data (raw image and corresponding label):
<!-- ![training-sample](/dataset/train_img/0.png) ![training-label](/dataset/train_label/0.png)  -->
<img align="left" src="/dataset/train_img/0.png">
<img align="right" src="/dataset/train_label/0.png">
Here are one example for test data (raw image and corresponding label):
![test-sample](/dataset/test_img/0.png) ![test-label](/dataset/test_label/0.png) 



# Group Members
Shiqu Wu

Shengyuan Hou

Binghao Yan
