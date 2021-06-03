# Electron Microscopy Image Segmentation
Image segmentation is the process of assigning a label to every pixel 
in an image such that pixels with the same label share certain visual characteristics. 
In this project, we try to solve the problem in ISBI challenge where a full stack of EM slices will be used to train 
machine learning algorithms for the purpose of automatic segmentation of neural structures.
The images are representative of actual images in the real-world, containing some noise and small image alignment errors. 
None of these problems led to any difficulties in the manual labeling of each element in the image stack by an expert 
human neuroanatomist. 

The aim of the challenge is to compare and rank the different competing methods based on their
pixel and object classification accuracy. The accuracy of segmentation is very important for medical images, because 
the edge segmentation error 
will lead to unreliable results, in that case it will be rejected for clinical application.
Obtaining these sample images to train the model may be a resource consuming process because of the need for high-quality, 
uncompressed and accurately annotated images reviewed by professionals.
Therefore, the algorithm designed for medical imaging must achieve high performance and accuracy with less data samples.

# Dataset
The data description is same with ISBI Challenge except that we split the raw train data set (consist of 30 samples) into two parts: the new train set and new
test set. The downloaded data set consists of 30 samples, 25 for train and 5 for test. We simply train our model on the newly split data sets and did not use pre-training models. 

Here is one example for training data (raw image and corresponding label):

<table>
  <tr>
    <td><img src="/dataset/train_img/0.png" width=270 height=270></td>
    <td><img src="/dataset/train_label/0.png" width=270 height=270></td>
  </tr>
 </table>

Here is one example for test data (raw image and corresponding label):
<table>
  <tr>
    <td><img src="/dataset/test_img/0.png" width=270 height=270></td>
    <td><img src="/dataset/test_label/0.png" width=270 height=270></td>
  </tr>
 </table>

# Data Augmentation

For the fact that the training data size is too small (we have only 25 training images),
we apply image augmentation by:

```
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
                                 zoom_range=0.2,
                                 vertical_flip=True, horizontal_flip=True, fill_mode='constant', cval=0)
```
After that, we get an augmented dataset of **10000** images. Here is one example for augmented training data:
<table>
  <tr>
    <td><img src="/dataset/aug/0_20.png" width=270 height=270></td>
    <td><img src="/dataset/aug_lb/0_20.png" width=270 height=270></td>
  </tr>
 </table>



# Models and methods

In this task, we implemented
- [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
- [U-Net++](https://arxiv.org/pdf/1807.10165.pdf)
- [CE-Net](https://arxiv.org/pdf/1903.02740.pdf)

And here are some useful links for understanding the model:

[Difference between U-Net and U-Net++](
https://sh-tsang.medium.com/review-unet-a-nested-u-net-architecture-biomedical-image-segmentation-57be56859b20)

[U-Net++ in medical image segmentation](https://www.yinxiang.com/everhub/note/d01d5753-28f8-4649-94e0-a810e8bee795)

[What's new in CENet compared with U-Net](https://zhuanlan.zhihu.com/p/273416963)



### Unet ++
UNet++的目标是通过在编码器和解码器之间加入Dense block和卷积层来提高分割精度。

在这个task中，分割的准确性对于医学图像至关重要，因为边缘分割错误会导致不可靠的结果，从而被拒绝用于临床中。
而获取这些样本图像来训练模型可能是一个消耗资源的过程，因为需要由专业人员审查的高质量、未压缩和精确注释的图像。 
因此为医学成像设计的算法必须在数据样本较少的情况下实现高性能和准确性。


## Requirements

- `python 3.6+`
- `numpy`
- `matplotlib`
- `torch`
- `torchvision`
- `keras`
- `glob`
- `os`


# Group Members
Shiqu Wu

Shengyuan Hou

Binghao Yan



