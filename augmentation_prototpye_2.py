import glob
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img


def data_augmentation(img_path, label_path, save_img_path, save_label_path):
    """
    do the image augmentation for training data set
    :param img_path: file path of training image
    :param label_path:  file path of training label
    """
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    if not os.path.exists(save_label_path):
        os.mkdir(save_label_path)
    image_list = glob.glob(img_path + "*.png")
    label_list = glob.glob(label_path + "*.png")
    # define data augmentor
    datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
                                 zoom_range=0.2,
                                 vertical_flip=True, horizontal_flip=True, fill_mode='constant', cval=0)
    assert len(image_list) == len(label_list)
    for i in range(len(image_list)):
        img = load_img(img_path + str(i) + ".png", color_mode="grayscale")
        label = load_img(label_path + str(i) + ".png", color_mode="grayscale")
        x = img_to_array(img, data_format='channels_first')  # [512,512] --> [1,512,512]
        y = img_to_array(label, data_format='channels_first')
        pack = np.ndarray(shape=(x.shape[1], x.shape[2], 3), dtype=np.uint8)  # [512,512,3]   # [channel,width,height]
        pack[:, :, 0] = np.squeeze(x, 0)
        pack[:, :, 1] = np.squeeze(y, 0)
        # add a batch_size to the first dim
        pack = pack.reshape((1,) + pack.shape)  # (1,512,512,3)
        # generate 40 augmentations for every image
        image_num = 40
        for id in range(image_num):
            a = datagen.flow(pack, batch_size=1)  # return ArrayFlow iterator
            X = next(a)  # get data: [1,512,512,3]
            X = np.squeeze(X, axis=0)
            im = np.expand_dims(X[:, :, 0], 0)  # [1,512,512]
            lb = np.expand_dims(X[:, :, 1], 0)  # [1,512,512]
            save_img(save_img_path + str(i) + "_" + str(id) + ".png", im, file_format="png",
                     data_format='channels_first')
            save_img(save_label_path + str(i) + "_" + str(id) + ".png", lb, file_format="png",
                     data_format='channels_first')


if __name__ == "__main__":
    img_path = r"dataset/train_img/"
    label_path = r"dataset/train_label/"
    save_img_path = r"dataset/aug_img_2/"
    save_label_path = r"dataset/aug_lb_2/"
    data_augmentation(img_path, label_path, save_img_path, save_label_path)
