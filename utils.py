from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import keras.backend as K
import cv2


Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def set_GPU_Memory_Limit():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def Unet_scheduler(epoch, lr):
    """
    learning rate decay
    """
    if epoch < 2:
        return lr
    elif epoch < 5:
        return 1e-4
    elif epoch < 10:
        return 1e-4
    else:
        return lr * tf.math.exp(-0.05)


def adjustData(img, mask, flag_multi_class, num_class):
    """
    Rescale image and turn the mask to one hot vector
    """
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]  # [batch_size,w,h,channel]
        new_mask = np.zeros(mask.shape + (num_class,))  # add one dimension for num_class size
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    
    else:
        img = img / 255  # can be replace by setting rescale parameter in ImageDataGenerator
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    if save_to_dir and not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield img, mask


def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    assert len(glob.glob(os.path.join(test_path,"*.png"))) <= num_image, "num_image need to be smaller than test image in current test_path"
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    """
    visualize the label image
    """
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    """
    save the visualized result
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img_as_ubyte(img))


def visualize_training_results(hist, save_path="../results/UNet/Unet_training", loss_flag=True, acc_flag=True,lr_flag=False):
    """
    visualize the loss function/acc/lr during the training process
    """
    print("Training history has key:")
    for key in hist.history:
        print(key)
    loss = hist.history['loss']
    acc = hist.history['accuracy']
    lr = hist.history['lr']
    if loss_flag:
        plt.plot(np.arange(len(loss)), loss)
        plt.scatter(np.arange(len(loss)), loss, c='g')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.savefig(os.path.join(save_path, "loss.png"))
        plt.show()
    if acc_flag:
        plt.plot(np.arange(len(acc)), acc)
        plt.scatter(np.arange(len(acc)), acc, c='g')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training accuracy")
        plt.savefig(os.path.join(save_path, "acc.png"))
        plt.show()
    if lr_flag:
        plt.plot(np.arange(len(lr)), lr)
        plt.scatter(np.arange(len(lr)), lr, c='g')
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Training learning rate decay")
        plt.savefig(os.path.join(save_path, "lr.png"))
        plt.show()


def bce_dice_loss(y_true, y_pred):
    """
    Training loss: BinaryCrossEntropy
    """
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    """
    Training loss: dice loss.
    Dice coefficient: 2* overlapped area space / total space
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def compute_dice(im1, im2, empty_score=1.0):
    """
    Evaluation metric: Dice
    """
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)
    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    
    intersection = np.logical_and(im1, im2)
    
    return 2. * intersection.sum() / im_sum


def compute_metrics(y_true, y_pred):
    """
    metrics of V_rand and V_info
    """
    v_rand,v_info=None,None
    pred_label = (y_pred > 0.5).astype(np.uint8)
    gt_label = (y_true > 0.5).astype(np.uint8)
    pred_num, pred_out = cv2.connectedComponents(pred_label, connectivity=4)
    gt_num, gt_out = cv2.connectedComponents(gt_label, connectivity=4)
    p = np.zeros((pred_num+1, gt_num+1))
    for i in range(pred_num+1):
        tmp_mask = (pred_out==i)
        for j in range(gt_num+1):
            if i==0 or j==0:
                p[i][j]=0
            else:
                p[i][j] = np.logical_and(tmp_mask, gt_out==j).sum()
    #normalize
    tot_sum = p.sum()
    p = p / tot_sum
    #marginal distribution
    s = p.sum(axis=0)
    t = p.sum(axis=1)
    #entropy
    sum_p_log = (p * np.log(p+1e-9)).sum()
    sum_s_log = (s * np.log(s+1e-9)).sum()
    sum_t_log = (t * np.log(t+1e-9)).sum()
    v_info = -2 * (sum_p_log - sum_s_log - sum_t_log) / (sum_s_log  + sum_t_log)
    sum_p_s = (p*p).sum()
    sum_s_s = (s*s).sum()
    sum_t_s = (t*t).sum()
    v_rand = 2 * sum_p_s / (sum_t_s + sum_s_s)
    return v_rand,v_info

def groundTruthLabelGenerator(label_path, num_label=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    assert len(glob.glob(os.path.join(label_path,
                                      "*.png"))) >= num_label, "num_label need to be smaller than test label in current label_path"
    masks = []
    for i in range(num_label):
        mask = io.imread(os.path.join(label_path, "%d.png" % i), as_gray=as_gray)
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = trans.resize(mask, target_size)
        mask = np.reshape(mask, mask.shape + (1,)) if (not flag_multi_class) else mask
        masks.append(mask)
    return masks


def predictLabelGenerator(label_path, num_label=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    assert len(glob.glob(os.path.join(label_path,"*.png"))) >= num_label, "num_label need to be smaller than test label in current label_path"
    masks = []
    for i in range(num_label):
        mask = io.imread(os.path.join(label_path, "%d_predict.png" % i), as_gray=as_gray)
        mask = mask / 255       # note: we can't use the threshold method as in groundTruthLabelGenerator()
        mask = trans.resize(mask, target_size)
        mask = np.reshape(mask, mask.shape + (1,)) if (not flag_multi_class) else mask
        masks.append(mask)
    return masks
