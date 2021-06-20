#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from utils import *
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def plot(acc_list,v_rand_list,v_info_list):
   x = np.array([0,1,2,3,4])
   a = np.array(acc_list)
   b = np.array(v_rand_list)
   c = np.array(v_info_list)

   total_width, n = 0.88, 3
   width = total_width / n
   x = x - (total_width - width) / 2

   plt.bar(x, a,  width=width, label='accuracy')
   plt.bar(x + width, b, width=width, label='v_rand')
   plt.bar(x + 2 * width, c, width=width, label='v_info')
   for x, acc, v_rand, v_info in zip(x, a, b, c):
      plt.text(x , acc + 0.04, '%.2f' % acc, ha='center', va='top')
      plt.text(x + width, v_rand + 0.04, '%.2f' % v_rand, ha='center', va='top')
      plt.text(x + 2*width, v_info + 0.04, '%.2f' % v_info, ha='center', va='top')

   plt.title("Performance of Unet++ on accuracy, V_rand and V_info")
   plt.xlabel("Test data")
   plt.ylim((0,1.2))
   plt.legend()
   plt.show()


def evaluate():
   TARGET_SIZE = (512, 512)
   masks = groundTruthLabelGenerator(label_path="dataset/test_label", num_label=5, target_size=TARGET_SIZE,
                                     flag_multi_class=False, as_gray=True)
   results = predictLabelGenerator(label_path=r"UNetPP/UNetPP_submit/setting3", num_label=5,
                                   target_size=TARGET_SIZE,
                                   flag_multi_class=False, as_gray=True)
   assert np.shape(masks) == np.shape(results), "Label and predict image didn't have the same shape"
   v_rand_list = []
   v_info_list = []
   acc_list = []
   for img in range(np.shape(results)[0]):
      v_rand, v_info = compute_metrics(y_true=masks[img], y_pred=results[img])
      pred_label = (results[img] > 0.5).astype(np.uint8)
      gt_label = (masks[img] > 0.5).astype(np.uint8)
      tmp_mask = (pred_label == gt_label)
      print("Image:", img, "V_rand", v_rand, "V_info:", v_info, "Accuracy: ",
            Counter(tmp_mask.flatten())[True] / (512 * 512))
      v_rand_list.append(v_rand)
      v_info_list.append(v_info)
      acc_list.append(Counter(tmp_mask.flatten())[True] / (512 * 512))
   print("Mean v_rand:", np.mean(v_rand_list), " Mean v_info:", np.mean(v_info_list), "Mean acc:", np.mean(acc_list))
   plot(acc_list, v_rand_list, v_info_list)
   # print(v_rand_list,v_info_list)

if __name__ == '__main__':
   evaluate()
