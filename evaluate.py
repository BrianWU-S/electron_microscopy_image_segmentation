from utils import compute_metrics, groundTruthLabelGenerator, predictLabelGenerator
import numpy as np


if __name__ == '__main__':
    TARGET_SIZE = (512, 512)
    masks = groundTruthLabelGenerator(label_path="dataset/test_label", num_label=5, target_size=TARGET_SIZE,
                                      flag_multi_class=False, as_gray=True)
    results = predictLabelGenerator(label_path=r"results/UNet/UNet_submit/setting1", num_label=5, target_size=TARGET_SIZE,
                                      flag_multi_class=False, as_gray=True)
    assert np.shape(masks) == np.shape(results), "Label and predict image didn't have the same shape"
    v_rand_list = []
    v_info_list = []
    for img in range(np.shape(results)[0]):
        v_rand, v_info = compute_metrics(ground=masks[img], predict=results[img])
        print("Image:", img, "V_rand", v_rand, "V_info:", v_info)
        v_rand_list.append(v_rand)
        v_info_list.append(v_info)
    print("Mean v_rand:", np.mean(v_rand_list), " Mean v_info:", np.mean(v_info_list))
    # print(v_rand_list,v_info_list)
    