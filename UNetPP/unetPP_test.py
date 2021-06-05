from UNetPP.unetPP_model import UNetPlusPlus
from utils import testGenerator, saveResult, set_GPU_Memory_Limit, compute_metrics, groundTruthLabelGenerator
import numpy as np

if __name__ == '__main__':
    set_GPU_Memory_Limit()
    TARGET_SIZE = (512, 512)
    model = UNetPlusPlus(pretrained_weights="../models/UNetPP/unetPP_membrane_bestIOU.hdf5")
    testGene = testGenerator(test_path="../dataset/test_img", num_image=5, target_size=TARGET_SIZE,
                             flag_multi_class=False, as_gray=True)
    masks = groundTruthLabelGenerator(label_path="../dataset/test_label", num_label=5, target_size=TARGET_SIZE,
                                      flag_multi_class=False, as_gray=True)
    results = model.predict_generator(testGene, steps=5, verbose=1)
    assert np.shape(masks) == np.shape(results), "Label and predict image didn't have the same shape"
    v_rand_list = []
    v_info_list = []
    for img in range(results.shape[0]):
        v_rand, v_info = compute_metrics(ground=masks[img], predict=results[img])
        print("Image:", img, "V_rand", v_rand, "V_info:", v_info)
        v_rand_list.append(v_rand)
        v_info_list.append(v_info)
    print("Mean v_rand:", np.mean(v_rand_list), " Mean v_info:", np.mean(v_info_list))
    # print(v_rand_list,v_info_list)
    
    saveResult(save_path="../results/UNetPP", npyfile=results, flag_multi_class=False, num_class=2)
