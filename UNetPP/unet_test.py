from unet_model import UNet
from utils import testGenerator, saveResult, set_GPU_Memory_Limit


if __name__ == '__main__':
    set_GPU_Memory_Limit()
    model = UNet(pretrained_weights="../models/UNet/unet_membrane.hdf5")
    testGene = testGenerator(test_path="../dataset/test_img", num_image=5, target_size=(512, 512),
                             flag_multi_class=False, as_gray=True)
    results = model.predict_generator(testGene, steps=5, verbose=1)
    saveResult(save_path="results/UNet", npyfile=results, flag_multi_class=False, num_class=2)
