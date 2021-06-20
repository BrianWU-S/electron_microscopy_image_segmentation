from unet_model import UNet
from utils import trainGenerator, testGenerator, geneTrainNpy, saveResult, set_GPU_Memory_Limit, Unet_scheduler, \
    visualize_training_results
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time

if __name__ == '__main__':
    set_GPU_Memory_Limit()
    EPOCHES = 15
    LR = 5e-4
    STEPS = 300
    BATCH_SIZE = 1
    TARGET_SIZE = (512, 512)
    t1 = time.time()
    #settings of data augmentation
    data_gen_args = dict(  # rescale=1. / 255,
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='constant')
    data_gen_args_2 = dict(  # rescale=1. / 255,
        rotation_range=20,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0)
    #genereate the data configuration
    myGenerator = trainGenerator(batch_size=BATCH_SIZE, train_path='../dataset',
                                 image_folder='train_img',  # set save_to_dir="../dataset/UNet" to view
                                 mask_folder='train_label',
                                 aug_dict=data_gen_args, image_color_mode="grayscale", mask_color_mode="grayscale",
                                 image_save_prefix="img", mask_save_prefix="lb", flag_multi_class=False, num_class=2,
                                 save_to_dir=None, target_size=TARGET_SIZE, seed=1234)
    #define the UNet model
    model = UNet(pretrained_weights=None, input_size=(512, 512, 1), lr=LR)  # we did not use pre-trained model here
    callbacks_list = [
        ModelCheckpoint('models/unet_membrane_best.hdf5', monitor='loss', verbose=1,save_best_only=True),
        LearningRateScheduler(schedule=Unet_scheduler)
    ]
    # handle and visualize the history
    hist = model.fit_generator(myGenerator, steps_per_epoch=STEPS, epochs=EPOCHES,callbacks=callbacks_list)
    visualize_training_results(hist=hist, save_path="Unet_submit/Unet_training", loss_flag=True, acc_flag=True, lr_flag=True)
    
    # test
    testGene = testGenerator(test_path="../dataset/test_img", num_image=5, target_size=TARGET_SIZE,flag_multi_class=False, as_gray=True)
    results = model.predict_generator(testGene, steps=5, verbose=1)
    saveResult(save_path="../UNetPP/results/UNet", npyfile=results, flag_multi_class=False, num_class=2)
    print("Training time:", time.time() - t1, "s")
    # save the final model
    model.save(filepath=r"../models/UNet/unet_e%i_s_%i_lr_%f.hdf5" % (EPOCHES, STEPS, LR))
    
    print("Training process finished. Congratulations, sir!")
