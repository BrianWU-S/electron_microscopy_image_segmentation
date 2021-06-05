import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from torchvision import models
from torch import nn
from torch.nn import functional as F
import torch
import torchvision
from functools import partial
from metrics import *
from plot import loss_plot
from plot import metrics_plot

EPOCHS = 100
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # pixel value -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                         )  # pixel value ->[-1,1]
])
y_transforms = transforms.ToTensor()  # Just convert label to tensor


class IsbiCellDataset:
    '''
        The data set for DataLoader
    '''

    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'./isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        # get image and label correspond to self.state
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        '''
            Get all the images and labels
        '''
        self.train_img_paths = glob(self.root + r'/train/images/*')
        self.train_mask_paths = glob(self.root + r'/train/label/*')
        self.val_img_paths = glob(self.root + r'/test/images/*')
        self.val_mask_paths = glob(self.root + r'/test/label/*')
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        # Validation set is the same as test set
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        '''
            override [] operation
            read and convert image and label
        '''
        pic_path = self.pics[index]  # get certain image and label paths according to index
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        # convert image and label to grayscale images then perform transformations
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    '''
        Dense Atrous Convolution block
        Implement atrous convolution by setting dilation > 1 in nn.Conv2d
    '''

    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(
            channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            # initialization of bias
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        '''
            Four branches of cascade atrous convolution operations
        '''
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(
            self.dilate3(self.dilate2(self.dilate1(x)))))
        # the final output of DAC is the sum of outputs of the four branches
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class RMPblock(nn.Module):
    '''
        Residual Multi-kernel Pooling block
        Consists of max pooling operations in different kernel size
    '''

    def __init__(self, in_channels):
        super(RMPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=1, kernel_size=1, padding=0)  # 1x1 convolutional layer to reduce weights and parameters

    def forward(self, x):
        '''
            Concatenate the output of each max pooling operations after 1x1 convolution together
        '''
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(
            self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(
            self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(
            self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(
            self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    '''
        Decoder block
        convolution --> deconvolution as up-sampling --> convolution
    '''

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CE_Net_(nn.Module):
    '''
        Context Encoder Network
    '''

    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_, self).__init__()

        filters = [64, 128, 256, 512]

        # Encoder, utilize pre-trained ResNet-34 to encode inputs
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # DAC and RMP in the center
        self.dblock = DACblock(512)
        self.rmp = RMPblock(512)

        # Decoder, consists of multiple Decoder blocks
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.rmp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


def getLog():
    '''
        log training & testing process
    '''
    filename = './log.log'
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logging


def getModel():
    '''
        Get CE-Net model
    '''
    model = CE_Net_().to(device)
    return model


def getDataset():
    '''
        Get training, validation and testing datasets in the form of torch.utils.data.DataLoader
    '''
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    train_dataset = IsbiCellDataset(
        r'train', transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataset = IsbiCellDataset(
        r"val", transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)
    test_dataset = IsbiCellDataset(
        r"test", transform=x_transforms, target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders, val_dataloaders, test_dataloaders


def val(model, best_iou, val_dataloaders):
    '''
        Validation process during training
    '''
    model = model.eval()  # set model to evaluation mode
    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        for x, _, pic, mask in val_dataloaders:  # for each image and label in the validation set
            x = x.to(device)
            y = model(x)
            img_y = torch.squeeze(y).cpu().numpy()  # get prediction

            # caculate hausdorff distance, iou and dice scores by comparing prediction and label
            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0], img_y)
            dice_total += get_dice(mask[0], img_y)
            if i < num:
                i += 1

        # calculate average scores and log them in the log file
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' %
              (aver_iou, aver_hd, aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' %
                     (aver_iou, aver_hd, aver_dice))

        # save best model
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(
                aver_iou, best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/CENET.pth')
        return best_iou, aver_iou, aver_dice, aver_hd


def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs, threshold):
    '''
        Training process
    '''
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = epochs
    threshold = threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()  # set model to training mode
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, _, mask in train_dataloader:  # for each images and labels in training set
            step += 1
            inputs = x.to(device)
            labels = y.to(device)

            optimizer.zero_grad()  # zero the parameter gradients
            output = model(inputs)
            loss = criterion(output, labels)  # calculate loss value
            if threshold != None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()  # optimization step
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) //
                                              train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) //
                                                     train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        # validate the model
        best_iou, aver_iou, aver_dice, aver_hd = val(
            model, best_iou, val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(num_epochs, loss_list)

    # plot training process
    metrics_plot(num_epochs, 'iou&dice', iou_list, dice_list)
    metrics_plot(num_epochs, 'hd', hd_list)
    return model


def test(val_dataloaders, save_predict=False):
    '''
        Test process
    '''
    # load model parameters
    logging.info('final test........')
    if save_predict == True:
        dir = r'./saved_predict'
        if not os.path.exists(dir):
            os.makedirs(dir)
    model.load_state_dict(torch.load(
        r'./saved_model/CENET.pth', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        for pic, _, pic_path, mask_path in val_dataloaders:  # for each image and label in the validation set
            pic = pic.to(device)
            predict = model(pic)
            predict = torch.squeeze(predict).cpu().numpy()

            # caculate hausdorff distance, iou and dice scores by comparing prediction and label
            iou = get_iou(mask_path[0], predict)
            miou_total += iou
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0], predict)
            dice_total += dice

            # plot the prediction label
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict, cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            if save_predict == True:
                plt.savefig(dir + '/' + mask_path[0].split('/')[-1])
                np.save(dir + '/' + mask_path[0].split('/')[-1], predict)
            print('iou={},dice={}'.format(iou, dice))
            if i < num:
                i += 1
        print('Miou=%f,aver_hd=%f,dv=%f' %
              (miou_total/num, hd_total/num, dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dv=%f' %
                     (miou_total/num, hd_total/num, dice_total/num))


if __name__ == '__main__':
    logging = getLog()
    print('**************************')
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\n========' %
                 ("CE-Net", str(EPOCHS), "4"))
    print('**************************')
    model = getModel()
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset()
    criterion = torch.nn.BCELoss()  # using BCELoss
    optimizer = optim.Adam(model.parameters())  # using Adam
    train(model, criterion, optimizer, train_dataloaders,
          val_dataloaders, EPOCHS, None)
    test(test_dataloaders, save_predict=True)
