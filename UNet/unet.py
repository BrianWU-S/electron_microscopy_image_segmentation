#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from torch import nn
import logging
import time
from utils import *
from torch.utils.data import DataLoader
from torch import optim
from sklearn import metrics
import numpy as np
from torch.autograd import Variable
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,save_img

batch_size=4
epochs=100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate=5e-4
log_dir = os.path.join('log', time.asctime(time.localtime(time.time()))).replace(" ", "_").replace(":", "_")
log=open(os.path.join(log_dir, 'log.txt'), 'a+')

def get_logger():
   filename = 'log/'+time.asctime(time.localtime(time.time())).replace(" ", "_").replace(":", "_")
   logging.basicConfig(
      filename=filename,
      filemode='w+',
      level=logging.INFO,
      format='%(asctime)s:%(levelname)s:%(message)s'
   )
   return logging


class V_rand_loss():
    def __init__(self,):
        self.loss=metrics.adjusted_rand_score

    def forward(self,y_true,y_pred):
        return self.loss(y_true,y_pred)


class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2


class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out


def load_dataset():
    train,validation,test = ISBIdataset()
    train_generator = DataLoader(train,batch_size=batch_size,shuffle=False)
    validation_generator = DataLoader(validation, batch_size=1,shuffle=False)
    test_generator = DataLoader(test,batch_size=1,shuffle=False)
    return train_generator,validation_generator,test_generator


def compute_metrics(ground,predict,reverse=False):
    acc,v_rand,v_info=None,None,None
    # pred, gt are both numpy arrays, both of which predicts edges as "1"
    if reverse:
        pred_label = (predict < 0.5).astype(np.uint8)
        gt_label = (ground < 0.5).astype(np.uint8)
    else:
        pred_label = (predict > 0.5).astype(np.uint8)
        gt_label = (ground > 0.5).astype(np.uint8)
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
    return acc,v_rand,v_info


def test(model,test_generator,save_predict=True):
    model = model.eval()
    total_acc = 0;total_rand = 0;total_info = 0
    with torch.no_grad():
        for (x,y) in test_generator:
            x = x.to(device)
            y = model(x)
            img_y = torch.squeeze(y[-1]).cpu().numpy()
            if save_predict == True:
                save_img(os.path.join("predict_result",""))
            acc,rand,info=compute_metrics(x,img_y)
            total_acc+=acc
            total_rand+=rand
            total_info+=info
    logger.info('Testing: Accuracy: ' + str(total_acc / len(test_generator)) +
                ', V_rand: ' + str(total_rand / len(test_generator)) +
                ', V_info: ' + str(total_info / len(test_generator)), file=log, flush=True)
    return total_acc / len(test_generator), total_rand / len(test_generator), total_info / len(test_generator)


def val(model,val_generator,epoch,best_rand=None,isval=True):
    model = model.eval()
    total_acc=0;total_rand=0;total_info=0
    with torch.no_grad():
        for (x,y) in val_generator:
            x = x.to(device)
            y = model(x)
            img_y = torch.squeeze(y[-1]).cpu().numpy()
            acc,rand,info=compute_metrics(x,img_y)
            total_acc+=acc
            total_rand+=rand
            total_info+=info
    if isval==True and best_rand<total_rand/len(val_generator):
        best_rand=total_rand/len(val_generator)
    logger.info('Training at Epoch ' + str(epoch + 1) +
                ', Accuracy: ' + str(total_acc/len(val_generator)) +
                ', V_rand: ' + str(total_rand/len(val_generator)) +
                ', V_info: ' + str(total_info/len(val_generator)), file=log, flush=True)
    return best_rand,total_acc/len(val_generator),total_rand/len(val_generator),total_info/len(val_generator)


def train(model,criterion,optimizer,train_generator,val_generator,epoch):
    train_loss = []
    best_rand=0
    for epo in range(epoch):
        model=model.train()
        loss_val = 0
        start = time.time()
        for i, (d, p, label) in enumerate(train_generator):
            p = p.float().to(device)
            pred = model(d, p)
            label = Variable(torch.from_numpy(np.array(label)).long()).to(device)
            loss = criterion(pred, label)
            loss_val += loss.item() * label.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(loss_val)
        end = time.time()
        logger.info(' Epoch: ' + str(epo + 1) + '  Loss ' + str(loss_val) + ". Consumed Time " + str(int(end - start) / 60) + " mins", file=log, flush=True)
        val(model,train_generator,epoch,isval=False)
        best_rand,acc,rand,info=val(model,val_generator,best_rand,epoch)
    torch.save(model, log_dir + '/model.h5')


if __name__ == "__main__":
    logger=get_logger()
    logging.info('=======epoch:%s, batch size:%s ========' %(str(epochs), str(batch_size)))
    logger.info("Preparing for dataset and model")
    model=UNet().to(device)
    train_generator,validation_generator,test_generator=load_dataset()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    criterion=torch.nn.BCELoss()
    logger.info("Start Training and Validation")
    train(model,criterion,optimizer,train_generator,validation_generator,epochs)
    logger.info("Start Testing")
    test(model,test_generator,save_predict=True)