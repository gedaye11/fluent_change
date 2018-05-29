from __future__ import division
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import scipy.io as sio
import array
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def load_image(image_path, descriptor_name):
    if descriptor_name == 'vgg16':
        sz = 224
    if descriptor_name == 'vgg19':
        sz = 224
    if descriptor_name == 'resnet50':
        sz = 224
    if descriptor_name == 'resnet101':
        sz = 224
    if descriptor_name == 'resnet152':
        sz = 224
    image = Image.open(image_path)
    # Image preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    scaler = transforms.Resize((sz, sz))
    to_tensor = transforms.ToTensor()
    image = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))
#    image = image.repeat(1, 3, 1, 1)
    # print(image)
    if use_cuda:
        image = image.cuda()
    return image

class Net(nn.Module):
    def __init__(self, descriptor_name):
        super(Net, self).__init__()

        # if descriptor_name == 'vgg16':
        #     self.select = ['30']
        #     self.vgg16 = models.vgg16(pretrained=True)
        #     self.sequence = []
        #     for name, layer in self.vgg16.features._modules.items():
        #         self.sequence += [layer]
        #     for name, layer in self.vgg16.classifier._modules.items():
        #         self.sequence += [layer]
        #         break
        #     self.model = nn.Sequential(*self.sequence)

        if descriptor_name == 'vgg16':
            self.select = ['30']
            self.vgg16 = models.vgg16(pretrained=True)
            self.sequence = []
            for name, layer in self.vgg16.features._modules.items():
                self.sequence += [layer]
            for name, layer in self.vgg16.classifier._modules.items():
                if name == '6':
                    break
                self.sequence += [layer]
            layer = nn.Linear(4096, 10)
            # init.xavier_normal(layer.weight.data, gain = 1)
            self.sequence += [layer]

            self.model = nn.Sequential(*self.sequence)

        elif descriptor_name == 'vgg19':
            self.select = ['36']
            self.vgg19 = models.vgg19(pretrained=True)
            self.sequence = []
            for name, layer in self.vgg19.features._modules.items():
                self.sequence += [layer]
            for name, layer in self.vgg19.classifier._modules.items():
                self.sequence += [layer]
                break
            self.model = nn.Sequential(*self.sequence)

        elif descriptor_name == 'resnet50':
            self.select = ['avgpool']
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 10)
            
        elif descriptor_name == 'resnet101':
            self.select = ['avgpool']
            self.model = models.resnet101(pretrained=True)

        elif descriptor_name == 'resnet152':
            self.select = ['avgpool']
            self.model = models.resnet152(pretrained=True)
            self.model.fc = nn.Linear(2048, 10)


    def forward(self, x, descriptor_name):
        if descriptor_name == 'resnet50':
            for name, layer in self.model._modules.items():
                # print(name, x.size())
                x = layer(x)
                if name in self.select:
                    return x

        if descriptor_name == 'resnet101':
            for name, layer in self.model._modules.items():
                # print(name, x.size())
                x = layer(x)
                if name in self.select:
                    return x

        if descriptor_name == 'resnet152':
            for name, layer in self.model._modules.items():
                # print(name, x.size())
                x = layer(x)
                if name in self.select:
                    return x

        if descriptor_name == 'vgg16':
            for name, layer in self.model._modules.items():
                # print(name)
                x = layer(x)
                if name in self.select:
                    x = x.view(1, -1)
                if name == '31':
                    return x

        if descriptor_name == 'vgg19':
            for name, layer in self.model._modules.items():
                x = layer(x)
                if name in self.select:
                    x = x.view(1, -1)
                    return x


if __name__ == "__main__":
    # dataset_name = 'CIFAR10'
    # dataset_dir = './data/CIFAR10'
    descriptor_name = 'resnet50'

    # train_dir = dataset_dir + '/train'
    # test_dir = dataset_dir + '/test'
    #
    # train_result_dir = '../dataset/CIFAR10/train'
    # test_result_dir = '../dataset/CIFAR10/test'

    net = Net(descriptor_name)
    print(net)
    
    # model_dir = "fine_model/{}/{}_epoch_45.pth".format(dataset_name, descriptor_name)
    # model_dict = torch.load(model_dir)
    # net.load_state_dict(model_dict)

    net.eval()
    if use_cuda:
        net.cuda()

    # train_files = os.listdir(train_dir)
    # for file in train_files:
    #     if os.path.isfile(train_dir + '/' + file):
    #         image = load_image(train_dir + '/' + file, descriptor_name)
    #         feature = net(image, descriptor_name)
    #         feature = feature.view(-1)
    #         # print(feature)
    #         feature = nn.functional.normalize(feature, dim = 0)
    #         feature = feature.cpu().data.numpy()
    #
    #         mat_name = train_result_dir + '/' + descriptor_name + '/' + file + '.mat'
    #         sio.savemat(mat_name, {descriptor_name : feature})
    #         print('Extract ' + mat_name + ' done...')
    #         # break

    # ima_files = os.listdir('/home/gexuri/project/fluent-cap-master/weibo/pic/')
    #fw = open('/home/gexuri/VisualSearch/weibotest/FeatureData/pyresnet152-pool5osl2/feature.bin', 'wb')
    #wff = open('/home/gexuri/VisualSearch/weibotest/FeatureData/pyresnet152-pool5osl2/wrong_pic.txt', 'w')

    fw = open('/home/gexuri/VisualSearch/weibotrain/FeatureData/pyresnet152-pool5osl2/feature-50.bin','wb')
    wff = open('/home/gexuri/VisualSearch/weibotrain/FeatureData/pyresnet152-pool5osl2/wrong_pic.txt', 'w')
    #wff = open('wrong_pic.txt', 'w')
    res = array.array('f')

    #rf = open('/home/gexuri/VisualSearch/weibotest/ImageSets/weibotest.txt', 'r')
    #rf = open('pic_dir_wf.txt', 'r')
    rf = open('/home/gexuri/VisualSearch/weibotrain/FeatureData/pyresnet152-pool5osl2/id.txt', 'r')
    line = rf.readline()
    rf.close()
    pic_files = line.strip().split()


    for ima_fil in tqdm(pic_files):
        try:
            image = load_image('/home/cfh3c/Datas/weibo/huati0/data/' + ima_fil, descriptor_name)
            # image = load_image('/home/gexuri/VisualSearch/flickr30k-images/' + ima_fil+ '.jpg', descriptor_name)
            feature = net(image, descriptor_name)
            feature = feature.view(-1)
            # print(feature)
            feature = nn.functional.normalize(feature, dim = 0)
            feature = feature.cpu().data.numpy().tolist()
            # print len(feature)
            res.extend(feature)
        except:
            wff.write(ima_fil + '\n')
        # break
    wff.close()
    res.tofile(fw, )
    fw.close()



            # test_files = os.listdir(test_dir)
    # for file in test_files:
    #     if os.path.isfile(test_dir + '/' + file):
    #         image = load_image(test_dir + '/' + file, descriptor_name)
    #         feature = net(image, descriptor_name)
    #         feature = feature.view(-1)
    #         feature = nn.functional.normalize(feature, dim = 0)
    #         feature = feature.cpu().data.numpy()
    #         mat_name = test_result_dir + '/' + descriptor_name + '/' + file + '.mat'
    #         sio.savemat(mat_name, {descriptor_name : feature})
    #         print('Extract ' + mat_name + ' done...')
