''' Implementation for the training process of Network '''
import sys

sys.path.append("/data1/fanweijia/programming/vpcl/vpcl")
from general import Logger

import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import warning
# warning.filterwarning('ignore')
from torchtoolbox.optimizer import CosineWarmupLr

from torch.optim.lr_scheduler import MultiStepLR
from dataset import MXFaceDataset
from backbone import LResNet50EIR, LResNet34EIR, LResNet18EIR

import random

from torch.utils.tensorboard import SummaryWriter   


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Hyperparameters():
    def __init__(self):
        self.cuda = True
        self.cudnn = False
        self.visible_devices = "6"
        self.fp16 = True
        self.base_lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.gamma = 0.1
        self.resume = None
        # self.resume = './Models-LResNet50EIR/LResNet50EIR_1th_checkpoint.tar'
        self.finetune = None
        # self.finetune = './Models-LResNet50EIR/LResNet50EIR_2th_epoch.pth'
        self.data_path = 'Your dataset path'
        self.img_size = [112, 112]
        self.train_batch_size = 512
        self.bs_mult = 1
        self.drop_last = True
        self.steps = [16, 24]
        self.start_epoch = 0
        self.epochs = 28
        self.warmups = 0
        self.display = 100.0
        self.workers = 32
        self.num_classes = 10572
        self.model_name = 'LResNet50EIR'
        self.model_dir = 'R50_casia_cosface_epl'
        self.log_dir = 'log-LResNet50EIR'
        

def main():
    global params
    
    ''' optionally finetune from a pre-trained model '''
    if params.finetune is not None:
        model = torch.load(params.finetune)
        print("=> load pre-trained model '{}'\n".format(params.finetune))
    else:
        ''' create Network for face recognition '''
        model = eval(params.model_name)(num_classes=params.num_classes)
    # print(model)
    print()

    model_params = []
    for name, value in model.named_parameters():
        model_params += [{'params': value}]

    ''' define loss function and optimizer '''
    optimizer = torch.optim.SGD(model_params, params.base_lr, momentum=params.momentum, weight_decay=params.weight_decay, nesterov=False)

    if params.cuda:
        model = nn.DataParallel(model).cuda()
        # model = model.cuda()
        
        # cudnn.benchmark = params.cudnn
        # net = model.module
        net = model.module
    else:
        net = model.cpu()

    ''' optionally resume from a checkpoint '''
    if params.resume is not None:
        checkpoint = torch.load(params.resume)
        params.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        print("=> resume from checkpoint '{}'\n".format(params.resume))

    ''' load image '''
    train_set = MXFaceDataset(root_dir=params.data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=params.train_batch_size*params.bs_mult, shuffle=True,
        num_workers=params.workers, pin_memory=True, drop_last=params.drop_last,
        worker_init_fn=seed_worker, prefetch_factor=2, persistent_workers=True)

    if os.path.exists(params.model_dir) is False:
        os.makedirs(params.model_dir)

    scheduler = MultiStepLR(optimizer, milestones=[v + params.warmups for v in params.steps], gamma=params.gamma, last_epoch=params.start_epoch-1)

    print(scheduler)
    if params.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(params.start_epoch, params.epochs):

        lr_scale = 1.0 * np.clip(max(epoch, 1e-6) * 10.0 / max(params.warmups, 1e-6), 1, 10) / 10
        loss_scale = 1.0 / params.bs_mult

        for param_group, lr in zip(optimizer.param_groups, scheduler._get_closed_form_lr()):
            param_group['lr'] = lr * lr_scale

        real_lr = optimizer.param_groups[-1]['lr']
        virtual_epoch = epoch + 1
        print('Epoch: {}\n'.format(virtual_epoch))

        ''' train for one epoch '''
        train(train_loader, model, optimizer, epoch, loss_scale, scaler)
        scheduler.step()

        if epoch>=int(params.epochs*0.8):
            save_name = params.model_dir + '/' + params.model_name + '_' + str(virtual_epoch) + 'th_epoch.pth'
            torch.save(net, save_name)
            save_name = params.model_dir + '/' + params.model_name + '_' + str(virtual_epoch) + 'th_checkpoint.tar'
            torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer' : optimizer.state_dict()}, save_name)


def train(train_loader, model, optimizer, epoch, loss_scale, scaler):
    global params
    model.train()
    if params.cuda:
        # net = model.module
        net = model.module
    else:
        net = model

    acc = 0; loss = 0; loss_ = 0; count = 0

    for i, (data_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        if params.cuda:
            datas, labels = data_batch.cuda(), label_batch.cuda()
            if scaler is None:
                face_loss = model(datas, labels, epoch)
            else:
                with torch.cuda.amp.autocast():
                    face_loss = model(datas, labels, epoch)
        if torch.cuda.device_count() > 1:
            face_loss = face_loss.mean()
        if scaler is None:
            face_loss.backward()
            optimizer.step()
        else:
            scaler.scale(face_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        net.restrict_weights()
        if (i % params.display == 0 or (i + 1) == len(train_loader)):
            print('{}, Iteration: {} ({}/{}={:.0f}%)  loss: {:.4f}  lr: {:.4f}\n'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i, params.train_batch_size*params.bs_mult*i+label_batch.size(0),
                    len(train_loader.dataset), 100.0 * (i + 1) / len(train_loader), face_loss.data, optimizer.param_groups[-1]['lr']))


if __name__ == '__main__':
    params = Hyperparameters()
    sys.stdout = Logger(os.path.join(params.model_dir, 'log_.txt'))
    
    print('Hyperparameters:')
    print('  cuda: {}\n  cudnn: {}\n  device_id: {}\n  fp16: {}\n  base_lr: {}\n  momentum: {}\n  weight_decay: {}\n  gamma: {}\n'
          '  data_path: {}\n  img_size: {} \n  batch_size: {}\n  bs_mult: {}\n  drop_last: {}\n  steps: {}\n  epochs: {}\n'
          '  warmups: {}\n  workers: {}\n  num_classes: {}\n  model_name: {}\n  model_dir: {}\n  log_dir: {}\n'.format(
          params.cuda, params.cudnn, params.visible_devices, params.fp16, params.base_lr, params.momentum, params.weight_decay, params.gamma,
          params.data_path, params.img_size, params.train_batch_size, params.bs_mult, params.drop_last, params.steps, params.epochs,
          params.warmups, params.workers, params.num_classes, params.model_name, params.model_dir, params.log_dir))

    os.environ["CUDA_VISIBLE_DEVICES"] = params.visible_devices

    setup_seed(seed=1, cuda_deterministic=True)
    main()
