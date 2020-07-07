import sys
import os
from optparse import OptionParser
import numpy as np
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data

from predict import predict
from model import AMTA_Net
from dataset import create_folds, Dataset
from dice_loss import DiceLoss

primary_dir = '/home/datasets/prostate_bed/' # please modify the primary directory to your dataset folder
dir_img = primary_dir + 'IMG_slice/' # sub folder storing image slice files
dir_pb = primary_dir + 'PB_slice/' # sub folder storing prostate bed mask slice files
dir_oar = primary_dir + 'OAR_slice/' # sub folder storing OAR (i.e., bladder and rectum) mask slice files
dir_pb_volume = primary_dir + 'PB/' # sub folder storing prostate bed mask volume files
dir_oar_volume = primary_dir + 'OAR/' # sub folder storing OAR (i.e., bladder and rectum) mask volume files
dir_models = primary_dir + 'trained_models/' # output folder storing trained model files and predicted masks.

def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

def train_net(net,
              model_name,
              weight_pb,
              weight_oar,
              fold_num,
              skip_fold,
              adj_slice,
              epoch_num,
              batch_size,
              lr,
              resample_size,
              resample_spacing,
              min_hu,
              max_hu,
              num_workers,
              buffered_in_memory):

    train_start_time = time.localtime()

    # create folds
    folds, folds_size = create_folds(
        dir_img, 
        fold_num=fold_num,
        # you can exclude specific cases from the dataset like this
        exclude_case=[
            #'PB001.nii.gz',
            # ...
            #'PB00N.nii.gz'
            ], 
        slicewise=True)
    
    acc_time = 0
    time_stamp = time.strftime("%Y%m%d%H%M%S", train_start_time)

    # create directory for results storage
    store_dir = dir_models + 'model_{}/'.format(time_stamp)
    results_file_name = store_dir + 'results.txt'
    log_file_name = store_dir + 'log.txt'
    os.makedirs(store_dir, exist_ok=True)

    # print training info
    training_setting_lines = "Training settings:\n\
            Model name: {}\n\
            Model parameters in total: {}\n\
            Weights of PB/OAR task: {}/{}\n\
            Epoch num: {}\n\
            Batch size: {}\n\
            Learning rate: {}\n\
            Fold num: {}\n\
            Fold size: {}\n\
            Resample image size: {} x {}\n\
            Resample image spacing: {:4.3f} x {:4.3f}\n\
            Rescale intensity from [{:.1f}, {:.1f}] HU to [0.0, 1.0]\n\
            Number of adjacent slices: {}\n\
            CPU threads: {}\n\
            GPU used: {}\n\
            Buffered in memory: {}\n\
            Start time: {}\n".format(
                    model_name,
                    sum(x.numel() for x in net.parameters()), 
                    weight_pb, weight_oar, 
                    epoch_num, batch_size, lr, 
                    fold_num,
                    '/'.join(['%d']*len(folds_size)) % tuple(folds_size),
                    resample_size[0], resample_size[1],
                    resample_spacing[0], resample_spacing[1],
                    min_hu, max_hu,
                    adj_slice,
                    num_workers,
                    os.environ['CUDA_VISIBLE_DEVICES'],
                    buffered_in_memory,
                    time.strftime("%Y-%m-%d %H:%M:%S", train_start_time))
    print(training_setting_lines)
    log_file = open(log_file_name,'a')
    log_file.write(training_setting_lines)
    log_file.close()

    g_pb_dice_dict = {}
    g_oar1_dice_dict = {}
    g_oar2_dice_dict = {}
    global dir_oar

    # cross validation
    for fold_id in range(fold_num):
        # skip fold
        if fold_id in skip_fold:
            print("Skip fold {} of {}.".format(fold_id+1, fold_num))
            continue

        # choose testing fold and validation fold
        # rest folds for training
        test_fold_id = fold_id
        val_fold_id = (fold_id + 1) % fold_num
        test_ids = folds[test_fold_id]
        val_ids = folds[val_fold_id]
        train_ids = []
        for i in range(fold_num):
            if i != val_fold_id and i != test_fold_id:
                train_ids += folds[i]

        # create dataloader
        train_set = Dataset(
            ids=train_ids, 
            dir_img=dir_img, dir_pb=dir_pb, dir_oar=dir_oar, 
            resample_size=resample_size, 
            resample_spacing=resample_spacing, 
            min_hu=min_hu, max_hu=max_hu, 
            oar_labels=[1,2], adjacent=adj_slice,
            is_training=True, buffered_in_memory=buffered_in_memory)
        train_loader = data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)

        val_set = Dataset(
            ids=val_ids, 
            dir_img=dir_img, dir_pb=dir_pb, dir_oar=dir_oar, 
            resample_size=resample_size, 
            resample_spacing=resample_spacing, 
            min_hu=min_hu, max_hu=max_hu, 
            oar_labels=[1,2], adjacent=adj_slice,
            is_training=False, buffered_in_memory=buffered_in_memory)
        val_loader = data.DataLoader(
            dataset=val_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num_workers)

        test_set = Dataset(
            ids=test_ids, 
            dir_img=dir_img, dir_pb=dir_pb, dir_oar=dir_oar, 
            resample_size=resample_size, 
            resample_spacing=resample_spacing, 
            min_hu=min_hu, max_hu=max_hu, 
            oar_labels=[1,2], adjacent=adj_slice,
            is_training=False, buffered_in_memory=buffered_in_memory)
        test_loader = data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num_workers)
    
        # print fold info
        training_setting_lines = "Fold {} of {}:\n\
                Validation fold id: {}\n\
                Testing fold id: {}\n\
                Dataset size (Train/Val/Test): {} ({}/{}/{})\n".format(
                        fold_id + 1, fold_num,
                        val_fold_id, test_fold_id, 
                        len(train_ids) + len(val_ids) + len(test_ids),
                        len(train_ids), len(val_ids), len(test_ids))
        print(training_setting_lines)
        log_file = open(log_file_name,'a')
        log_file.write(training_setting_lines)
        log_file.close()

        # create loss function amd optimizer        
        criterion = DiceLoss()
        optimizer = optim.Adam(net.parameters(),
                            lr=lr,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0)

        best_val_acc = 0.0
        best_model_filename = store_dir + 'fold-{}_epoch_{}.pth.tar'.format(fold_id + 1, 1)
        initial_net(net)

        # training and validation
        for epoch in range(epoch_num):

            t0 = time.perf_counter()

            print('Starting epoch {}/{}.'.format(epoch + 1, epoch_num))
            net.train()

            train_loss = 0
            train_sample_num = 0

            # training for one epoch
            for batch_id, batch in enumerate(train_loader):
                # fetch data
                imgs = batch['data']
                gt_pb_mask = batch['pb_label']
                gt_oar_mask = batch['oar_label']
                n = len(imgs)

                # convert to GPU memory
                imgs = imgs.cuda()
                gt_pb_mask = gt_pb_mask.cuda()
                gt_oar_mask = gt_oar_mask.cuda()

                # forward propagation
                pd_pb_prob, pd_oar_prob = net(imgs)

                # compute loss
                loss_pb = criterion(pd_pb_prob, gt_pb_mask)
                loss_oar = criterion(pd_oar_prob, gt_oar_mask)
                loss = weight_pb * loss_pb + weight_oar * loss_oar
                train_loss += n * loss.item()
                train_sample_num += n

                print('Fold {0:d}/{1:d} --- Epoch {2:d}/{3:d} --- Progress {4:5.2f}% (+{5:d}) --- Loss: {6:.6f} ({7:.6f}/{8:.6f})'.format(
                    fold_id+1, fold_num, epoch+1, epoch_num, 100.0 * batch_id * batch_size / len(train_ids), n, loss.item(), loss_pb.item(), loss_oar.item()))
                
                # backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = train_loss / train_sample_num
            print('Fold {0:d}/{1:d} --- Epoch {2:d}/{3:d} --- Finished.'.format(fold_id+1, fold_num, epoch+1, epoch_num))
            print('Training loss: {:.6f}'.format(train_loss))

            # validation
            pb_dice_dict, oar1_dice_dict, oar2_dice_dict = predict(net=net, dataloader=val_loader, gt_pb_dir=dir_pb_volume, gt_oar_dir=dir_oar_volume, output_file=False, output_dir='')
            pb_dice = np.array(list(pb_dice_dict.values()), dtype=float)
            oar1_dice = np.array(list(oar1_dice_dict.values()), dtype=float)
            oar2_dice = np.array(list(oar2_dice_dict.values()), dtype=float)
            print('Validation accuracy (DSC [mean(std)%]): {:.3f}({:.3f})%, {:.3f}({:.3f})%, {:.3f}({:.3f})%'.format(
                pb_dice.mean()*100.0, pb_dice.std(ddof=1)*100.0, 
                oar1_dice.mean()*100.0, oar1_dice.std(ddof=1)*100.0, 
                oar2_dice.mean()*100.0, oar2_dice.std(ddof=1)*100.0))

            # output results to file
            t1 = time.perf_counter()
            epoch_t = t1 - t0
            acc_time += epoch_t
            print("Epoch time cost: {h:>02d}:{m:>02d}:{s:>02d}".format(h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

            val_acc = 0.5 * pb_dice.mean() + 0.25 * oar1_dice.mean() + 0.25 * oar2_dice.mean()
            results_file_line = '{epoch:>05d}\t{train_loss:>8.6f}\t{val_pb_acc:>8.6f}\t{val_oar1_acc:>8.6f}\t{val_oar2_acc:>8.6f}\t{val_acc:>8.6f}\n'.format(
                epoch=epoch+1,train_loss=train_loss,val_pb_acc=pb_dice.mean(),val_oar1_acc=oar1_dice.mean(),val_oar2_acc=oar2_dice.mean(),val_acc=val_acc)

            with open(results_file_name,'a') as results_file:
                results_file.write(results_file_line)

            # save best model
            if epoch == 0 or val_acc > best_val_acc:
                # remove former best model
                if os.path.exists(best_model_filename):
                    os.remove(best_model_filename)
                # save current best model
                best_val_acc = val_acc
                best_model_filename = store_dir + 'fold-{}_epoch_{}.pth.tar'.format(fold_id + 1, epoch + 1)                            
                torch.save({
                            'fold':fold_id,
                            'epoch':epoch,
                            'acc_time':acc_time,
                            'time_stamp':time_stamp,
                            'best_val_acc':best_val_acc,
                            'best_model_filename':best_model_filename,
                            'model_state_dict':net.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict()}, 
                            best_model_filename)
                print('Best model of fold-{} (epoch = {}) saved.'.format(fold_id + 1, epoch + 1))

        # test
        net.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
        pb_dice_dict, oar1_dice_dict, oar2_dice_dict = predict(net=net, dataloader=test_loader, gt_pb_dir=dir_pb_volume, gt_oar_dir=dir_oar_volume, output_file=True, output_dir=store_dir+'results/')
        g_pb_dice_dict.update(pb_dice_dict)
        g_oar1_dice_dict.update(oar1_dice_dict)
        g_oar2_dice_dict.update(oar2_dice_dict)
        pb_dice = np.array(list(pb_dice_dict.values()), dtype=float)
        oar1_dice = np.array(list(oar1_dice_dict.values()), dtype=float)
        oar2_dice = np.array(list(oar2_dice_dict.values()), dtype=float)
        results_file_line = 'Test results of fold-{fold_id:d}:\tPB: {pb_mean:.3f}({pb_std:.3f})%\tOAR1: {oar1_mean:.3f}({oar1_std:.3f})%\tOAR2: {oar2_mean:.3f}({oar2_std:.3f})%\n'''.format(
            fold_id=fold_id+1,
            pb_mean=pb_dice.mean()*100.0, pb_std=pb_dice.std(ddof=1)*100.0,
            oar1_mean=oar1_dice.mean()*100.0, oar1_std=oar1_dice.std(ddof=1)*100.0,
            oar2_mean=oar2_dice.mean()*100.0, oar2_std=oar2_dice.std(ddof=1)*100.0)

        with open(results_file_name,'a') as results_file:
            results_file.write(results_file_line + '\n')

        with open(log_file_name,'a') as log_file:
            log_file.write("\
                Finish time: {finish_time}\n\
                Accumulated training time: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
                    finish_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    h=int(acc_time) // 3600, m=(int(acc_time) % 3600) // 60, s=int(acc_time) % 60))

    # output global and case-wise results
    g_casename = list(g_pb_dice_dict.keys())
    g_pb_dice = np.array(list(g_pb_dice_dict.values()), dtype=float)
    g_oar1_dice = np.array(list(g_oar1_dice_dict.values()), dtype=float)
    g_oar2_dice = np.array(list(g_oar2_dice_dict.values()), dtype=float)
    results_file_line = '\nGlobal test results:\tPB: {pb_mean:.3f}({pb_std:.3f})%\tbladder: {oar1_mean:.3f}({oar1_std:.3f})%\trectum: {oar2_mean:.3f}({oar2_std:.3f})%\n'''.format(
        pb_mean=g_pb_dice.mean()*100.0, pb_std=g_pb_dice.std(ddof=1)*100.0,
        oar1_mean=g_oar1_dice.mean()*100.0, oar1_std=g_oar1_dice.std(ddof=1)*100.0,
        oar2_mean=g_oar2_dice.mean()*100.0, oar2_std=g_oar2_dice.std(ddof=1)*100.0)

    results_file_line += '\nCase-wise results:\n'
    for i in range(len(g_casename)):
        results_file_line += '{casename:<12s}\t{pb_dice:6.3f}%\t{oar1_dice:6.3f}%\t{oar2_dice:6.3f}%\n'.format(
            casename=g_casename[i],
            pb_dice=g_pb_dice[i]*100.0,
            oar1_dice=g_oar1_dice[i]*100.0,
            oar2_dice=g_oar2_dice[i]*100.0
        )

    with open(results_file_name,'a') as results_file:
        results_file.write(results_file_line + '\n')

    with open(log_file_name,'a') as log_file:
        log_file.write("\
            Finish time: {finish_time}\n\
            Total training time: {h:>02d}:{m:>02d}:{s:>02d}".format(
                finish_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                h=int(acc_time) // 3600, m=(int(acc_time) % 3600) // 60, s=int(acc_time) % 60))

if __name__ == '__main__':

    # you can use multiple GPUs like this
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    fold_num = 5 # 5-fold cross validation 
    skip_fold = [] # you can skip the training procedure on specific folds. e.g., skip_fold = [0,3,4] 
    epoch_num = 100 # training epochs
    batch_size = 120 # batch size
    learn_rate = 0.001 # base learning rate

    adjacent_slice_num = 1 # adjacent slices are combined with the center slice to compose a (1+2*adjacent_slice_num)-Channel input image.
    resample_size = [128, 128] # resample size of the input CT slices
    resample_spacing = [2.0, 2.0] # resample spacing (resolution) of the input CT slices
    # HU values in CT images are rescaled from [min_hu, max_hu] to [0, 1]
    # The values exceeding this range are cropped to 0 or 1.
    min_hu = -200.0 # minimum HU of the input CT slices
    max_hu = 800.0 # maximum HU of the input CT slices

    num_workers = 12 # you can use multiple CPU threads to speed up data loading.
    buffered_in_memory = True
    
    weight_pb = 1.0 # loss weight of prostate bed segmentation
    weight_oar = 1.0 # loss weight of OAR segmentation

    model = AMTA_Net(in_ch=adjacent_slice_num * 2 + 1)

    net = nn.DataParallel(module=model)
    net.cuda()

    try:
        train_net(net=net,
                  model_name=model.name(),
                  weight_pb = weight_pb,
                  weight_oar = weight_oar,
                  fold_num=fold_num,
                  skip_fold=skip_fold,
                  adj_slice=adjacent_slice_num,
                  epoch_num=epoch_num,
                  batch_size=batch_size,
                  lr=learn_rate,
                  resample_size=resample_size,
                  resample_spacing=resample_spacing, 
                  min_hu=min_hu, 
                  max_hu=max_hu,
                  num_workers=num_workers,
                  buffered_in_memory=buffered_in_memory)
                  
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
