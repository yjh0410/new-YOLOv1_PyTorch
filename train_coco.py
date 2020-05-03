from __future__ import division

from utils.cocoapi_evaluator import COCOAPIEvaluator
from data import *
from utils.augmentations import SSDAugmentation
from data.cocodataset import *
import tools

import os
import random
import argparse
import time
import math
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser(description='YOLAC Detection')
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo.')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC or COCO dataset')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('-fl', '--use_focal', action='store_true', default=False,
                        help='use focal loss')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=4,
                        help='The upper bound of warm-up')
    parser.add_argument('--dataset_root', default='/home/k303/object-detection/dataset/COCO/', 
                        help='Location of VOC root directory')
    parser.add_argument('--num_classes', default=80, type=int, 
                        help='The number of dataset classes')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=0, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/coco/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()
    data_dir = args.dataset_root

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    hr = False  
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    
    cfg = coco_af

    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.multi_scale:
        print('Let us use the multi-scale trick.')
        input_size = [608, 608]
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=608,
                    transform=SSDAugmentation([608, 608], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                    debug=args.debug)
    else:
        input_size = cfg['min_dim']
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=cfg['min_dim'][0],
                    transform=SSDAugmentation(cfg['min_dim'], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                    debug=args.debug)

    # build model
    if args.version == 'yolo':
        from models.yolo import myYOLO
        yolo_net = myYOLO(device, input_size=input_size, num_classes=args.num_classes, trainable=True, hr=hr)
        print('Let us train yolo on the COCO dataset ......')

    else:
        print('We only support YOLO !!!')
        exit()

    
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the MSCOCO dataset...')
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")


    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    
    model = yolo_net
    model.to(device).train()

    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers)

    evaluator = COCOAPIEvaluator(
                    data_dir=data_dir,
                    img_size=cfg['min_dim'],
                    device=device,
                    transform=BaseTransform(cfg['min_dim'], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
                    )

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()

    for epoch in range(max_epoch):

        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # use step lr
        else:
            if epoch in cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
    
        # COCO evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            model.set_grid(cfg['min_dim'])
            # evaluate
            ap50_95, ap50 = evaluator.evaluate(model)
            print('ap50 : ', ap50)
            print('ap50_95 : ', ap50_95)
            # convert to training mode.
            model.trainable = True
            model.train()
            if args.tfboard:
                writer.add_scalar('val/COCOAP50', ap50, epoch + 1)
                writer.add_scalar('val/COCOAP50_95', ap50_95, epoch + 1)


        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
        
            # make labels
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=input_size, stride= yolo_net.stride, label_lists=targets)


            # to device
            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            if args.tfboard:
                # viz loss
                writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)

            if iter_i % 10 == 0:
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), input_size[0], t1-t0),
                        flush=True)

                t0 = time.time()

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                size = random.randint(10, 19) * 32
                input_size = [size, size]

                # change input dim
                # But this operation will report bugs when we use more workers in data loader, so I have to use 0 workers.
                # I don't know how to make it suit more workers, and I'm trying to solve this question.
                dataloader.dataset.reset_transform(SSDAugmentation(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))


        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')
                        )  

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
