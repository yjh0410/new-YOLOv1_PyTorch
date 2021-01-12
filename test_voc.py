import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import VOC_ROOT, VOC_CLASSES, VOCDetection, BaseTransform, VOC_CLASSES
import numpy as np
import cv2
import tools
import time


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo',
                    help='yolo.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('--trained_model', default='weights/voc/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input size')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()


print("----------------------------------------Object Detection--------------------------------------------")

def test_net(net, device, testset, transform, thresh, mode='voc'):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img = testset.pull_image(index)
        # img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        bbox_pred, scores, cls_inds = net(x)      # forward pass
        t1 = time.time()
        print("detection time used ", t1 - t0, "s")
        # scale each detection back up to the image
        scale = np.array([[img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]]])
        # map the boxes to origin image scale
        bbox_pred *= scale

        CLASSES = VOC_CLASSES
        class_color = tools.CLASS_COLOR
        for i, box in enumerate(bbox_pred):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            # print(xmin, ymin, xmax, ymax)
            if scores[i] > thresh:
                box_w = int(xmax - xmin)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmin+box_w*0.55), int(ymin)), class_color[int(cls_indx)], -1)
                mess = '%s: %.3f' % (CLASSES[int(cls_indx)], scores[i])
                cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.imshow('detection', img)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)



def test():
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load net
    input_size = [args.input_size, args.input_size]
    num_classes = 20
    testset = VOCDetection(VOC_ROOT, img_size=None, image_sets=[('2007', 'test')], transform=None)

    # build model
    if args.version == 'yolo':
        from models.yolo import myYOLO
        net = myYOLO(device, input_size=input_size, num_classes=num_classes, trainable=False)
        print('Let us test yolo on the VOC0712 dataset ......')

    else:
        print('Unknown Version !!!')
        exit()


    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Finished loading model!')

    net = net.to(device)

    # evaluation
    test_net(net, device, testset,
             BaseTransform(net.input_size),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test()
