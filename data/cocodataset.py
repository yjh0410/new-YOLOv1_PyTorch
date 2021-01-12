import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO


coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

coco_root = '/home/k303/object-detection/dataset/COCO/'


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, data_dir='COCO', json_file='instances_train2017.json',
                 name='train2017', img_size=416,
                 transform=None, min_size=1, debug=False, mosaic=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.transform = transform
        self.mosaic = mosaic


    def __len__(self):
        return len(self.ids)


    def pull_image(self, index):
        id_ = self.ids[index]
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        elif self.json_file == 'image_info_test-dev2017.json' and img is None:
            img_file = os.path.join(self.data_dir, 'test2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        elif self.json_file == 'image_info_test2017.json' and img is None:
            img_file = os.path.join(self.data_dir, 'test2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        return img, id_


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt


    def pull_item(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)
        
        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        assert img is not None

        height, width, channels = img.shape
        
        # COCOAnnotation Transform
        # start here :
        target = []
        for anno in annotations:
            x1 = np.max((0, anno['bbox'][0]))
            y1 = np.max((0, anno['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
            if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                label_ind = anno['category_id']
                cls_id = self.class_ids.index(label_ind)
                x1 /= width
                y1 /= height
                x2 /= width
                y2 /= height

                target.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
        # end here .

        # mosaic augmentation
        if self.mosaic and np.random.randint(2):
            ids_list_ = self.ids[:index] + self.ids[index+1:]
            # random sample 3 indexs
            id2, id3, id4 = random.sample(ids_list_, 3)
            ids = [id2, id3, id4]
            img_lists = [img]
            tg_lists = [target]
            # load other 3 images and targets
            for id_ in ids:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
                annotations = self.coco.loadAnns(anno_ids)

                # load image and preprocess
                img_file = os.path.join(self.data_dir, self.name,
                                        '{:012}'.format(id_) + '.jpg')
                img_i = cv2.imread(img_file)
                
                if self.json_file == 'instances_val5k.json' and img_i is None:
                    img_file = os.path.join(self.data_dir, 'train2017',
                                            '{:012}'.format(id_) + '.jpg')
                    img_i = cv2.imread(img_file)

                assert img_i is not None

                height_i, width_i, channels_i = img_i.shape             
                # COCOAnnotation Transform
                # start here :
                target_i = []
                for anno in annotations:
                    x1 = np.max((0, anno['bbox'][0]))
                    y1 = np.max((0, anno['bbox'][1]))
                    x2 = np.min((width_i - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
                    y2 = np.min((height_i - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
                    if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                        label_ind = anno['category_id']
                        cls_id = self.class_ids.index(label_ind)
                        x1 /= width_i
                        y1 /= height_i
                        x2 /= width_i
                        y2 /= height_i

                        target_i.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
                # end here .
                img_lists.append(img_i)
                tg_lists.append(target_i)

            mosaic_img = np.zeros([self.img_size*2, self.img_size*2, img.shape[2]], dtype=np.uint8)
            # mosaic center
            yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]

            mosaic_tg = []
            for i in range(4):
                img_i, target_i = img_lists[i], tg_lists[i]
                h0, w0, _ = img_i.shape

                # resize image to img_size
                r = self.img_size / max(h0, w0)
                if r != 1:  # always resize down, only resize up if training with augmentation
                    img_i = cv2.resize(img_i, (int(w0 * r), int(h0 * r)))
                h, w, _ = img_i.shape

                # place img in img4
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b

                # labels
                target_i = np.array(target_i)
                target_i_ = target_i.copy()
                if len(target_i) > 0:
                    # a valid target, and modify it.
                    target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                    target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                    target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                    target_i_[:, 3] = (h * (target_i[:, 3]) + padh)     
                    
                    mosaic_tg.append(target_i_)

            if len(mosaic_tg) == 0:
                mosaic_tg = np.zeros([1, 5])
            else:
                mosaic_tg = np.concatenate(mosaic_tg, axis=0)
                # Cutout/Clip targets
                np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
                # normalize
                mosaic_tg[:, :4] /= (self.img_size * 2)

            # augment
            mosaic_img, boxes, labels = self.transform(mosaic_img, mosaic_tg[:, :4], mosaic_tg[:, 4])
            # to rgb
            mosaic_img = mosaic_img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            mosaic_tg = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            scale =  np.array([[1., 1., 1., 1.]])
            offset = np.zeros([1, 4])

            return torch.from_numpy(mosaic_img).permute(2, 0, 1).float(), mosaic_tg, self.img_size, self.img_size

        # data augmentation
        if self.transform is not None:
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

if __name__ == "__main__":
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels

    img_size = 640
    dataset = COCODataset(
                data_dir='/home/k303/object-detection/dataset/COCO/',
                img_size=img_size,
                transform=BaseTransform([img_size, img_size], (0, 0, 0)),
                debug=False,
                mosaic=True)
    
    for i in range(1000):
        im, gt, h, w = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        print(img.shape)
        cv2.imwrite('-1.jpg', img)
        img = cv2.imread('-1.jpg')

        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', img)
        cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
