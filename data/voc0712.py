"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
path_to_dir = osp.dirname(osp.abspath(__file__))
VOC_ROOT = path_to_dir + "/VOCdevkit/"
VOC_ROOT = "/home/k303/object-detection/dataset/VOCdevkit/"


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, img_size,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712', mosaic=False):
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.mosaic = mosaic
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # mosaic augmentation
        if self.mosaic and np.random.randint(2):
            ids_list_ = self.ids[:index] + self.ids[index+1:]
            # random sample 3 indexs
            id2, id3, id4 = random.sample(ids_list_, 3)
            ids = [id2, id3, id4]
            img_lists = [img]
            tg_lists = [target]
            for id_ in ids:
                img_ = cv2.imread(self._imgpath % id_)
                height_, width_, channels_ = img_.shape

                target_ = ET.parse(self._annopath % id_).getroot()              
                target_ = self.target_transform(target_, width_, height_)

                img_lists.append(img_)
                tg_lists.append(target_)
            
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

        # basic augmentation(SSDAugmentation or BaseTransform)
        if self.transform is not None:
            # check labels
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
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

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
    # dataset
    dataset = VOCDetection(VOC_ROOT, img_size, [('2007', 'trainval')],
                            BaseTransform([img_size, img_size], (0, 0, 0)),
                            VOCAnnotationTransform(), mosaic=True)
    for i in range(1000):
        im, gt, h, w = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
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
        cv2.waitKey(0)
