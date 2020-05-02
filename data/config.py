# config.py
import os.path


# yolo++ config
voc_af = {
    'num_classes': 20,
    'lr_epoch': (60, 90, 160),
    'max_epoch': 160,
    'min_dim': [416, 416],
    'name': 'VOC',
}

coco_af = {
    'num_classes': 80,
    'lr_epoch': (60, 90, 160),
    'max_epoch': 160,
    'min_dim': [416, 416],
    'name': 'VOC',
}