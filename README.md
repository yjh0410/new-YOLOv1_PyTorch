# new-YOLOv1_PyTorch
In this project, you can enjoy: 
- a new version of yolov1


# Network
This is a a new version of YOLOv1 built by PyTorch:
- Backbone: resnet18
- Head: SPP, SAM

# Train
- Batchsize: 32
- Base lr: 1e-3
- Max epoch: 160
- LRstep: 60, 90
- optimizer: SGD

Before I tell you how to use this project, I must say one important thing about difference between origin yolo-v2 and mine:

- For data augmentation, I copy the augmentation codes from the https://github.com/amdegroot/ssd.pytorch which is a superb project reproducing the SSD. If anyone is interested in SSD, just clone it to learn !(Don't forget to star it !)

So I don't write data augmentation by myself. I'm a little lazy~~

My loss function and groundtruth creator both in the ```tools.py```, and you can try to change any parameters to improve the model.

## Experiment
Environment:

- Python3.6, opencv-python, PyTorch1.1.0, CUDA10.0,cudnn7.5
- For training: Intel i9-9940k, TITAN-RTX-24g
- For inference: Intel i5-6300H, GTX-1060-3g

VOC:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> mAP </td><td bgcolor=white> FPS </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 320 </td><td bgcolor=white> 64.4 </td><td bgcolor=white> - </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 68.5 </td><td bgcolor=white> - </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 608 </td><td bgcolor=white> 71.5 </td><td bgcolor=white> - </td></tr>
</table></tbody>

COCO:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 320 </td><td bgcolor=white> 14.50 </td><td bgcolor=white> 30.15 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 416 </td><td bgcolor=white> 17.34 </td><td bgcolor=white> 35.28 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 608 </td><td bgcolor=white> 19.90 </td><td bgcolor=white> 39.27 </td></tr>
</table></tbody>

## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

## Dataset
As for now, I only train and test on PASCAL VOC2007 and 2012. 

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is:

- ```data/VOCdevkit/VOC2007```
- ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### MSCOCO Dataset
I copy the download files from the following excellent project:
https://github.com/DeNA/PyTorch_YOLOv3

#### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017:

- ```data/COCO/annotations/```
- ```data/COCO/train2017/```
- ```data/COCO/val2017/```
- ```data/COCO/test2017/```


## Train
### VOC
```Shell
python train_voc.py -ms --cuda
```

```-ms``` means you select multi-scale training trick, else cancel it.

You can run ```python train_voc.py -h``` to check all optional argument.

By default, I set num_workers in pytorch dataloader as 0 to guarantee my multi-scale trick. But the trick can't work when I add more wokers. I know little about multithreading. So sad...

### COCO
```Shell
python train_coco.py -ms --cuda
```

## Test
### VOC
```Shell
python test_voc.py --trained_model [ Please input the path to model dir. ] --cuda
```

### COCO
```Shell
python test_coco.py --trained_model [ Please input the path to model dir. ] --cuda
```


## Evaluation
### VOC
```Shell
python eval_voc.py --train_model [ Please input the path to model dir. ] --cuda
```

### COCO
To run on COCO_val:
```Shell
python eval_coco.py --train_model [ Please input the path to model dir. ] --cuda
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval_coco.py --train_model [ Please input the path to model dir. ] --cuda -t
```
You will get a .json file which can be evaluated on COCO test server.
