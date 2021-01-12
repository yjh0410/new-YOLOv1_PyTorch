import torch
import torch.nn as nn
from utils import SPP, SAM, BottleneckCSP, Conv
from backbone import resnet18
import numpy as np
import tools

class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5, hr=False):
        super(myYOLO, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        self.grid_cell = self.create_grid(input_size)
        self.input_size = input_size
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()

        # we use resnet18 as backbone
        self.backbone = resnet18(pretrained=True)

        # neck
        self.SPP = nn.Sequential(
            Conv(512, 256, k=1),
            SPP(),
            BottleneckCSP(256*4, 512, n=1, shortcut=False)
        )
        self.SAM = SAM(512)
        self.conv_set = BottleneckCSP(512, 512, n=3, shortcut=False)

        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)
    
    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_boxes(self, pred):
        """
        input box :  [tx, ty, tw, th]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2
        
        return output

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        """
        bbox_pred: (HxW, 4), bsize = 1
        prob_pred: (HxW, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds

    def forward(self, x, target=None):
        # backbone
        _, _, C_5 = self.backbone(x)

        # head
        C_5 = self.SPP(C_5)
        C_5 = self.SAM(C_5)
        C_5 = self.conv_set(C_5)

        # pred
        prediction = self.pred(C_5)
        prediction = prediction.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
        B, HW, C = prediction.size()

        # Divide prediction to obj_pred, txtytwth_pred and cls_pred   
        # [B, H*W, 1]
        conf_pred = prediction[:, :, :1]
        # [B, H*W, num_cls]
        cls_pred = prediction[:, :, 1 : 1 + self.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = prediction[:, :, 1 + self.num_classes:]

        # test
        if not self.trainable:
            with torch.no_grad():
                # batch size = 1
                all_conf = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)
                
                # separate box pred and class conf
                all_conf = all_conf.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()
                
                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                return bboxes, scores, cls_inds
        else:
            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target)

            return conf_loss, cls_loss, txtytwth_loss, total_loss