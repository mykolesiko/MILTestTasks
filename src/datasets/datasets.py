import torch
import torch.nn as nn
import torch.optim as optim
#import torchtext
import random
import math
import time
import torch.nn.functional as F
import glob
import os


from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from  scipy import stats
import scipy
import cv2

import torchvision
import torch.optim as optim
import torchvision.models as models
from tqdm.notebook import tqdm
from torch.nn import functional as fnn

import pandas as pd
import PIL
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np

import glob
from pycocotools.coco import COCO
import more_itertools as mit


class LayoutDataset(Dataset):
    def __init__(self, root, transforms, json_file):
        super(LayoutDataset, self).__init__()
        self.root = root
        self.catNms = ['text', 'figure']
        self.coco = COCO(os.path.join(root, json_file))
        self.catIds = self.coco.getCatIds(catNms=['text', 'figure']);
        imgIds = self.coco.getImgIds(catIds=self.catIds);
        self.imgs = self.coco.loadImgs(imgIds)
        self.transforms = transforms
        self.json_file = json_file

    def __getitem__(self, idx):

        sample = {}
        img = self.imgs[idx]
        filename = os.path.join(self.root, "data/" + self.imgs[idx]['file_name'])
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image
        h = image.shape[0]
        w = image.shape[1]
        for s, catId in enumerate(self.catIds):
            annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=[catId], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            h = image.shape[0]
            w = image.shape[1]
            # print(anns)
            mask = np.zeros(shape=image.shape[:2])
            mask_boxes = np.zeros(shape=image.shape[:2])
            # boxes = []
            for i in range(len(anns)):
                npoints = int(len(anns[i]['segmentation'][0]) / 2)
                if self.json_file == 'train.json':
                    poly = np.array(anns[i]['segmentation'][0]) * np.array([w, h] * npoints)

                else:
                    poly = np.array(anns[i]['segmentation'][0])

                xmin = min(poly[::2])
                xmax = max(poly[::2])
                ymin = min(poly[1::2])
                ymax = max(poly[1::2])
                box = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
                points_box = np.array(list((mit.chunked(box, 2)))).astype(int)
                # print(points_box)
                points = np.array(list((mit.chunked(poly, 2)))).astype(int)
                cv2.fillConvexPoly(mask_boxes, points_box, 255)
                cv2.fillConvexPoly(mask, points, 255)

            mask = (mask.astype(np.float32) / 255.0) > 0.5
            mask_boxes = (mask_boxes.astype(np.float32) / 255.0) > 0.5

            if s == 0:
                masks = mask.reshape(1, h, w)
                masks_boxes = mask_boxes.reshape(1, h, w)
            else:
                masks = np.concatenate((masks, mask.reshape(1, h, w)), axis=0)
                masks_boxes = np.concatenate((masks_boxes, mask_boxes.reshape(1, h, w)), axis=0)


        sample['mask'] = masks
        sample['image'] = image
        sample['boxes'] = masks_boxes
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample['image'], sample['mask'], sample['image_cropped'], sample['boxes']
        # return image

    def __len__(self):
        return len(self.imgs)

