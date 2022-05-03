import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import cv2

import torch.optim as optim
import torchvision.models as models
from torch.nn import functional as fnn
import pandas as pd
import PIL
import os
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
from src.utils import dice_coeff, jaccard_coeff, jaccard_loss
from src.transforms import train_transforms, val_transforms
from src.datasets import LayoutDataset

class Model:
    def __init__(self, model_path:str, device):
        self.model_path = model_path
        self.device = device
        self.get_model()

    def get_model(self):
        self.model = smp.DeepLabV3(encoder_name='resnet101', encoder_depth=5, \
                     encoder_weights='imagenet', decoder_channels=256, in_channels=3, classes=2, \
                     activation=None, upsampling=8, aux_params=None)

    def validate(self):
            self.model.eval().to(self.device)
            val_dice = []
            val_jaccard = []
            for batch in tqdm(self.val_dataloader):
                images, true_masks, _, _ = batch
                with torch.no_grad():
                    masks_pred = self.model(images.to(self.device)).squeeze(1)  # (b, 1, h, w) -> (b, h, w)
                masks_pred = (torch.sigmoid(masks_pred) > 0.5).float()  # * 255

                dice = dice_coeff(masks_pred.cpu(), true_masks).item()
                val_dice.append(dice)
                val_jaccard.append((jaccard_coeff(masks_pred.cpu()[:, 0, :, :], true_masks[:, 0, :, :]),
                                    jaccard_coeff(masks_pred.cpu()[:, 1, :, :], true_masks[:, 1, :, :])))
            return np.mean(val_dice), np.mean(val_jaccard, axis=0)
    def train_proc(self):
        self.model.train()
        epoch_losses = []
        epoch_bce_losses, epoch_dice_losses, const_loss = [], [], []

        history = []
        tqdm_iter = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for i, batch in tqdm_iter:
            imgs, true_masks, _, boxes = batch
            masks_pred = self.model(imgs.to(self.device))
            masks_probs = torch.sigmoid(masks_pred)
            bce_loss_value = self.criterion[0](masks_probs.cpu().view(-1), true_masks.view(-1))
            jaccard_loss_value = self.criterion[1](masks_probs.cpu(), true_masks)
            loss = bce_loss_value + jaccard_loss_value

            epoch_bce_losses.append(bce_loss_value.item())
            epoch_dice_losses.append(jaccard_loss_value.item())
            epoch_losses.append(loss.item())
            tqdm_iter.set_description(f"mean loss: {np.mean(epoch_losses):.4f}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            history.append(loss.cpu().data.numpy())

            #logger.info(
               # f"Epoch finished! Loss: {np.mean(epoch_losses):.5f} ({np.mean(epoch_bce_losses):.5f} | {np.mean(epoch_dice_losses):.5f})")

        return np.mean(epoch_losses)
    def train_loop(self, args):
        #args.epochs = 40
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=2, threshold=0.001,
                                                         threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
                                                         verbose=True)
        best_model_info = {"epoch": -1, "val_dice": 0., "train_dice": 0., "train_loss": 0.}
        for epoch in range(args.epochs):
            #logger.info(f"Starting epoch {epoch + 1}/{args.epochs}.")

            train_loss = self.train_proc()#model, optimizer, criterion, scheduler, train_dataloader, logger, False, device)
            print(train_loss)
            val_dice = self.validate()#model, val_dataloader, device)
            #print(val_dice)
            self.scheduler.step(train_loss)
            if val_dice[0] > best_model_info["val_dice"]:
                best_model_info["val_dice"] = val_dice[0]
                best_model_info["train_loss"] = train_loss

                best_model_info["epoch"] = epoch
                with open(os.path.join(self.model_path, args.model_name), "wb") as fp:
                    torch.save(self.model.state_dict(), fp)
                #logger.info(f"Validation Dice Coeff: {val_dice[0]:.3f} (best)")
            #else:
                #logger.info(f"Validation Dice Coeff: {val_dice[0]:.5f} (best {best_model_info['val_dice']:.5f})")

        #with open(os.path.join(args.output_dir, "CP-last_seg.pth"), "wb") as fp:
        #    torch.save(model.state_dict(), fp)

    def initialize_weights(self, m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)


    def train(self, args):
        self.train_transforms = train_transforms
        self.data_path = args.data_dir
        self.model.requires_grad_(True)

        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.segmentation_head.parameters():
            param.requires_grad = True

        self.model.segmentation_head.apply(self.initialize_weights)
        self.model.to(self.device)
        #logger.info(f"Model type: {model.__class__.__name__}")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=0.001,
        #                                                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
        #                                                 verbose=True)


        criterion1 = lambda x, y: (args.weight_bce * nn.BCELoss()(x, y))
        criterion2 = lambda x, y: (1. - args.weight_bce) * jaccard_loss(x, y)
        self.criterion = (criterion1, criterion2)

        self.train_dataset = LayoutDataset(args.data_dir, self.train_transforms, "train.json")
        #val_dataset = LayoutDataset(self.data_path, val_transforms, "test.json")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=1,
                                      pin_memory=True, shuffle=True,
                                      drop_last=True)  # ,worker_init_fn=seed_worker,   generator=g,)
        #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1,
           #                         pin_memory=True, shuffle=False,
          #                          drop_last=False)  # , worker_init_fn=seed_worker,   generator=g,)

        #logger.info(f"Length of train / val = {len(train_dataset)} / {len(val_dataset)}")
        #logger.info(f"Number of batches of train / val = {len(train_dataloader)} / {len(val_dataloader)}")
        self.train_loop(args)

    def load_model(self, model_file: str):
        self.get_model()
        model_state = torch.load(os.path.join(self.model_path, model_file), map_location=torch.device('cpu') )
        self.model.load_state_dict(model_state)


    def predict(self, args):
        self.model_path = args.model_dir
        self.load_model(args.model_name)
        self.val_dataset = LayoutDataset(args.data_dir, val_transforms, "test.json")
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=16, num_workers=1,\
                                    pin_memory=True, shuffle=False,\
                                    drop_last=False)  # , worker_init_fn=seed_worker,   generator=g,)
        val_dice, val_jaccard = self.validate()
        return val_dice, val_jaccard












