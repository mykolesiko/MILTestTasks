from torchvision import transforms
import albumentations
from src.constants import CROP_SIZE
import numpy as np
import cv2
import torch


class ScaleMaxSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
        # self.size = torch.tensor(size, dtype=torch.float)
        self.size = np.asarray(size, dtype=np.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h < w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        interpolation = cv2.INTER_AREA if f <= 1 else cv2.INTER_LINEAR
        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=interpolation)
        masks_resized = []
        masks_boxes_resized = []
        for i in range(2):
            mask = sample['mask'][i].astype('float32')
            # print(mask.shape)
            mask = cv2.resize(mask, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            masks_resized.append(mask)
            mask_boxes = sample['boxes'][i].astype('float32')
            # print(mask.shape)
            mask_boxes = cv2.resize(mask_boxes, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            masks_boxes_resized.append(mask_boxes)
        sample['boxes'] = np.array(masks_boxes_resized)
        sample['mask'] = np.array(masks_resized)
        # sample['boxes'] = np.array(sample['boxes']).astype(np.float32) * f

        # print(sample['mask'].shape)
        # sample["scale_coef"] = f

        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name].copy()
        # print(type(img[0, 0, 0]))
        mask = sample['mask'].copy()
        mask_boxes = sample['boxes'].copy()
        h, w, _ = img.shape
        margin_h = (self.size - h) // 2
        margin_w = (self.size - w) // 2
        sample[self.elem_name] = np.full((CROP_SIZE, CROP_SIZE, 3), 255, np.uint8)
        sample[self.elem_name][margin_h:margin_h + h, margin_w:margin_w + w] = img
        sample['mask'] = np.zeros((2, CROP_SIZE, CROP_SIZE), np.float32)
        sample['mask'][:, margin_h:margin_h + h, margin_w:margin_w + w] = mask
        sample['boxes'] = np.zeros((2, CROP_SIZE, CROP_SIZE), np.float32)
        sample['boxes'][:, margin_h:margin_h + h, margin_w:margin_w + w] = mask_boxes
        # print(sample['mask'].shape)
        # sample["crop_margin_x"] = margin_w
        # sample["crop_margin_y"] = margin_h
        sample['image_cropped'] = torch.LongTensor(sample[self.elem_name].copy())
        # for i in range(len(sample['boxes'])):
        #     sample['boxes'][i][0] += margin_w
        #     sample['boxes'][i][2] += margin_w
        #     sample['boxes'][i][1] += margin_h
        #     sample['boxes'][i][3] += margin_h

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        # print(sample['mask'].shape)
        # print(sample['image'])
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])
        # print(sample['mask'].shape)
        # print(sample['image'])

        return sample


from torchvision import transforms
import albumentations


class apply_albumentations(object):
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def __call__(self, sample):
        # print(sample)
        # print(self.augmenter)
        sample['image'] = self.augmenter(image=np.array(sample['image']))["image"]
        # plt.imshow(sample['image'])
        return sample


augmenter = albumentations.Compose([
    # albumentations.ShiftScaleRotate(rotate_limit=90, p=0.7),
    # albumentations.RandomBrightnessContrast(p=0.4),
    # albumentations.RandomGamma(p=0.4),
    # albumentations.Blur(blur_limit=2, p=0.1),
    # albumentations.GaussNoise((10, 100), p=0.2),
    albumentations.HorizontalFlip(p=0.5)
])

train_transforms = transforms.Compose([
    # apply_albumentations(augmenter),
    ScaleMaxSideToSize((CROP_SIZE, CROP_SIZE)),
    CropCenter(CROP_SIZE),
    TransformByKeys(transforms.ToPILImage(), ("image",)),
    TransformByKeys(transforms.ToTensor(), ("image",)),
    # TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image",)),#
    TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",))
])

val_transforms = transforms.Compose([

    ScaleMaxSideToSize((CROP_SIZE, CROP_SIZE)),
    CropCenter(CROP_SIZE),
    TransformByKeys(transforms.ToPILImage(), ("image",)),
    TransformByKeys(transforms.ToTensor(), ("image",)),
    # TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image",)),#
    TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",))
])
