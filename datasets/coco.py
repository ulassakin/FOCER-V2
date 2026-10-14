# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

import cv2
import numpy as np
import json
import scipy.io
import math


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, matfiles, image_set, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        #new
        self.matfiles = matfiles
        self.image_set = image_set

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        #new
        if self.image_set == "val":
            mat = scipy.io.loadmat(self.matfiles[image_id])
            img = (mat["rgb_dem"]).astype(np.float64)
            img = np.clip(img,0,255)
        if self._transforms is not None:
            #new
            #24bit[0,1]
            """
            img = np.clip(img/255,0,1)
            for i in [1,2,8,32,256,1024,4096]:
                ldrimg = (gammma_correction((np.clip((img*i),0,1))*255)).astype(np.uint8)
                ldrimg, target = self._transforms(ldrimg, target)
                ldrimg = ldrimg.numpy()
                if i == 1:
                    imgs = ldrimg
                else:
                    np.concatenate([imgs,ldrimg],2)
                print(imgs.shape)
            """
        return imgs, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        #h, w, _ = image.shape

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set,mode):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            #normalize,
            LocalBrightnessNorm(),
            T.ToTensor(),
        ])

    if image_set == 'val' and mode == "log":
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            TonemapLog(),
            normalize,
            T.ConvertImageDtype(),
        ])
    elif image_set == 'val' and mode == "reinhard":
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            TonemapReinhard(epsr, fh, fl),
            normalize,
            T.ConvertImageDtype(),
        ])
    else :
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            #TonemapReinhard(),
            normalize,
            T.ConvertImageDtype(),
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    #new
    matfiles = []
    if image_set == "val":
        with open(ann_file,mode="rt",encoding="utf-8") as f:
            ann = json.load(f)
        images = ann["images"]
        matpath = args.mat_dir
        for i in range(len(images)):
            image_path = images[i]["file_name"]
            image_path = image_path.replace("JPEGImages/","")
            image_path = image_path.replace(".jpg","")
            image_path = matpath + image_path + "_dem.mat"
            matfiles.append(image_path)
    dataset = CocoDetection(img_folder, ann_file, matfiles, image_set, transforms=make_coco_transforms(image_set,args.modes), return_masks=args.masks)
    return dataset


class LocalBrightnessNorm:
    def __init__(self, sigma=20, eps=1e-5, ksize=0, mean_subtraction=True):
        self.ksize = ksize
        if ksize <= 0:
            self.ksize = int(sigma * 3) * 2 + 1
        self.sigma = sigma
        self.eps = eps
        self.mean_subtraction = mean_subtraction

    def __call__(self, img, target):
        img = np.asarray(img).astype(np.float64)
        img = np.clip(img,0,255)
        #img = LogTonemap(img)
        img = ReinhardTonemap(img)
        #img = DragoTonemap(img)
        #img = gamma(img)
        if self.mean_subtraction:
            m = cv2.GaussianBlur(img, (self.ksize, self.ksize), self.sigma)
            mean = m.copy()
            for j in range(3):
                mean[..., j] = (m[..., 0] + m[..., 1] + m[..., 2]) / 3.
            img = img - mean

        img2 = np.square(img)
        v = cv2.GaussianBlur(img2, (self.ksize, self.ksize), self.sigma)
        v_m = v.copy()
        for j in range(3):
            v_m[..., j] = (v[..., 0] + v[..., 1] + v[..., 2]) / 3.

        img = img / np.sqrt((v_m + self.eps))
        target = target

        return img, target

class TonemapLog:
    def __call__(self, img, target):
        img = np.asarray(img).astype(np.float64)
        img = np.clip(img,0,255)
        img = LogTonemap(img)
        target = target
        return img, target

class TonemapReinhard:
    def __init__(self, eps, fh, fl):
        self.eps = eps
        self.fh = fh
        self.fl = fl
    def __call__(self, img, target):
        img = np.asarray(img).astype(np.float64)
        img = np.clip(img,0,255)
        img = ReinhardTonemap(img, self.eps, self.fh, self.fl)
        target = target
        return img, target

def LogTonemap(image):
    if np.mean(image) > 1:
        log = 10000
    else :
        log = 100000
    image = (np.log(1 + log*(image/255))/np.log(1 + log))
    return image

def ReinhardTonemap2(image):
    if np.mean(image) > 1:
        intensity = 0
    else :
        intensity = 1
    tonemap = cv2.createTonemapReinhard(2,intensity,0,0)
    image = (image).astype(np.float32)
    image = tonemap.process(image/255)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    print(np.max(image))
    print(np.min(image))
    return image

def ReinhardTonemap(img, eps, fh, fl):
    if np.mean(img) > 1:
        f = fh
    else :
        f = fl
    img = img/255
    L = Luminance(img)
    #eps = 1e-5
    k = (np.max(L)-np.mean(L))/(np.max(L)-np.min(L))
    m = 0.3 + 0.7*(k**1.4)
    #m = 0.43
    f = math.exp(-f)
    for i in range(3):
        img[:,:,i] = img[:,:,i]/(img[:,:,i] + (f*L)**m + eps)
    img = np.clip(img,0,1)
    return img

def DragoTonemap(image):
    tonemap = cv2.createTonemapDrago()
    image = image.astype(np.float32)
    image = tonemap.process(image/255)
    image = np.clip(image,0,1)
    return image

def gamma(image):
    return 255*((image/255)**(1/2.2))

def Luminance(img):
    return 0.2125*img[:,:,2] + 0.7154*img[:,:,1] + 0.0721*img[:,:,0]

def gammma_correction(img):
    gamma = 2.2
    return 255*((img/255)**(1/gamma))