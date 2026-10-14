CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

import torch
from PIL import Image
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os
import torchvision.transforms.functional as F
import scipy.io
import random
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def gammma_correction(img):
    gamma = 2.2
    return ((img)**(1/gamma))
def LogTonemap(image):
    if np.mean(image*255) > 1:
        log = 10000
    else :
        log = 100000
    image = (np.log(1 + log*(image))/np.log(1 + log))
    return image
def ReinhardTonemap(image):
    if np.mean(image*255) > 1:
        intensity = 0
    else :
        intensity = 1
    tonemap = cv2.createTonemapReinhard(2,intensity,0,0)
    image = (image).astype(np.float32)
    #image = np.stack([tonemap.process(image[0]),tonemap.process(image[1]),tonemap.process(image[2])])
    image = tonemap.process(image)
    image = image.astype(np.float64)
    return image
def log_paramfree(img,param):
    return (np.log(1+param*img))/np.log(1+param)
def reinhard_paramfree(img,param):
    tonemap = cv2.createTonemapReinhard(2,param,0,0)
    return (tonemap.process(img.astype(np.float32))).astype(np.float64)

class TonemapLog:
    def __init__(self,logparam):
        self.logparam = logparam
    def __call__(self, img):
        img = np.asarray(img).astype(np.float64)
        img = np.clip(img,0,1)
        img = log_paramfree(img,self.logparam)
        return img

class TonemapReinhard:
    def __init__(self,reinhardparam):
        self.reinhardparam = reinhardparam
    def __call__(self, img):
        img = np.asarray(img).astype(np.float64)
        img = np.clip(img,0,1)
        img = reinhard_paramfree(img,self.reinhardparam)
        return img

class LocalBrightnessNorm:
    def __init__(self, sigma, eps, ksize=0, mean_subtraction=True):
        self.ksize = ksize
        if ksize <= 0:
            self.ksize = int(sigma * 3) * 2 + 1
        self.sigma = sigma
        self.eps = eps
        self.mean_subtraction = mean_subtraction

    def __call__(self, img):
        img = np.asarray(img*255).astype(np.float64)
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

        return img

class ConvertImageDtype:
    def __call__(self, img):
        return F.convert_image_dtype(img, torch.float)

class RandomResizeNP:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = random.choice(self.sizes)
        return resizenp(img, size, self.max_size)

def resizenp(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        #w, h = image_size
        h, w, _ = image.shape
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    h_orig, w_orig = size
    #rescaled_image = F.resize(image, size)
    rescaled_image = cv2.resize(image, (w_orig,h_orig))
    h_res,w_res,_ = rescaled_image.shape

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip((w_res,h_res), (w_orig,h_orig)))
    ratio_width, ratio_height = ratios

    return rescaled_image

class RandomResizePIL:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = random.choice(self.sizes)
        return resizepil(img, size, self.max_size)

def resizepil(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        #h, w, _ = image.shape
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    return rescaled_image

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size,device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h],device = device,dtype=torch.float32)
    return b

def make_baselist(base,max_base):
    b = base
    base_list = [1]
    while b <= max_base:
        base_list.append(b)
        b = b*base
    return base_list

def box_cxcywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def detrprocess(outputs,w,h,device):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    prob = out_logits.softmax(-1)
    scores, labels = prob[..., :-1].max(-1)
    # convert to [x0, y0, x1, y1] format
    boxes = box_cxcywh2xyxy(out_bbox)
    scale_fct = torch.tensor([w, h, w, h])
    scale_fct = scale_fct.to(device)
    boxes = torch.mul(boxes[:],scale_fct)
    for i in range(scores.size()[0]):
        if i == 0:
            out_bboxes = boxes[0]
            out_scores = scores[0]
            out_labels = labels[0]
        else:
            out_bboxes = torch.cat([out_bboxes, boxes[i]],dim=0)
            out_scores = torch.cat([out_scores, scores[i]],dim=0)
            out_labels = torch.cat([out_labels, labels[i]],dim=0)
    #results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    results = {"scores":out_scores.tolist(),"bboxes":out_bboxes.tolist(),"labels":out_labels.tolist()}
    return results

def late_fusion(img,image_id,base_list,transform,threshold,model,args,device):
    threshold_nms = args.threshold_nms
    inputs = []
    im_results = []
    h,w,_ = img.shape
    size = w,h
    if args.fusion_mode == "normal":
        for i in base_list:
            #ldrimg = ((np.clip((img*i),0,1))*255).astype(np.uint8)
            ldrimg = gammma_correction((np.clip((img*i),0,1)))
            input = transform(ldrimg)
            input = input.to(device)
            inputs.append(input)
    elif args.fusion_mode == "log":
        log_params = make_baselist(args.logbase,args.max_logbase)
        for log_param in log_params:
            ldrimg = log_paramfree(img,log_param)
            input = (transform(ldrimg)).to(device)
            inputs.append(input)
    elif args.fusion_mode == "reinhard":
        reinhard_params = [-3,-2,-1,0,1,2,3]
        for reinhard_param in reinhard_params:
            ldrimg = reinhard_paramfree(img,reinhard_param)
            input = (transform(ldrimg)).to(device)
            inputs.append(input)
    elif args.fusion_mode == "mix":
        for i in base_list:
            #ldrimg = ((np.clip((img*i),0,1))*255).astype(np.uint8)
            ldrimg = LogTonemap((np.clip((img*i),0,1)))
            input = transform(ldrimg)
            input = input.to(device)
            inputs.append(input)
    outputs = model(inputs)
    out_results = detrprocess(outputs,w,h,device)
    bbox_num = len(out_results["labels"])
    results_class_boxes = {}
    results_class_scores = {}
    for k in range(bbox_num):
        out_label = out_results["labels"][k]
        out_box = out_results["bboxes"][k]
        out_score = out_results["scores"][k]
        if out_label not in results_class_boxes:
            results_class_boxes[out_label] = []
            results_class_scores[out_label] = []
        results_class_boxes[out_label].append(out_box)
        results_class_scores[out_label].append(out_score)
    for c in results_class_boxes:
        bboxes = torch.Tensor(np.array(results_class_boxes[c]))
        probs = torch.Tensor(np.array(results_class_scores[c]))
        nms_result = (torchvision.ops.nms(bboxes,probs,threshold_nms)).tolist()
        if len(nms_result) != 0:
            for b in nms_result:
                box = bboxes[b].tolist()
                box[2] = box[2]-box[0]
                box[3] = box[3]-box[1]
                score = probs[b].item()
                label = c
                result = {
                    "image_id":image_id,
                    "category_id":label,
                    "bbox":box,
                    "score":score,
                }
                im_results.append(result)
    return im_results

def tonemapping(img,image_id,model,args,device):
    if args.mode == "log":
        transformtonemap = transforms.Compose(
            [
            RandomResizeNP([800], max_size=1333),
            TonemapLog(args.logparam),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ConvertImageDtype()
            ])
    elif args.mode == "reinhard":
        transformtonemap = transforms.Compose(
            [
            RandomResizeNP([800], max_size=1333),
            TonemapReinhard(args.reinhardparam),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ConvertImageDtype()
            ])
    elif args.mode == "lbn":
        transformtonemap = transforms.Compose(
            [
            RandomResizeNP([800], max_size=1333),
            LocalBrightnessNorm(args.sigma,args.eps),
            transforms.ToTensor(),
            ConvertImageDtype()
            ])
    im_results = []
    h,w,_ = img.shape
    inputs = transformtonemap(img)
    inputs = inputs.to(device)
    inputs = [inputs]
    outputs = model(inputs)
    out_results = detrprocess(outputs,w,h,device)
    bbox_num = len(out_results["labels"])
    for i in range(bbox_num):
        out_box = out_results["bboxes"][i]
        out_label = out_results["labels"][i]
        out_score = out_results["scores"][i]
        out_box[2]=out_box[2]-out_box[0]
        out_box[3]=out_box[3]-out_box[1]
        result = {
            "image_id":image_id,
            "category_id":out_label,
            "bbox":out_box,
            "score":out_score,
        }
        im_results.append(result)
    return im_results

def intermediatefusion(img,image_id,base_list,model,args,device):
    transform = transforms.Compose(
            [
            RandomResizeNP([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ConvertImageDtype()
            ])
    inputs = []
    if args.fusion_mode == "normal":
        for b in base_list:
            ldrimg = (gammma_correction((np.clip((img*b),0,1)))).astype(np.float32)
            ldrimg = cv2.cvtColor(ldrimg,cv2.COLOR_BGR2RGB)
            input = transform(ldrimg)
            input = input.to(device)
            inputs.append(input)
    elif args.fusion_mode == "log":
        log_params = make_baselist(args.logbase,args.max_logbase)
        for log_param in log_params:
            ldrimg = log_paramfree(img,log_param)
            input = (transform(ldrimg)).to(device)
            inputs.append(input)
    h,w,_ = img.shape
    size = w,h
    im_results = []
    outputs = model(inputs)
    out_results = detrprocess(outputs,w,h,device)
    bbox_num = len(out_results["labels"])
    for i in range(bbox_num):
        out_box = out_results["bboxes"][i]
        out_label = out_results["labels"][i]
        out_score = out_results["scores"][i]
        out_box[2]=out_box[2]-out_box[0]
        out_box[3]=out_box[3]-out_box[1]
        result = {
            "image_id":image_id,
            "category_id":out_label,
            "bbox":out_box,
            "score":out_score,
        }
        im_results.append(result)
    return im_results

def result_eval(model,args,device):
    all_results = []
    threshold = 0
    transform = transforms.Compose(
            [
            RandomResizeNP([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ConvertImageDtype()
            ])
    model.eval()
    matdir = args.matdir
    ann_dir = args.coco_path + "/annotations/instances_val2017.json"
    base_list = make_baselist(args.base,args.max_base)
    os.makedirs(args.det_dir, exist_ok=True)
    detection_dir = args.det_dir + "{}/".format(args.mode)
    os.makedirs(detection_dir, exist_ok=True)
    with open(ann_dir,mode="rt",encoding="utf-8") as f:
        ann = json.load(f)
    images = ann["images"]
    if args.mode == "multi":
        if args.fusion_mode == "normal":
            resfile = detection_dir + "base{}_{}.json".format(str(args.base),str(args.max_base))
        elif args.fusion_mode == "log":
            resfile = detection_dir + "logbase{}_{}.json".format(str(args.logbase),str(args.max_logbase))
        elif args.fusion_mode == "reinhard":
            resfile = detection_dir + "{}.json".format(args.fusion_mode)
        elif args.intermediate:
            resfile = detection_dir + "intermediate.json"
    else:
        if args.mode == "log":
            resfile = detection_dir + "{}.json".format(str(args.logparam))
        elif args.mode == "reinhard":
            resfile = detection_dir + "{}.json".format(str(args.reinhardparam))
        elif args.mode == "lbn":
            resfile = detection_dir + "{}.json".format(args.mode)

    if not args.nomodel:
        for j in range(len(images)):
            image_id = images[j]["id"]
            name = (images[j]["file_name"].replace("JPEGImages/","")).replace(".jpg","")
            if args.dataset_mode == "Ours":
                matname = matdir + name + ".mat"
            elif args.dataset_mode == "rod":
                matname = matdir + name + ".mat"
            mat = scipy.io.loadmat(matname)
            img = mat["rgb_dem"]
            img = (np.clip(img/255,0,1)).astype(np.float64)
            tmp_maxpixel = np.amax(img)
            print(tmp_maxpixel)
            if args.intermediate:
                results = intermediatefusion(img,image_id,base_list,model,args,device)
            elif args.mode == "multi":
                results = late_fusion(img,image_id,base_list,transform,threshold,model,args,device)
            elif args.mode == "log" or "reinhard" or "lbn":
                results = tonemapping(img,image_id,model,args,device)
            all_results.extend(results)
        ann["annotations"] = all_results
        if not args.nosave:
            with open(resfile,"w") as f:
                json.dump(ann,f)
    else:
        with open(resfile,mode="rt",encoding="utf-8") as f:
            res = json.load(f)
        all_results = res["annotations"]

    #catIdlist = [1,2,3,8]
    if args.dataset_mode == "Ours":
        catIdlist = [1,2,3,8]
    else:
        catIdlist = [3,8]
    #1:Person
    #2:Bicycle
    #3:Car
    #4:Motorcycle
    #6:Bus
    #8:Truck
    #10:Traffic light
    if args.mode == "multi":
        if args.intermediate:
            mode = "intermediate" + "base{}_{}".format(str(args.base),str(args.max_base))
        elif args.fusion_mode == "normal":
            mode = args.mode + "base{}_{}".format(str(args.base),str(args.max_base))
        elif args.fusion_mode == "log" :
            mode = args.mode + args.fusion_mode + "logbase{}_{}".format(str(args.logbase),str(args.max_logbase))
        elif args.fusion_mode == "reinhard":
            mode = args.mode + args.fusion_mode
        
    else:
        if args.mode == "log":
            mode = args.mode + str(args.logparam)
        elif args.mode == "reinhard":
            mode = args.mode + str(args.reinhardparam)
        elif args.mode == "lbn":
            mode = args.mode
    print(args.dataset_mode)
    cocoeval(ann_dir,all_results,catIdlist,mode)
    return

def cocoeval(ann_dir,all_results,catIdlist,mode):
    coco_gt = COCO(ann_dir)
    coco_results = coco_gt.loadRes(all_results)
    mAP = 0
    APs = {}
    for catId in catIdlist:
        coco_eval = COCOeval(coco_gt, coco_results, "bbox")
        coco_eval.params.catIds = [catId]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        AP = coco_eval.stats[0]*100
        mAP += AP/len(catIdlist)
        APs[catId] = AP
    for cl in APs:
        print("AP({}):{}".format(CLASSES[cl],APs[cl]))
    print("mAP({}):{}".format(mode,mAP))