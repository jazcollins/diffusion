''' 
'''

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import glob
import numpy as np
import os, json, cv2, random
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default=None, type=str, help="Path to input images")
parser.add_argument('--vis', action=argparse.BooleanOptionalAction, help="Write out detections")
args = parser.parse_args()

assert args.input_dir is not None

def cv2_imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# load detectron model
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)

# optional - visualize and save outputs
if args.vis:
    save_dir = args.input_dir + '_vis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
cls2id = {}
for i, cls in enumerate(class_names):
    cls2id[cls] = i

results_dict = {}
img_paths = glob.glob(os.path.join(args.input_dir, '*.png'))
overall_acc = 0.0
for img_path in tqdm.tqdm(img_paths): # loop through images (smarter - batchify this)
    # get GT class
    img_name = img_path.split('/')[-1]
    img_prompt = img_name.split('_')[0]

    # not ideal, do this better next time
    if 'A pair of ' in img_prompt:
        coco_class = img_prompt.replace('A pair of ', '') 
    elif 'A head of ' in img_prompt:
        coco_class = img_prompt.replace('A head of ', '') 
    elif 'An ' in img_prompt: 
        coco_class = img_prompt.replace('An ', '') 
    elif 'A ' in img_prompt:
        coco_class = img_prompt.replace('A ', '')  

    coco_id = cls2id[coco_class] # look up id to match with prediction

    # load image
    im = cv2_imread(img_path)
    
    # run maskrcnn
    outputs = predictor(im)

    pred_ids = outputs["instances"].pred_classes.cpu().numpy()
    is_cls_detected = np.any(pred_ids == coco_id)

    if is_cls_detected:
        score = 1.0
    else:
        score = 0.0

    results_dict[img_name] = score
    overall_acc += score

    if args.vis:
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_im = out.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(save_dir, img_name), cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB))

# save results dict
with open(os.path.join(args.input_dir, 'results_dict.json'), 'w+') as f:
    json.dump(results_dict, f)

print('overall acc is:', overall_acc/len(img_paths) * 100.0)