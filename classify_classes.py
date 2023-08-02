''' 
    Classify ImageNet classes in generated images.
'''

# import some common libraries
import glob
import numpy as np
import os, json, cv2, random
import argparse
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
from torchvision import transforms

import torch
# from torchvision.io import read_image
# from torchvision.models import resnet50, ResNet50_Weights

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default=None, type=str, help="Path to input images")
args = parser.parse_args()

assert args.input_dir is not None

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def visualize_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

def transform_image(image, resolution):
    transform = transforms.Compose([
        transforms.Resize(
            resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    image = transform(image)
    # we also add a batch dimension to the image since that is what the model expects
    image = image[None, :]
    return image

device = torch.device("cuda")
# load model
model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
# we also convert the model to eval mode
model.eval()
model = model.to(device)
resolution = 518

with open("class_mappings/imagenet_class_index.json", "r") as f:
    imagenet_id_to_name = {int(cls_id): name for cls_id, (label, name) in json.load(f).items()}


# TODO - batchify! altho to run thru 1k images takes only a few minutes

results_dict = {}
img_paths = glob.glob(os.path.join(args.input_dir, '*.png'))
overall_acc = 0.0
for img_path in tqdm.tqdm(img_paths): # loop through images (smarter - batchify this)

    # get GT class
    img_name = img_path.split('/')[-1]
    gt_class = int(img_name.split('_')[0])

    image = load_image(img_path)
    image = transform_image(image, resolution)
    image = image.to(device)
    
    # we do not need to track gradients for inference
    with torch.no_grad():
        _, preds = model(image).topk(5)
    
    # convert preds to a Python list and remove the batch dimension
    preds = preds.tolist()[0]
    # print([imagenet_id_to_name[cls_id] for cls_id in preds])
    top1_pred = preds[0]

    if top1_pred == gt_class:
        score = 1.0
    else:
        score = 0.0

    results_dict[img_name] = {'score': score, 'pred_class': imagenet_id_to_name[top1_pred], 'gt_class': img_name.split('_')[1]}
    overall_acc += score

# save results dict
with open(os.path.join(args.input_dir, 'results_dict.json'), 'w+') as f:
    json.dump(results_dict, f)

print('------------')
print('overall acc is:', overall_acc/len(img_paths) * 100.0)
print('------------')