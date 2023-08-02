''' 
    Modified inference script from Cory for generating images of ImageNet classes. 
    imagenet_class_index.json derived from 
    https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
'''

import os
import torch
from composer import Trainer
from diffusion.models.models import stable_diffusion_2
from torchvision.transforms.functional import to_pil_image
import argparse
from PIL import Image
import json

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', 
                    default='/mnt/workdisk/jasmine/oci-checkpoints/aaron/stable-diffusion-256-blip-yfcc100m-H100/ep11-ba550000-rank0.pt', 
                    type=str, 
                    help="Path to load checkpoint from")
parser.add_argument('--output_path', default='outputs/imagenet/stable-diffusion-256-blip-yfcc100m-H100', type=str, help="Path to save outputs")
parser.add_argument('--num_images', default=1, type=int, help="How many images to generate per prompt?")
args = parser.parse_args()

assert args.load_path is not None

# Check if the output directory exists
if not os.path.exists(args.output_path):
    # If it doesn't exist, create the directory
    os.makedirs(args.output_path)

# Load the model
if args.load_path is not None:
    print('loading model')
    model = stable_diffusion_2(model_name='stabilityai/stable-diffusion-2-base',val_metrics=[], pretrained=False, encode_latents_in_fp16=False, fsdp=False)
    trainer = Trainer(model=model,
                    load_path=args.load_path,
                    load_weights_only=True)
    print('model loaded')

def generate(prompt, save_name, negative_prompt=None, guidance_scale=5.0, height=256, width=256, num_inference_steps=50, seed=1138, num_images=1):
    prompts = [prompt] * num_images

    tokenized_prompts = []
    for prompt in prompts:
        tokenized_caption = model.tokenizer(prompt, padding='max_length', max_length=model.tokenizer.model_max_length, truncation=True)['input_ids']
        tokenized_caption = torch.tensor(tokenized_caption)
        tokenized_prompts.append(tokenized_caption)
    prompt_tensor = torch.stack(tokenized_prompts).cuda()

    if negative_prompt is not None:
        negative_prompts = [negative_prompt] * num_images

    imgs = model.generate(tokenized_prompts=prompt_tensor,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=1,
                        seed=seed,)

    # Images are a torch tensor of shape (num_images, 3, height, width)
    # Convert to PIL images and save to disk
    for i, img in enumerate(imgs):
        to_pil_image(img).save(f"{args.output_path}/{save_name}_{prompts[i][0:50]}_{seed}_{i}.png")
    return imgs

# Generate prompts
imagenet_json = '/mnt/workdisk/jasmine/diffusion/class_mappings/imagenet_class_index.json'
# Opening JSON file
with open(imagenet_json) as f:
    imagenet_classes = json.load(f)

for cls_idx in imagenet_classes:
    imagenet_class, imagenet_class_str = imagenet_classes[cls_idx]
    imagenet_class_str = imagenet_class_str.replace('_', ' ') # replace underscores

    if imagenet_class_str[0].lower() in ['a', 'e', 'i', 'o', 'u']: # first letter is vowel --> 'An'
        prompt = 'An ' + imagenet_class_str
    else: # --> 'A'
        prompt = 'A ' + imagenet_class_str

    save_name = cls_idx + '_' + imagenet_class_str + '_' + imagenet_class

    generate(prompt, save_name, negative_prompt="", seed=12345, num_images=args.num_images)