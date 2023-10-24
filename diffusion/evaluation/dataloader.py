# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Image-Caption dataset."""

import logging
import random
from io import BytesIO
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from diffusion.datasets.laion.transforms import LargestCenterSquare, RandomCropAspectRatioTransorm, RandomCropSquare
from diffusion.models.models import SDXLTokenizer

import urllib.request
import json
from composer.utils import dist

log = logging.getLogger(__name__)

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class ImageNetClassesDataset(Dataset):
    """Streaming dataset for prompt-GT class pairs.

    Args:
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        sdxl (bool): Whether or not we're training SDXL. Default: `False`.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
        sdxl: bool = False,
    ) -> None:

        path_to_imagenet_classes = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with urllib.request.urlopen(path_to_imagenet_classes) as url:
            imagenet_classes = json.load(url)
        # print(imagenet_classes)

        self.imagenet_prompts = [] # human readable
        self.imagenet_classes = [] # numerical class name
        self.imagenet_class_indexes = []
        self.class_strings = []
        for class_idx in imagenet_classes:
            imagenet_class, imagenet_class_str = imagenet_classes[class_idx]
            self.class_strings.append(imagenet_class_str)
            imagenet_class_str = imagenet_class_str.replace('_', ' ') # replace underscores

            if imagenet_class_str[0].lower() in ['a', 'e', 'i', 'o', 'u']: # first letter is vowel --> 'An'
                prompt = 'An ' + imagenet_class_str
            else: # --> 'A'
                prompt = 'A ' + imagenet_class_str

            self.imagenet_prompts.append(prompt)
            self.imagenet_classes.append(imagenet_class)
            self.imagenet_class_indexes.append(class_idx)

        # TODO we could probably also tokenize here...
        self.sdxl = sdxl
        if self.sdxl:
            self.tokenizer = SDXLTokenizer(tokenizer_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, subfolder='tokenizer')

    def get_class_strings(self):
        return self.class_strings

    def __len__(self):
        return len(self.imagenet_prompts)

    def __getitem__(self, idx):
        # TODO should we just have one prompt per class? handle making multiple outside of this?
        # saves time to just tokenize once..
        prompt = self.imagenet_prompts[idx]
        class_idx = self.imagenet_class_indexes[idx]
        imagenet_class = self.imagenet_classes[idx]


        max_length = None if self.sdxl else self.tokenizer.model_max_length  # type: ignore
        tokenized_caption = self.tokenizer(prompt,
                                           padding='max_length',
                                           max_length=max_length,
                                           truncation=True,
                                           return_tensors='pt')['input_ids']
        if self.sdxl:
            tokenized_caption = [tokenized_cap.squeeze() for tokenized_cap in tokenized_caption]
            tokenized_caption = torch.stack(tokenized_caption)
        else:
            tokenized_caption = tokenized_caption.squeeze()

        return {'tokenized_prompt': tokenized_caption, 'prompt': prompt, 
                'class_idx': int(class_idx), 'imagenet_class': imagenet_class}


def build_imagenet_dataloader(
    batch_size: int,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a dataloader.

    Args:
        batch_size (int): The batch size to use for both the ``StreamingDataset`` and ``DataLoader``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        dataloader_kwargs (dict, optional): Additional arguments to pass to the ``DataLoader``. Default: ``None``.
    """

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    # Infer SDXL from tokenizer path
    sdxl = (tokenizer_name_or_path == 'stabilityai/stable-diffusion-xl-base-1.0')
    if sdxl:
        log.info('Detected SDXL tokenizer, using SDXL crop transform and tokenizers.')

    dataset = ImageNetClassesDataset(
        tokenizer_name_or_path=tokenizer_name_or_path,
        sdxl=sdxl,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(dataset),
        **dataloader_kwargs,
    )

    return dataloader
