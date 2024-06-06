# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Image-Caption Dataset for SDXL with Pre-computed Text Latents."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion.datasets.laion.transforms import LargestCenterSquare, RandomCropAspectRatioTransorm, RandomCropSquare

log = logging.getLogger(__name__)

class StreamingTextLatentsDataset(StreamingDataset):

    def __init__(
            self,
            streams: Sequence[Stream],
            caption_drop_prob: float = 0.0,
            microcond_drop_prob: float = 0.0,
            crop: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            image_key: str = 'image',
            text_latent_keys: Tuple[str, ...] = ('T5_LATENTS', 'CLIP_LATENTS'),
            text_latent_shapes: Tuple[Tuple, ...] = ((512, 4096), (77, 768)),
            attention_mask_keys: Tuple[str, ...] = ('T5_ATTENTION_MASK', 'CLIP_ATTENTION_MASK'),
            **streaming_kwargs,
    ):
        
        # Set defaults for vision-friendly streaming args.
        streaming_kwargs.setdefault('shuffle_block_size', 1 << 18)
        streaming_kwargs.setdefault('shuffle_algo', 'py1s')
        super().__init__(streams=streams, **streaming_kwargs)

        self.crop = crop
        self.transform = transform
        self.caption_drop_prob = caption_drop_prob
        self.microcond_drop_prob = microcond_drop_prob
        self.image_key = image_key
        self.text_latent_keys = text_latent_keys
        self.text_latent_shapes = text_latent_shapes
        self.attention_mask_keys = attention_mask_keys

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        out = {}

        # Image
        img = sample[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.open(BytesIO(sample[self.image_key]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        out['cond_original_size'] = torch.tensor(img.size)

        # Image transforms
        if self.crop is not None:
            img, crop_top, crop_left = self.crop(img)
        else:
            crop_top, crop_left = 0, 0
        out['cond_crops_coords_top_left'] = torch.tensor([crop_top, crop_left])

        if self.transform is not None:
            img = self.transform(img)
        out['image'] = img

        # Get the new height and width
        if isinstance(img, torch.Tensor):
            img_h, img_w = img.shape[-2], img.shape[-1]
        elif isinstance(img, Image.Image):
            img_w, img_h = img.size
        else:
            raise ValueError('Image after transformations must either be a PIL Image or Torch Tensor')
        out['cond_target_size'] = torch.tensor([img_w, img_h])

        # Microconditioning dropout as in Stability repo
        # https://github.com/Stability-AI/generative-models/blob/477d8b9a7730d9b2e92b326a770c0420d00308c9/sgm/modules/encoders/modules.py#L151-L160
        if torch.rand(1) < self.microcond_drop_prob:
            out['cond_crops_coords_top_left'] = out['cond_crops_coords_top_left'] * 0
        if torch.rand(1) < self.microcond_drop_prob:
            out['cond_original_size'] = out['cond_original_size'] * 0
        if torch.rand(1) < self.microcond_drop_prob:
            out['cond_target_size'] = out['cond_target_size'] * 0

        # Load text latents, attention masks, and clip pooled embeddings
        for i in range(len(self.text_latent_keys)): 
            latent_key = self.text_latent_keys[i]
            latent_shape = self.text_latent_shapes[i]
            attention_key = self.attention_mask_keys[i]

            if torch.rand(1) < self.caption_drop_prob:
                out[latent_key] = torch.zeros(latent_shape, dtype=torch.float16)
                out[attention_key] = torch.zeros(latent_shape[0])
                if latent_key == 'CLIP_LATENTS':
                    out['CLIP_POOLED'] = torch.zeros(latent_shape[0])
            else:
                text_latent = np.frombuffer(sample[latent_key], dtype=np.float16).copy()
                out[latent_key] = torch.from_numpy(text_latent).reshape(latent_shape)
                print(i, len(sample[attention_key]))
                attention_mask = np.frombuffer(sample[attention_key], dtype=np.bool_).copy()
                out[attention_key] = torch.from_numpy(attention_mask).reshape(latent_shape[0])
                if latent_key == 'CLIP_LATENTS':
                    clip_pooled = np.frombuffer(sample['CLIP_POOLED_TEXT'], dtype=np.float16).copy()
                    out['CLIP_POOLED'] = torch.from_numpy(clip_pooled).reshape(latent_shape[1])
        return out
    

def build_streaming_text_latents_dataloader(
        remote: Union[str, List],
        batch_size: int,
        local: Optional[Union[str, List]] = None,
        caption_drop_prob: float = 0.0,
        microcond_drop_prob: float = 0.0,
        resize_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]] = 256,
        ar_bucket_boundaries: Optional[Tuple[float, ...]] = None,
        transform: Optional[List[Callable]] = None,
        crop_type: Optional[str] = 'square',
        image_key: str = 'image',
        text_latent_keys: Tuple[str, ...] = ('T5_LATENTS', 'CLIP_LATENTS'),
        text_latent_shapes: Tuple[Tuple, ...] = ((512, 4096), (77, 768)),
        attention_mask_keys: Tuple[str, ...] = ('T5_ATTENTION_MASK', 'CLIP_ATTENTION_MASK'),
        streaming_kwargs: Optional[Dict] = None,
        dataloader_kwargs: Optional[Dict] = None,
):
    
    # Check crop type
    if crop_type is not None:
        crop_type = crop_type.lower()
        if crop_type not in ['square', 'random', 'aspect_ratio']:
            raise ValueError(f'Invalid crop_type: {crop_type}. Must be ["square", "random", "aspect_ratio", None]')
        if crop_type == 'aspect_ratio' and (isinstance(resize_size, int) or isinstance(resize_size[0], int)):
            raise ValueError(
                'If using crop_type="aspect_ratio", specify aspect ratio buckets in resize_size as a tuple of tuples.')
        
    # Handle ``None`` kwargs
    if streaming_kwargs is None:
        streaming_kwargs = {}
    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    
        # Check types for remote and local

    if isinstance(remote, str):
        remote = [remote]
    if isinstance(local, str):
        local = [local]
    if not local:
        local = [_make_default_local_path(r) for r in remote]
    if isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        ValueError(f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l))

    # Set the crop to apply
    if crop_type == 'square':
        crop = LargestCenterSquare(resize_size)
    elif crop_type == 'random':
        crop = RandomCropSquare(resize_size)
    elif crop_type == 'aspect_ratio':
        crop = RandomCropAspectRatioTransorm(resize_size, ar_bucket_boundaries)  # type: ignore
    else:
        crop = None

    if transform is None:
        transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform)
    assert isinstance(transform, Callable)

    dataset = StreamingTextLatentsDataset(
        streams=streams,
        caption_drop_prob=caption_drop_prob,
        microcond_drop_prob=microcond_drop_prob,
        crop=crop,
        transform=transform,
        image_key=image_key,
        text_latent_keys=text_latent_keys,
        text_latent_shapes=text_latent_shapes,
        attention_mask_keys=attention_mask_keys,
        **streaming_kwargs,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        **dataloader_kwargs,
    )

    return dataloader

def _make_default_local_path(remote_path):
    return str(Path(*['/tmp'] + list(Path(remote_path).parts[1:])))