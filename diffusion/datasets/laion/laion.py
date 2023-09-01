# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming LAION dataset."""

from io import BytesIO
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusion.datasets.laion.transforms import LargestCenterSquare, RandomCropSquareReturnTransform, LargestCenterSquareReturnTransform

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class StreamingLAIONDataset(StreamingDataset):
    """Implementation of the LAION dataset as a streaming dataset.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from. StreamingLAIONDataset
            uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        split (str, optional): The dataset split to use. Currently, only ``None`` is supported. Default: ``None``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``False``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        tokenizer_name_or_path_2 (Optional[str]): The name or path of the second tokenizer (for SDXL). Default: ``None``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
        image_size (Optional[int]): The size to resize the image to. Default: ``None``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: bool = False,
        tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
        tokenizer_name_or_path_2: Optional[str] = None,
        caption_drop_prob: float = 0.0,
        transform: Optional[Callable] = None,
        predownload: int = 100_000,
        download_retry: int = 2,
        download_timeout: float = 120,
        batch_size: Optional[int] = None,
        image_size: Optional[int] = None,
        num_canonical_nodes: Optional[int] = None,
        sdxl: Optional[bool] = False,
        cond_drop_prob: float = 0.0,
        zero_dropped_captions: bool = False,
        rand_crop: bool = True,
    ) -> None:

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            shuffle=shuffle,
            predownload=predownload,
            keep_zip=False,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=None,
            batch_size=batch_size,
            num_canonical_nodes=num_canonical_nodes,
        )

        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path, subfolder='tokenizer')
        if sdxl:
            if tokenizer_name_or_path_2:
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_name_or_path_2, subfolder='tokenizer_2')
            else:
                raise ValueError('Must provide value for tokenizer_name_or_path_2')
            
        self.caption_drop_prob = caption_drop_prob
        self.image_size = image_size
        self.sdxl = sdxl
        if sdxl:
            if rand_crop:
                self.sdxl_transform = RandomCropSquareReturnTransform(self.image_size)
            else:
                self.sdxl_transform = LargestCenterSquareReturnTransform(self.image_size)
        self.cond_drop_prob = cond_drop_prob # sdxl
        self.zero_dropped_captions = zero_dropped_captions

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        img = Image.open(BytesIO(sample['jpg']))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        crop_top, crop_left, image_height, image_width = None, None, None, None
        if self.sdxl:
            # sdxl crop to return params
            img, crop_top, crop_left, image_height, image_width = self.sdxl_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        # Drop the caption with probability `caption_drop_prob`
        if torch.rand(1) < self.caption_drop_prob:
            caption = ''
            tokenized_caption = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )['input_ids']
            tokenized_caption = torch.tensor(tokenized_caption)
            if self.zero_dropped_captions:
                tokenized_caption = torch.zeros_like(tokenized_caption)
        else:
            caption = sample['caption']
            tokenized_caption = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )['input_ids'] 
            tokenized_caption = torch.tensor(tokenized_caption)

        out = {'image': img, 'captions': tokenized_caption}

        # optional SDXL tokenizer_2
        if self.sdxl:
            tokenized_caption_2 = self.tokenizer_2(
                caption,
                padding='max_length',
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
            )['input_ids']
            tokenized_caption_2 = torch.tensor(tokenized_caption_2)
            if self.zero_dropped_captions:
                tokenized_caption_2 = torch.zeros_like(tokenized_caption_2)
            out['captions_2'] = tokenized_caption_2

        if 'caption_latents' in sample:
            out['caption_latents'] = torch.from_numpy(
                np.frombuffer(sample['caption_latents'], dtype=np.float16).copy()).reshape(77, 1024)
        if self.image_size == 256 and 'latents_256' in sample:
            out['image_latents'] = torch.from_numpy(np.frombuffer(sample['latents_256'],
                                                                  dtype=np.float16).copy()).reshape(4, 32, 32)
        if self.image_size == 512 and 'latents_512' in sample:
            out['image_latents'] = torch.from_numpy(np.frombuffer(sample['latents_512'],
                                                                  dtype=np.float16).copy()).reshape(4, 64, 64)

        if self.sdxl:  # add crop and img size params
            if torch.rand(1) < self.cond_drop_prob: # TODO i think stability does separate zero-ing for each element? doing this for now
                out['cond_crops_coords_top_left'] = torch.tensor([crop_top, crop_left]) * 0
                out['cond_original_size'] = torch.tensor([image_width, image_height]) * 0
                out['cond_target_size'] = torch.tensor([self.image_size, self.image_size]) * 0
            else:
                out['cond_crops_coords_top_left'] = torch.tensor([crop_top, crop_left])
                out['cond_original_size'] = torch.tensor([image_width, image_height])
                out['cond_target_size'] = torch.tensor([self.image_size, self.image_size])
        return out


def build_streaming_laion_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    tokenizer_name_or_path_2: Optional[str] = None,
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    num_samples: Optional[int] = None,
    predownload: int = 100_000,
    download_retry: int = 2,
    download_timeout: float = 120,
    drop_last: bool = True,
    shuffle: bool = True,
    num_canonical_nodes: Optional[int] = None,
    sdxl: bool = False,
    cond_drop_prob: float = 0.0,
    zero_dropped_captions: bool = False,
    rand_crop: bool = True
    **dataloader_kwargs,
):
    """Builds a streaming LAION dataloader.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        tokenizer_name_or_path_2 (Optional[str]): The name or path of the second tokenizer to use (for SDXL). Default: ``None``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        resize_size (int): The size to resize the image to. Default: ``256``.
        num_samples (Optional[int]): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
        sdxl (bool): Whether training SDXL or not. If True, return img size and crop params for conditioning. Default: ``False``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    if isinstance(remote, str) and isinstance(local, str):
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [remote], [local]
    elif isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        ValueError(f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l, download_retry=download_retry, download_timeout=download_timeout))

    # Normalize from 0 to 1 to -1 to 1
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if sdxl:
        # do center square crop separately
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        center_square_crop = LargestCenterSquare(resize_size)
        transform = transforms.Compose([center_square_crop, transforms.ToTensor(), normalize])

    dataset = StreamingLAIONDataset(
        streams=streams,
        split=None,
        shuffle=shuffle,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_name_or_path_2=tokenizer_name_or_path_2,
        caption_drop_prob=caption_drop_prob,
        transform=transform,
        predownload=predownload,
        download_retry=download_retry,
        download_timeout=download_timeout,
        batch_size=batch_size,
        image_size=resize_size,
        num_canonical_nodes=num_canonical_nodes,
        sdxl=sdxl,
        cond_drop_prob=cond_drop_prob,
        zero_dropped_captions=zero_dropped_captions,
        rand_crop=rand_crop,
    )
    # Create a subset of the dataset
    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_samples))  # type: ignore

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
