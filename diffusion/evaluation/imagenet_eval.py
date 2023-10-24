# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation using ImageNet eval."""

import json
import os
from typing import List, Optional, Literal, Any, Union

import clip
import torch
import wandb
from cleanfid import fid
from composer import ComposerModel, Trainer
from composer.core import get_precision_context
from composer.loggers import LoggerDestination, WandBLogger
from composer.utils import dist
from torch.utils.data import DataLoader
from torchmetrics.multimodal import CLIPScore
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from torchmetrics import Metric

import glob
import numpy as np
import os, json, cv2, random
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pandas as pd


def get_classification_model(model_path):
    if model_path == 'vit_h14_in1k':
        resolution = 518
        transform = transforms.Compose([
            transforms.ToPILImage(), # kind of dumb
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

        # otherwise os.mkdirs error in torch.hub.load
        torch.hub.set_dir('/root/.cache/torch/hub/%d'%dist.get_local_rank())
        model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
    else:
        assert NotImplementedError

    model.eval()

    return model, transform

class ImageNetGenAccuracyMetric(Metric):
    def __init__(self, model_name_or_path: str = 'vit_h14_in1k', **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model, self.transform = get_classification_model(model_name_or_path)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("preds", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("gts", default=torch.tensor([]), dist_reduce_fx="cat")

    # def update(self, images: Union[Tensor, List[Tensor]], text: Union[str, List[str]]) -> None:
    def update(self, images, gt_classes):
        """todo

        take batch of prompts? then update generates image AND classifies it?
        OR takes batch of images and then classifies it

        Args:
            images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
            text: Either a single caption or a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        device = images[0].device
        gt_classes = gt_classes.to(device)
        self.model = self.model.to(device)

        # we do not need to track gradients for inference
        with torch.no_grad():
            # TODO not ideal for preprocessing images :/
            images = torch.stack([self.transform(images[i]) for i in range(images.shape[0])])
            images = images.to(device)
            _, preds = self.model(images).topk(1) # [B, 1]
        preds = preds.squeeze()
        
        n_correct = torch.sum(preds == gt_classes).item()
        n_total = preds.shape[0]

        self.correct += n_correct
        self.total += n_total
        self.preds = torch.cat([self.preds, preds.cpu()])
        self.gts = torch.cat([self.gts, gt_classes.cpu()])

    def compute(self): #  -> Tensor:
        return self.correct.float() / self.total


def prep_coarse_classes():
    ''' Prepare ImageNet coarse classes for ImageNetEval plot '''

    csv_path = 'https://raw.githubusercontent.com/noameshed/novelty-detection/master/imagenet_categories.csv'
    data = pd.read_csv(csv_path)
    data = data.fillna('')

    # construct class --> coarse class mapping
    # note: maillot repeated
    # this csv has crane (bird) and crane
    # also coarse_class_mapping['amphibian'] --> 'vehicle' :/
    coarse_class_mapping = {}
    for index, row in data.iterrows():
        coarse_cat = row[0]

        if coarse_cat in ['armadillo', 'bear', 'bird', 'cat', 'crocodile', 'crustacean', 'dinosaur', 'dog', 'echinoderms', 'ferret',
                        'fish', 'frog', 'hog', 'lizard', 'marine mammals', 'marsupial', 'mollusk', 'mongoose', 'monotreme', 'primate',
                        'rabbit', 'rodent', 'salamander', 'shark', 'sloth', 'snake', 'trilobite', 'turtle', 'ungulate', 'wild cat',
                        'wild dog', 'coral']:
            coarse_cat = 'animal'
        elif coarse_cat in ['arachnid', 'bug', 'butterfly']:
            coarse_cat = 'insect'
        elif coarse_cat in ['flower', 'plant', 'fungus']:
            coarse_cat = 'plant'
        elif coarse_cat in ['hat', 'clothing']:
            coarse_cat = 'clothing'
        elif coarse_cat in ['vegetable', 'fruit', 'food']:
            coarse_cat = 'food'
        elif coarse_cat in ['outdoor scene', 'fence']:
            coarse_cat = 'outdoor scene'
        elif coarse_cat in ['train', 'vehicle', 'boat', 'aircraft']:
            coarse_cat = 'vehicle'
        elif coarse_cat in ['lab equipment', 'other', 'toy', 'paper']:
            coarse_cat = 'other'
        elif coarse_cat in ['ball', 'sports equipment']:
            coarse_cat = 'sports equipment'
        elif coarse_cat in ['accessory', 'clothing']:
            coarse_cat = 'clothing'
        
        for cls_name in row[1:]:
            if cls_name != '':
                coarse_class_mapping[cls_name] = coarse_cat


    # compute coarse class --> fine class count, just for plotting class counts
    coarse_to_fine = {}
    for cls_name in coarse_class_mapping:
        coarse_class = coarse_class_mapping[cls_name]
        if coarse_class in coarse_to_fine:
            coarse_to_fine[coarse_class] += 1
        else:
            coarse_to_fine[coarse_class] = 1

    return coarse_class_mapping, coarse_to_fine


class ImageNetEval:
    """Evaluator for CLIP, FID, KID, CLIP-FID scores using clean-fid.

    See https://github.com/GaParmar/clean-fid for more information on clean-fid.

    CLIP scores are computed using the torchmetrics CLIPScore metric.

    Args:
        model (ComposerModel): The model to evaluate.
        eval_dataloader (DataLoader): The dataloader to use for evaluation.
        clip_metric (CLIPScore): The CLIPScore metric to use for evaluation. # TODO we are not using this
        load_path (str, optional): The path to load the model from. Default: ``None``.
        guidance_scales (List[float]): The guidance scales to use for evaluation.
            Default: ``[1.0]``.
        size (int): The size of the images to generate. Default: ``256``.
        batch_size (int): The per-device batch size to use for evaluation. Default: ``16``.
        loggers (List[LoggerDestination], optional): The loggers to use for logging results. Default: ``None``.
        seed (int): The seed to use for evaluation. Default: ``17``.
        output_dir (str): The directory to save results to. Default: ``/tmp/``.
        num_samples (int, optional): The maximum number of samples to generate. Depending on batch size, actual
            number may be slightly higher. If not specified, all the samples in the dataloader will be used.
            Default: ``None``.
        precision (str): The precision to use for evaluation. Default: ``'amp_fp16'``.
        prompts (List[str], optional): The prompts to use for image visualtization.
            Default: ``["A shiba inu wearing a blue sweater]``.
    """
    def __init__(self,
                 classifier_model_name_or_path: str,
                 model: ComposerModel,
                 eval_dataloader: DataLoader,
                 clip_metric: CLIPScore,
                 load_path: Optional[str] = None,
                 guidance_scales: Optional[List[float]] = None,
                 num_epochs: int = 10,
                 size: int = 256,
                 batch_size: int = 16,
                 loggers: Optional[List[LoggerDestination]] = None,
                 seed: int = 17,
                 output_dir: str = '/tmp/',
                 num_samples: Optional[int] = None,
                 precision: str = 'amp_fp16',
                 prompts: Optional[List[str]] = None):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.num_epochs = num_epochs
        self.clip_metric = clip_metric
        self.load_path = load_path
        self.guidance_scales = guidance_scales if guidance_scales is not None else [1.0]
        self.size = size
        self.batch_size = batch_size
        self.loggers = loggers
        self.seed = seed
        self.output_dir = output_dir
        self.num_samples = num_samples if num_samples is not None else float('inf')
        self.precision = precision
        self.prompts = prompts if prompts is not None else ['A Labrador retriever']

        # Init loggers
        if self.loggers and dist.get_local_rank() == 0:
            for logger in self.loggers:
                if isinstance(logger, WandBLogger):
                    wandb.init(**logger._init_kwargs)

        # Load the model
        Trainer(model=self.model,
                load_path=self.load_path,
                load_weights_only=True,
                load_strict_model_weights=False,
                eval_dataloader=self.eval_dataloader,
                seed=self.seed)

        # Move CLIP metric to device
        self.device = dist.get_local_rank()
        self.metric = ImageNetGenAccuracyMetric(classifier_model_name_or_path)

        # Set up things for plotting
        self.coarse_class_mapping, self.coarse_to_fine = prep_coarse_classes()
        self.class_strings = self.eval_dataloader.dataset.get_class_strings()

    def _generate_images(self, guidance_scale: float):
        """Core image generation function. Generates images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.
        """
        # Verify output dirs exist, if they don't, create them
        gen_image_path = os.path.join(self.output_dir, f'gen_images_gs_{guidance_scale}')
        if not os.path.exists(gen_image_path) and dist.get_local_rank() == 0:
            os.makedirs(gen_image_path)

        # Reset the CLIP metric
        self.metric.reset()

        # TODO aggregate and then write out per-rank predictions
        pred_dict = {}
        all_preds = []
        all_gt_classes = []

        # Iterate over the eval dataloader
        # num_batches = len(self.eval_dataloader) * self.num_epochs
        # starting_seed = self.seed + num_batches * dist.get_local_rank()
        for ep in range(self.num_epochs):
            seed = self.seed + ep # just need different seed per epoch?
            for batch_id, batch in tqdm(enumerate(self.eval_dataloader)):
                # Break if enough samples have been generated
                if batch_id * self.batch_size * dist.get_world_size() >= self.num_samples:
                    break

                captions = batch['tokenized_prompt']
                prompts = batch['prompt']
                # # Ensure a new seed for each batch, as randomness in model.generate is fixed.
                # seed = starting_seed + (ep*len(self.eval_dataloader) + batch_id)
                # Generate images from the captions
                with get_precision_context(self.precision):
                    generated_images = self.model.generate(tokenized_prompts=captions,
                                                        height=self.size,
                                                        width=self.size,
                                                        guidance_scale=guidance_scale,
                                                        seed=seed,
                                                        progress_bar=False)  # type: ignore
                # Compute metric
                self.metric.update((generated_images * 255).to(torch.uint8), batch['class_idx'])

                # # Save the generated images
                # for i, img in enumerate(generated_images):
                #     if i%4 == 0: # subsample
                #         save_prompt = prompts[i].replace(' ', '-')
                #         to_pil_image(img).save(f'{gen_image_path}/{save_prompt}_{batch_id}_{i}_rank_{dist.get_local_rank()}.png')

        # Save the prompts as json
        # TODO save so we can keep it later
        # json.dump(pred_dict, open(f'{real_image_path}/prompts_rank_{dist.get_local_rank()}.json', 'w'))

    def _generate_images_from_prompts(self, guidance_scale: float):
        """Generate images from prompts for visualization."""
        if self.prompts:
            with get_precision_context(self.precision):
                generated_images = self.model.generate(prompt=self.prompts,
                                                       height=self.size,
                                                       width=self.size,
                                                       guidance_scale=guidance_scale,
                                                       seed=self.seed)  # type: ignore
        else:
            generated_images = []
        return generated_images

    def _generate_coarse_cat_plot(self, gts, preds):
        # construct empty counter dict
        coarse_class_dict = {}
        for key in self.coarse_to_fine.keys():
            coarse_class_dict['%s (%d)'%(key, self.coarse_to_fine[key])] = []

        for idx, class_str in enumerate(self.class_strings):
            gt_mask = gts == idx # boolean arr picking out location of that class
            correct_classification = gts[gt_mask] == preds[gt_mask] # whether or not the classification was corrrect
            coarse_class = self.coarse_class_mapping[class_str]
            coarse_class_dict['%s (%d)'%(coarse_class, self.coarse_to_fine[coarse_class])].extend(correct_classification.tolist())

        coarse_cats = []
        avg_classification = []
        for key in sorted(coarse_class_dict.keys()):
            n_correct = np.sum(coarse_class_dict[key])
            n_total = len(coarse_class_dict[key])
            avg_class = n_correct / float(n_total)
            coarse_cats.append(key)
            avg_classification.append(avg_class)

        fig = plt.figure(figsize = (10, 5), constrained_layout=True)
        plt.bar(coarse_cats, avg_classification)
        plt.xlabel("Coarse Categories", fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0, 1.0)
        plt.ylabel("Classification Acc on Generated Images", fontsize=12)

        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        img = torch.tensor(np.array(img)).permute(2,0,1) # horrible lol
        return img

    def _compute_metrics(self, guidance_scale: float):
        """Compute metrics for the generated images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.

        Returns:
            Dict[str, float]: The computed metrics.
        """

        metrics = {}
        score = self.metric.compute()
        metrics['ImagenetAcc'] = score
        print(f'{guidance_scale} Generated ImageNet Class Accuracy: {score}')

        # Generate accuracy per coarse category
        fig = self._generate_coarse_cat_plot(self.metric.gts.int(), self.metric.preds.int())
        metrics['CoarseCatPlot'] = fig

        return metrics

    def evaluate(self):
        # Generate images and compute metrics for each guidance scale      
        for guidance_scale in self.guidance_scales:
            dist.barrier()
            # Generate images and compute metrics
            self._generate_images(guidance_scale=guidance_scale)
            # Need to wait until all ranks have finished generating images before computing metrics
            dist.barrier()

            # Compute the metrics on the generated images
            metrics = self._compute_metrics(guidance_scale=guidance_scale)

            # Generate images from prompts for visualization
            generated_images = self._generate_images_from_prompts(guidance_scale=guidance_scale)

            # Log metrics and images on rank 0
            if self.loggers and dist.get_local_rank() == 0:
                for logger in self.loggers:
                    for metric, value in metrics.items():
                        if metric == 'CoarseCatPlot':
                            logger.log_images(images=value, name=f'barchart_gs_{guidance_scale}')
                        else:
                            logger.log_metrics({f'{guidance_scale}/{metric}': value})
                    for prompt, image in zip(self.prompts, generated_images):
                        logger.log_images(images=image, name=f'{prompt}_gs_{guidance_scale}')
