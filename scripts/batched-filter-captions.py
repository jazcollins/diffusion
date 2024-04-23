# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Script to filter captions with an LLM."""

import os
import threading
import time
from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
import wandb
from composer.utils import dist
from tqdm.auto import tqdm
from streaming import Stream
from streaming.base import MDSWriter

from transformers import AutoModelForCausalLM, AutoTokenizer
from streaming import StreamingDataset

class LLMFilter:
    """LLM filterer class."""

    def __init__(self,
                 model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2', # vs the Instruct version?
                 max_tokens: int = 1024,
                 compile: bool = False,
                 system_prompt: Optional[str] = None,
                 device: Optional[torch.device] = None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = torch.device('cuda') if device is None else device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO check we're using the right one for this model 
        self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %} {% set loop_messages = messages[1:] %} {% set system_message = messages[0]['content'].strip() + '\n\n' %} {% else %} {% set loop_messages = messages %} {% set system_message = '' %} {% endif %} {{ bos_token }} {% for message in loop_messages %} {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %} {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }} {% endif %} {% if loop.index0 == 0 %} {% set content = system_message + message['content'] %} {% else %} {% set content = message['content'] %} {% endif %} {% if message['role'] == 'user' %} {{ '[INST] ' + content.strip() + ' [/INST]' }} {% elif message['role'] == 'assistant' %} {{ ' '  + content.strip() + ' ' + eos_token }} {% endif %} {% endfor %}"

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        self.system_prompt = system_prompt

        if self.system_prompt:
            print('Using system prompt:')
            print(self.system_prompt)
        else:
            raise ValueError('must pass system prompt')
        

        self.generate = self.model.generate
        if compile:
            self.model = torch.compile(self.model)
            self.generate = torch.compile(self.generate)

    def get_outputs(self, caption: str) -> list:
        """Get the output from LLM."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": caption}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)

        # Forward through the model
        with torch.no_grad():
            output_ids = self.generate(inputs,
                                       do_sample=False, # True,
                                    #    temperature=0.2,
                                    #    top_p=None,
                                       num_beams=1,
                                       max_new_tokens=self.max_tokens,
                                       use_cache=True,
                                       pad_token_id=self.tokenizer.eos_token_id)
        # Postprocess outputs
        decoded_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs = [o.strip() for o in decoded_output]
        return outputs

def parse_args():
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--remote', type=str, help='Remote to use for the dataset.')
    parser.add_argument('--local', type=str, help='Local directory to use for the dataset.')
    parser.add_argument('--output', help='Output path for the filtered dataset.')
    parser.add_argument('--output_caption_key', type=str, default='llm_rating', help='Dataset output caption key.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the model.')
    parser.add_argument('--model_name',
                        type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='LLM model to use.')
    parser.add_argument('--system_prompt',
                        type=str,
                        default="""You rate a stock image descriptions. The user will give you an image description and you will output a rating, whether the description is "good" or "bad".
<bad-examples reason="doesn't contain enough detail to fully visualize the image">
- house interior
- Together is our favorite place to be- positive text, with heart. Use for greeting card, wall art, textile print, and gift.
</bad-examples>
<good-examples reason="excellently describes the image, mentioning objects in in the scene, colors, camera settings, and image style">
- Young hispanic man wearing casual white tshirt laughing nervous and excited with hands on chin looking to the side. Hair slicked back. He has moderate amounts of facial hair and is wearing a watch. Isolated shot red background.
- Ingredients for mojito cocktail, whole, sliced lime and mini limes, mint leaves, brown crystal sugar over gray stone texture background with gauze textile. Top view, space
</good-examples>
<instructions>
Be picky, most images should be rated as "bad". If you have any doubts or difficutly visualizing the image, it is "bad".
First, respond with your analysis, wrapped in <thoughts>(reasoning)</thoughts>
Then, respond with your single word rating, wrapped in rating tags: <rating>good/bad</rating>. After the closing tag, do not write anything else.</instructions>""",
                        help='Prompt to use for LLM.')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum tokens to generate.')
    parser.add_argument('--compile', action='store_true', help='Compile the model.')
    parser.add_argument('--quantize', action='store_true', help='Quantize the model.')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multi-gpu.')
    parser.add_argument('--start', type=int, default=0, help='Start index for the dataset.')
    parser.add_argument('--end', type=int, default=None, help='Optional end index for the dataset.')
    # Add wandb arguments
    parser.add_argument('--wandb_disabled', action='store_true')
    parser.add_argument('--wandb_name', type=str, default='llava-captions')
    parser.add_argument('--wandb_project', type=str, default='llava-captions')
    parser.add_argument('--wandb_entity', type=str, default='mosaic-ml')
    return parser.parse_args()


def make_dataset(remote: str,
                 local: str):
    """Make a streaming image dataset."""
    streams = []
    for r, l in zip([remote], [local]):
        streams.append(Stream(remote=r, local=l))

    dataset = StreamingDataset(
        streams=streams,
        shuffle=False,
    )
    return dataset


def prefetch_samples(dataset, start_idx, end_idx):
    """Walk through the dataset to prefetch samples."""
    for i in range(start_idx, end_idx):
        _ = dataset[i]


def main(args: Namespace) -> None:
    """Rate captions as good or bad.

    Args:
        args (Namespace): Command-line arguments.
    """

    good_captions = ['Two cups of cappuccino coffee with froth. On a wooden tray. On white background. Breakfast.',
                     'Happy laughing little girl holding sunflower bouquet. Child playing with sunflowers. Kids picking fresh sun flowers in the garden. Children gardening in summer. Outdoor fun for family.',
                     'Discontent caucasian teen girl wearing animal print sweater over white background shows disapproval sign, keeps thumb down, expresses dislike, frowns face in discontent. Negative feelings.',
                     'Large art brush with white bristles, paints a dark surface in bright turquoise, blue. Tool for artistic and restoration work, close-up view.',
                     'Happy Birthday Banner. Vector Illustration of a Happy Birthday Greeting Card with balloons for children. The words "Happy Birthday" are written with each letter a different color of the rainbow. 4 colorful balloons on the left side, 5 balloons on the right side',
                     'A pack of cigarettes isolated on a white background. Four cigarettes are partially pulled out of the pack, with varying heights. The pack is white.',
                     'Glass jug with cold lemonade and slices of lemon, orange. Homemade refreshing, summer drink. Doodle. Drawn by hand. Vector illustration. Outline. Black and white line drawing.',
                     'Head of a piebal horned goat in the pasture. Animal nose close-up, selective focus. Goat looking at the camera. Feeding of cattle on farmland grassland. Countryside concepts.']
    bad_captions = ['Flag and map of China',
                    'spike chart outline vector icon. simple element illustration. spike chart outline icon from editable business concept. can be used for web and mobile',
                    'food in a bowl on a table with a white tablecloth at christmas dinner',
                    'Rough, scratch, splatter grunge pattern design brush strokes. Overlay texture. Faded black-white dyed paper texture. Sketch grunge design. Use for poster, cover, banner, mock-up, stickers layout.',
                    'Infinity logo template vector illustration',
                    'portrait of a happy family on the nature',
                    'Geometry texture repeat creative modern pattern',
                    'chocolate isolated on white background. studio shooting']

    # Device should be first gpu if available, else cpu
    device = torch.device(f'cuda:{dist.get_local_rank()}' if torch.cuda.is_available() else 'cpu')
    captioner = LLMFilter(model_name=args.model_name,
                          max_tokens=args.max_tokens,
                          compile=args.compile,
                          system_prompt=args.system_prompt,
                          device=device)
    
    all_captions = good_captions + bad_captions
    all_labels = ['good'] * len(good_captions) + ['bad'] * len(bad_captions)

    n_correct = 0
    for caption, label in zip(all_captions, all_labels):
        outputs = captioner.get_outputs(caption)
        
        response = outputs[0].split('[/INST]')[-1] # TODO make sure we're doing the right splitting for the right chat template
        if '<rating>' in response:
            rating = response.split('<rating>')[1].split('</rating>')[0]
            thoughts = response.split('<thoughts>')[1].split('</thoughts>')[0]
        else:
            print('no rating logged, skipping')
            continue

        print('caption:', caption)
        print('gt:', label)
        print('rating:', rating) # check if not good/bad (sometimes model says neutral)
        if label == rating:
            n_correct += 1
        print('justification:', thoughts)
        print('----------')
    
    print('total accuracy:', n_correct/len(all_captions))


if __name__ == '__main__':
    main(parse_args())
