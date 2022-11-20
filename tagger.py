import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from typing import Tuple, List, Dict
from PIL import Image

from modules import shared, scripts
from modules.deepbooru import re_special as tag_escape_pattern

trained_image_size = 448

script_dir = scripts.basedir()
model_dir = os.path.join(script_dir, "networks", "ViTB16_11_03_2022_07h05m53s")
tags_path = os.path.join(
    script_dir, "2022_0000_0899_6549", "selected_tags.csv")


try:
    import Utils.dbimutils as dbimutils
except ImportError:
    raise Exception(
        'Utils.dbimutils not found, did you follow the instruction?')

if not os.path.isdir(model_dir):
    raise Exception('model not found, did you follow the instruction?')

if not os.path.isfile(tags_path):
    raise Exception('tags not found, did you follow the instruction?')

# tensorflow maps nearly all vram by default, so we limit this
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# TODO: is safe to set set_memory_growth...?
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

# load model into gpu or cpu
if shared.cmd_opts.use_cpu == 'all' or shared.cmd_opts.use_cpu == 'interrogate':
    device_name = '/cpu:0'  # 0 by default?
else:
    device_name = '/gpu:0'  # 0 by default?

    if shared.cmd_opts.device_id is not None:
        try:
            device_name = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')

with tf.device(device_name):
    # TODO: need gc after restart, is there anyway to hook into restart event?
    model = tf.keras.models.load_model(model_dir, compile=False)

selected_tags = pd.read_csv(tags_path)


def interrogate_tags(image: Image) -> Tuple[
    Dict[str, float],  # found rating tags (tag, confident)
    Dict[str, float]  # found general tags (tag, confident)
]:
    image.convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = dbimutils.smart_24bit(image)
    image = dbimutils.make_square(image, trained_image_size)
    image = dbimutils.smart_resize(image, trained_image_size)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    confidents = model(image, training=False)

    all_tags = selected_tags[:][['name']]
    all_tags['confidents'] = confidents[0]

    # first 4 items are for rating (general, sensitive, questionable, explicit)
    ratings = dict(all_tags[:4].values)

    # rest are regular tags
    tags = dict(all_tags[4:].values)

    return ratings, tags


def postprocess_tags(
    tags: Dict[str, float],

    threshold=0.35,
    exclude_tags: List[str] = [],
    sort_by_alphabetical_order=False,
    add_confident_as_weight=False,
    replace_underscore=False,
    replace_underscore_excludes: List[str] = [],
    escape_tag=False
) -> Dict[str, float]:

    # those lines are totally not "pythonic" but looks better to me
    tags = {
        t: c

        # sort by tag name or confident
        for t, c in sorted(
            tags.items(),
            key=lambda i: i[0 if sort_by_alphabetical_order else 1],
            reverse=not sort_by_alphabetical_order
        )

        # filter tags
        if (
            c >= threshold
            and t not in exclude_tags
        )
    }

    for tag in list(tags):
        new_tag = tag

        if replace_underscore and tag not in replace_underscore_excludes:
            new_tag = new_tag.replace('_', ' ')

        if escape_tag:
            new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

        if add_confident_as_weight:
            new_tag = f'({new_tag}:{tags[tag]})'

        if new_tag != tag:
            tags[new_tag] = tags.pop(tag)

    return tags
