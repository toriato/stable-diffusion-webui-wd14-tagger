import os
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from typing import Tuple, Dict
from pandas import Series
from PIL import Image

from modules import shared, scripts

trained_image_size = 448
supported_extensions = [
    e for e, f in Image.registered_extensions().items() if f in Image.OPEN
]

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


def interrogate_tags(
    image: Image,
    threshold=0.35, sort_by_alpha=False, add_confident=False, space_instead_underscore=False, escape_tags=False
) -> Tuple[
    Series,  # filtered and sorted tag names
    Dict[str, int],  # found rating tags (tag, confident)
    Dict[str, int]  # found general tags (tag, confident)
]:
    image.convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = dbimutils.smart_24bit(image)
    image = dbimutils.make_square(image, trained_image_size)
    image = dbimutils.smart_resize(image, trained_image_size)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    probs = model(image, training=False)

    all_tags = selected_tags.copy()
    all_tags['probs'] = probs[0]

    # first 4 items are for rating (general, sensitive, questionable, explicit)
    ratings = all_tags[:4][['name', 'probs']]

    # rest are regular tags
    tags = all_tags[4:]
    tags = tags[tags['probs'] > threshold]
    tags = tags[['name', 'probs']]

    rating_confidents = dict(ratings.values)
    tag_confidents = dict(tags.values)

    if sort_by_alpha:
        tags = tags.sort_values('name')

    if space_instead_underscore:
        tags['name'] = tags['name'].str.replace('_', ' ')

    if escape_tags:
        from modules.deepbooru import re_special
        tags['name'] = tags['name'].str.replace(
            re_special,
            r'\\\1',
            regex=True
        )

    if add_confident:
        tags['name'] = '(' + tags['name'] + ':' + \
            tags['probs'].astype(str) + ')'

    return tags['name'], rating_confidents, tag_confidents
