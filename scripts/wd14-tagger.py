import os
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from webui import wrap_gradio_gpu_call
from modules import scripts, script_callbacks

image_size = 448

script_dir = scripts.basedir()
model_dir = os.path.join(script_dir, "networks", "ViTB16_11_03_2022_07h05m53s")
tags_path = os.path.join(
    script_dir, "2022_0000_0899_6549", "selected_tags.csv")

try:
    # not sure this line is valid or not... I'm not good at python :\
    import Utils.dbimutils as dbimutils
except ImportError:
    raise Exception(
        'Utils.dbimutils not found, did you follow the instruction?')

if not os.path.isdir(model_dir):
    raise Exception('model not found, did you follow the instruction?')

if not os.path.isfile(tags_path):
    raise Exception('tags not found, did you follow the instruction?')

# TODO: support gpu?
with tf.device('/cpu:0'):
    model = tf.keras.models.load_model(model_dir, compile=False)

selected_tags = pd.read_csv(tags_path)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                threshold = gr.Slider(
                    label='Threshold',
                    minimum=0,
                    maximum=1,
                    value=0.35
                )

                sort_by_alpha = gr.Checkbox(
                    label='sort by alphabetical order')
                add_confident = gr.Checkbox(
                    label='include confident of tags matches in results')
                space_instead_underscore = gr.Checkbox(
                    label='use spaces instead of underscore')
                escape_tags = gr.Checkbox(
                    label='escape brackets')

            with gr.Column(variant='panel'):
                image = gr.Image(
                    label='Source',
                    source='upload',
                    interactive=True,
                    type="pil"
                )

                info = gr.HTML()

            with gr.Column(variant='panel'):
                tags = gr.Textbox(
                    label='Tags',
                    placeholder='Found tags'
                )

                # TODO: button to move found tags to txt2img or img2img

                rating_confidents = gr.Label(label='Rating confidents')
                tag_confidents = gr.Label(label='Tag confidents')

        def on_image_change(image, threshold, sort_by_alpha, add_confident, space_instead_underscore, escape_tags):
            if image is None:
                return ['', None, None, '']

            image.convert("RGB")
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = dbimutils.smart_24bit(image)
            image = dbimutils.make_square(image, image_size)
            image = dbimutils.smart_resize(image, image_size)
            image = image.astype(np.float32)
            image = np.expand_dims(image, 0)

            probs = model(image, training=False)

            all_tags = selected_tags.copy()
            all_tags['probs'] = probs[0]

            # first 4 items is for rating (general, sensitive, questionable, explicit)
            ratings = all_tags[:4][['name', 'probs']]

            # rest are regular tags
            tags = all_tags[4:]
            tags = tags[tags['probs'] > threshold]
            tags = tags[['name', 'probs']]

            rating_confidents = dict(ratings.values)
            tag_confidents = dict(tags.values)

            # TODO: implement add_confident

            if sort_by_alpha:
                tags = tags.sort_index(by='name')

            if space_instead_underscore:
                tags['name'] = tags['name'].str.replace('_', ' ')

            if escape_tags:
                from modules.deepbooru import re_special
                tags['name'] = tags['name'].str.replace(
                    re_special,
                    r'\\\1',
                    regex=True
                )

            return [
                ', '.join(tags['name']),
                rating_confidents,
                tag_confidents,
                ''
            ]

        image.change(
            fn=wrap_gradio_gpu_call(on_image_change),
            inputs=[
                image,
                threshold,
                add_confident,
                sort_by_alpha,
                space_instead_underscore,
                escape_tags
            ],
            outputs=[
                tags,
                rating_confidents,
                tag_confidents,
                info
            ]
        )

    return [(tagger_interface, "Tagger", "tagger")]


script_callbacks.on_ui_tabs(on_ui_tabs)


class Script(scripts.Script):
    def title(self):
        return "Waifu Diffusion 1.4 Tagger"

    def show(self, _):
        return scripts.AlwaysVisible
