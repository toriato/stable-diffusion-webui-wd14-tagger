import os
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from typing import Tuple, Dict
from pathlib import Path
from pandas import Series
from PIL import Image, UnidentifiedImageError

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks
from modules import generation_parameters_copypaste as parameters_copypaste

image_size = 448
image_supported_exts = [
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
# TODO: is safe to use None with the tensorflow.device(device_name)?
device_name = None

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
    image = dbimutils.make_square(image, image_size)
    image = dbimutils.smart_resize(image, image_size)
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


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        dummy_component = gr.Label(visible=False)

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                # input components
                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )

                    if shared.cmd_opts.hide_ui_dir_config:
                        batch_input_glob = dummy_component
                        batch_input_recursive = dummy_component
                        batch_output_dir = dummy_component
                        batch_output_type = dummy_component

                    else:
                        with gr.TabItem(label='Batch from directory'):
                            batch_input_glob = gr.Textbox(
                                label='Input directory',
                                placeholder='/path/to/images/* or /path/to/images/**/*'
                            )
                            batch_input_recursive = gr.Checkbox(
                                label='Use recursive with glob pattern'
                            )

                            batch_output_dir = gr.Textbox(
                                label='Output directory',
                                placeholder='Leave blank to save images to the same path.'
                            )

                            batch_output_type = gr.Dropdown(
                                label='Output type',
                                value='Flatfile',
                                choices=[
                                    'Flatfile',
                                    'JSON'
                                ]
                            )

                submit = gr.Button(value='Interrogate')

                info = gr.HTML()

                # option components
                # TODO: move to shared.opts?
                threshold = gr.Slider(
                    label='Threshold',
                    minimum=0,
                    maximum=1,
                    value=0.35
                )

                sort_by_alpha = gr.Checkbox(
                    label='Sort by alphabetical order')
                add_confident = gr.Checkbox(
                    label='Include confident of tags matches in results')
                space_instead_underscore = gr.Checkbox(
                    label='Use spaces instead of underscore')
                escape_tags = gr.Checkbox(
                    label='Escape brackets')

            # output components
            with gr.Column(variant='panel'):
                tags = gr.Textbox(
                    label='Tags',
                    placeholder='Found tags',
                    interactive=False
                )

                with gr.Row():
                    parameters_copypaste.bind_buttons(
                        parameters_copypaste.create_buttons(
                            ["txt2img", "img2img"],
                        ),
                        None,
                        tags
                    )

                rating_confidents = gr.Label(label='Rating confidents')
                tag_confidents = gr.Label(label='Tag confidents')

        def give_me_the_tags(
            image: Image,
            batch_input_glob: str, batch_input_recursive: bool, batch_output_dir: str, batch_output_type: str,
            threshold: bool, sort_by_alpha: bool, add_confident: bool, space_instead_underscore: bool, escape_tags: bool
        ):
            # single process
            if image is not None:
                outputs = [*interrogate_tags(
                    image,
                    threshold, sort_by_alpha, add_confident, space_instead_underscore, escape_tags
                )]

                outputs[0] = ', '.join(outputs[0])

                return [*outputs, '']

            # batch process
            batch_input_glob = batch_input_glob.strip()
            batch_output_dir = batch_output_dir.strip()

            if batch_input_glob != '':
                # get root directory of input glob pattern
                base_dir = batch_input_glob.replace('?', '*')
                base_dir = base_dir.split('/*').pop(0)

                from glob import glob
                for path in glob(batch_input_glob, recursive=batch_input_recursive):
                    path = Path(path)

                    if path.suffix not in image_supported_exts:
                        continue

                    try:
                        image = Image.open(path)
                    except UnidentifiedImageError:
                        # just in case, user has mysterious file...
                        print(f'${path} is not supported image type')
                        continue

                    tag_names, rating_confidents, tag_confidents = interrogate_tags(
                        image,
                        threshold, sort_by_alpha, add_confident, space_instead_underscore, escape_tags
                    )

                    # TODO: switch for less print
                    print(
                        f'found {len(tag_names)} tags with {list(rating_confidents.keys())[0]} rating from {path}'
                    )

                    # guess the output path
                    output_dir = Path(
                        base_dir if batch_output_dir == '' else batch_output_dir,
                        os.path.dirname(str(path).removeprefix(base_dir + '/'))
                    )

                    output_ext = 'txt'
                    if batch_output_type == 'JSON':
                        output_ext = 'json'

                    output_path = Path(output_dir, f'{path}.{output_ext}')

                    os.makedirs(output_dir, exist_ok=True)

                    # save output
                    if batch_output_type == 'JSON':
                        import json
                        output = json.dumps(
                            (rating_confidents, tag_confidents))
                    else:
                        output = ', '.join(tag_names)

                    with output_path.open('w') as f:
                        f.write(output)

            return ['', None, None, '']

        # register events
        for func in [image.change, submit.click]:
            func(
                fn=wrap_gradio_gpu_call(give_me_the_tags),
                inputs=[
                    # single process
                    image,

                    # batch process
                    batch_input_glob,
                    batch_input_recursive,
                    batch_output_dir,
                    batch_output_type,

                    # options
                    threshold,
                    sort_by_alpha,
                    add_confident,
                    space_instead_underscore,
                    escape_tags
                ],
                outputs=[
                    tags,
                    rating_confidents,
                    tag_confidents,

                    # contains execution time, memory usage and other stuffs...
                    # generated from modules.ui.wrap_gradio_call
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
