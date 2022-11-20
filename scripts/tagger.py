import os
import json
import gradio as gr

from typing import List
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks
from modules import generation_parameters_copypaste as parameters_copypaste

from tagger import interrogate_tags, postprocess_tags


def split_str(s: str, separator=',') -> List[str]:
    return list(map(str.strip, s.split(separator)))


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

                exclude_tags = gr.Textbox(
                    label='Exclude tags (split by comma)',
                )

                sort_by_alphabetical_order = gr.Checkbox(
                    label='Sort by alphabetical order')
                add_confident_as_weight = gr.Checkbox(
                    label='Include confident of tags matches in results')
                replace_underscore = gr.Checkbox(
                    label='Use spaces instead of underscore')
                replace_underscore_excludes = gr.Textbox(
                    label='Excudes (split by comma)',
                    # kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
                    value='0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
                )
                escape_tag = gr.Checkbox(
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

            threshold: float,
            exclude_tags: str,
            sort_by_alphabetical_order: bool,
            add_confident_as_weight: bool,
            replace_underscore: bool,
            replace_underscore_excludes: str,
            escape_tag: bool
        ):

            postprocess_opts = (
                threshold,
                split_str(exclude_tags),
                sort_by_alphabetical_order,
                add_confident_as_weight,
                replace_underscore,
                split_str(replace_underscore_excludes),
                escape_tag
            )

            # single process
            if image is not None:
                ratings, tags = interrogate_tags(image)
                processed_tags = postprocess_tags(tags, *postprocess_opts)

                return [
                    ', '.join(processed_tags),
                    ratings,
                    tags,
                    ''
                ]

            # batch process
            batch_input_glob = batch_input_glob.strip()
            batch_output_dir = batch_output_dir.strip()

            if batch_input_glob != '':
                # if there is no glob pattern, insert it automatically
                if not batch_input_glob.endswith('*'):
                    if not batch_input_glob.endswith('/'):
                        batch_input_glob += '/'
                    batch_input_glob += '*'

                # get root directory of input glob pattern
                base_dir = batch_input_glob.replace('?', '*')
                base_dir = base_dir.split('/*').pop(0)

                # check the input directory path
                if not os.path.isdir(base_dir):
                    return ['', None, None, f'input path is not a directory']

                # this line is moved here because some reason
                # PIL.Image.registered_extensions() returns only PNG if you call too early
                supported_extensions = [
                    e
                    for e, f in Image.registered_extensions().items()
                    if f in Image.OPEN
                ]

                paths = [
                    Path(p)
                    for p in glob(batch_input_glob, recursive=batch_input_recursive)
                    if '.' + p.split('.').pop().lower() in supported_extensions
                ]

                print(f'found {len(paths)} image(s)')

                for path in paths:
                    try:
                        image = Image.open(path)
                    except UnidentifiedImageError:
                        # just in case, user has mysterious file...
                        print(f'${path} is not supported image type')
                        continue

                    ratings, tags = interrogate_tags(image)
                    processed_tags = postprocess_tags(tags, *postprocess_opts)

                    # TODO: switch for less print
                    print(
                        f'found {len(processed_tags)} tags with {list(ratings.keys())[0]} rating from {path}'
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
                        output = json.dumps((ratings, tags))
                    else:
                        output = ', '.join(processed_tags)

                    with output_path.open('w') as f:
                        f.write(output)

                print('all done :)')

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
                    exclude_tags,
                    sort_by_alphabetical_order,
                    add_confident_as_weight,
                    replace_underscore,
                    replace_underscore_excludes,
                    escape_tag
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
