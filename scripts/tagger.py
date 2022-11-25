import os
import json
import gradio as gr

from typing import Tuple, List, Dict
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks, ui
from modules import generation_parameters_copypaste as parameters_copypaste
from modules.images import sanitize_filename_part

from tagger import format
from tagger.interrogator import Interrogator, DeepDanbooruInterrogator, WaifuDiffusionInterrogator
from tagger.utils import split_str

script_dir = Path(scripts.basedir())
preset_dir = script_dir.joinpath('presets')

interrogators: Dict[str, Interrogator] = {}

option_components: List[object] = []


def load_preset(filename: str) -> Tuple[str, Dict[str, Dict[str, any]]]:
    if not filename.endswith('.json'):
        filename += '.json'

    path = preset_dir.joinpath(sanitize_filename_part(filename))
    configs = {}

    if path.is_file():
        configs = json.loads(path.read_text())

    return path, configs


def save_preset(filename: str, *values) -> Tuple:
    path, configs = load_preset(filename)

    for index, component in enumerate(option_components):
        config = configs.get(component.path, {})
        config['value'] = values[index]

        for attr in ['visible', 'min', 'max', 'step']:
            if hasattr(component, attr):
                config[attr] = config.get(attr, getattr(component, attr))

        configs[component.path] = config

    preset_dir.mkdir(0o777, True, True)
    path.write_text(
        json.dumps(configs, indent=4)
    )

    return 'successfully saved the preset'


def apply_preset(filename: str) -> Tuple:
    configs = load_preset(filename)[1]
    outputs = []

    for component in option_components:
        config = configs.get(component.path, {})

        if 'value' in config and hasattr(component, 'choices'):
            if config['value'] not in component.choices:
                config['value'] = None

        outputs.append(component.update(**config))

    return (*outputs, 'successfully loaded the preset')


def refresh_presets() -> List[str]:
    presets = [
        p.name
        for p in preset_dir.glob('*.json')
        if p.is_file()
    ]

    if len(presets) < 1:
        presets.append('default.json')

    return presets


def refresh_interrogators() -> List[str]:
    global interrogators
    interrogators = {}

    # load waifu diffusion 1.4 tagger models
    # TODO: temporary code, should use shared.models_path later
    if os.path.isdir(Path(script_dir, '2022_0000_0899_6549')):
        interrogators['wd14'] = WaifuDiffusionInterrogator(
            Path(script_dir, 'networks', 'ViTB16_11_03_2022_07h05m53s'),
            Path(script_dir, '2022_0000_0899_6549', 'selected_tags.csv')
        )

    # load deepdanbooru project
    os.makedirs(
        shared.cmd_opts.deepdanbooru_projects_path,
        exist_ok=True
    )

    for path in os.scandir(shared.cmd_opts.deepdanbooru_projects_path):
        if not path.is_dir():
            continue

        if not Path(path, 'project.json').is_file():
            continue

        interrogators[path.name] = DeepDanbooruInterrogator(path)

    return sorted(interrogators.keys())


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
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

                        batch_output_filename_format = gr.Textbox(
                            label='Output filename format',
                            placeholder='Leave blank to use same filename as original.',
                            value='[name].[output_extension]'
                        )

                        import hashlib
                        with gr.Accordion(
                            label='Output filename formats',
                            open=False
                        ):
                            gr.Markdown(
                                value=f'''
                                ### Related to original file
                                - `[name]`: Original filename without extension
                                - `[extension]`: Original extension
                                - `[hash:<algorithms>]`: Original extension
                                    Available algorithms: `{', '.join(hashlib.algorithms_available)}`

                                ### Related to output file
                                - `[output_extension]`: Output extension (has no dot)

                                ## Examples
                                ### Original filename without extension
                                `[name].[output_extension]`

                                ### Original file's hash (good for deleting duplication)
                                `[hash:sha1].[output_extension]`
                                '''
                            )

                submit = gr.Button(
                    value='Interrogate',
                    variant='primary'
                )

                info = gr.HTML()

                # preset selector
                with gr.Row(variant='compact'):
                    presets = refresh_presets()
                    preset = gr.Dropdown(
                        label='Preset',
                        choices=presets,
                        value=presets[0]
                    )

                    preset_save_button = gr.Button(
                        value=ui.save_style_symbol
                    )

                    ui.create_refresh_button(
                        preset,
                        lambda: None,
                        lambda: {'choices': refresh_presets()},
                        'refresh_preset'
                    )

                # option components
                # components in this block can be saved or loaded as presets
                with gr.Blocks() as option_block:
                    default_configs = load_preset('default.json')[1]

                    # finds the config key from the label and makes the default kwargs
                    def new(cls: object, **kwargs):
                        return cls(**{
                            **kwargs,
                            **default_configs.get(f"/{kwargs.get('label')}", {})
                        })

                    # interrogator selector
                    with gr.Row(variant='compact'):
                        interrogator_names = refresh_interrogators()
                        interrogator = new(
                            gr.Dropdown,
                            label='Interrogator',
                            choices=interrogator_names,
                            value=(
                                None
                                if len(interrogator_names) < 1 else
                                interrogator_names[-1]
                            )
                        )

                        ui.create_refresh_button(
                            interrogator,
                            lambda: None,
                            lambda: {'choices': refresh_interrogators()},
                            'refresh_interrogator'
                        )

                    threshold = new(
                        gr.Slider,
                        label='Threshold',
                        minimum=0,
                        maximum=1,
                        value=0.35
                    )

                    exclude_tags = new(
                        gr.Textbox,
                        label='Exclude tags (split by comma)'
                    )

                    sort_by_alphabetical_order = new(
                        gr.Checkbox,
                        label='Sort by alphabetical order',
                    )
                    add_confident_as_weight = new(
                        gr.Checkbox,
                        label='Include confident of tags matches in results',
                    )
                    replace_underscore = new(
                        gr.Checkbox,
                        label='Use spaces instead of underscore',
                        value=True
                    )
                    replace_underscore_excludes = new(
                        gr.Textbox,
                        label='Excudes (split by comma)',
                        # kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
                        value='0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
                    )
                    escape_tag = new(
                        gr.Checkbox,
                        label='Escape brackets',
                    )

                # list of components that can be set via preset
                def on_visit(path, component):
                    setattr(component, 'path', path)
                    option_components.append(component)

                ui.visit(option_block, on_visit)

                preset.change(
                    fn=apply_preset,
                    inputs=[preset],
                    outputs=[*option_components, info]
                )

                preset_save_button.click(
                    fn=save_preset,
                    inputs=[preset, *option_components],  # values only
                    outputs=[info]
                )

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
            batch_input_glob: str,
            batch_input_recursive: bool,
            batch_output_dir: str,
            batch_output_type: str,
            batch_output_filename_format: str,

            interrogator: str,
            threshold: float,
            exclude_tags: str,
            sort_by_alphabetical_order: bool,
            add_confident_as_weight: bool,
            replace_underscore: bool,
            replace_underscore_excludes: str,
            escape_tag: bool
        ):
            if interrogator not in interrogators:
                return ['', None, None, f"'{interrogator}' is not a valid interrogator"]

            interrogator: Interrogator = interrogators[interrogator]

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
                ratings, tags = interrogator.interrogate(image)
                processed_tags = Interrogator.postprocess_tags(
                    tags,
                    *postprocess_opts
                )

                return [
                    ', '.join(processed_tags),
                    ratings,
                    tags,
                    ''
                ]

            # batch process
            batch_input_glob = batch_input_glob.strip()
            batch_output_dir = batch_output_dir.strip()
            batch_output_filename_format = batch_output_filename_format.strip()

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
                    return ['', None, None, 'input path is not a directory']

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

                    ratings, tags = interrogator.interrogate(image)
                    processed_tags = Interrogator.postprocess_tags(
                        tags,
                        *postprocess_opts
                    )

                    # TODO: switch for less print
                    print(
                        f'found {len(processed_tags)} tags out of {len(tags)} from {path}'
                    )

                    # guess the output path
                    output_dir = Path(
                        base_dir if batch_output_dir == '' else batch_output_dir,
                        os.path.dirname(str(path).removeprefix(base_dir + '/'))
                    )

                    output_ext = 'txt'
                    if batch_output_type == 'JSON':
                        output_ext = 'json'

                    # format output filename
                    format_info = format.Info(path, output_ext)

                    try:
                        formatted_output_filename = format.pattern.sub(
                            lambda m: format.format(m, format_info),
                            batch_output_filename_format
                        )
                    except (TypeError, ValueError) as error:
                        return ['', None, None, str(error)]

                    output_path = Path(
                        output_dir,
                        formatted_output_filename
                    )

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
                    batch_output_filename_format,

                    # options
                    interrogator,
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
