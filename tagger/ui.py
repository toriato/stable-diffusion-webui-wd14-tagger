import os
import json
import gradio as gr
import tqdm
import re

from collections import OrderedDict
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError

from webui import wrap_gradio_gpu_call
from modules import ui
from modules import generation_parameters_copypaste as parameters_copypaste

from tagger import format, utils
from tagger.utils import split_str
from tagger.interrogator import Interrogator


def unload_interrogators():
    unloaded_models = 0

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1

    return [f'Successfully unload {unloaded_models} model(s)']


def on_interrogate(
    image: Image,
    batch_input_glob: str,
    batch_input_recursive: bool,
    batch_output_dir: str,
    batch_output_filename_format: str,
    batch_output_action_on_conflict: str,
    batch_remove_duplicated_tag: bool,
    batch_output_save_json: bool,

    interrogator: str,
    threshold: float,
    tag_count_threshold: float,
    additional_tags: str,
    exclude_tags: str,
    search_tags: str,
    replace_tags: str,
    ingore_case: bool,
    sort_by_alphabetical_order: bool,
    add_confident_as_weight: bool,
    replace_underscore: bool,
    replace_underscore_excludes: str,
    escape_tag: bool,

    unload_model_after_running: bool,
    verbose: bool
):
    if interrogator not in utils.interrogators:
        return ['', None, None, f"'{interrogator}' is not a valid interrogator"]

    interrogator: Interrogator = utils.interrogators[interrogator]

    postprocess_opts = (
        threshold,
        split_str(additional_tags),
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

        if unload_model_after_running:
            interrogator.unload()

        return [
            ', '.join(processed_tags.keys() if add_confident_as_weight else processed_tags),
            ratings,
            dict(filter(lambda x: x[1] >= threshold / 2, tags.items())),
            ''
        ]

    # batch process
    batch_input_glob = batch_input_glob.strip()
    batch_output_dir = batch_output_dir.strip()
    batch_output_filename_format = batch_output_filename_format.strip()
    combined = {}
    nr_of_files = 0

    if batch_input_glob != '':
        # if there is no glob pattern, insert it automatically
        if not batch_input_glob.endswith('*'):
            if not batch_input_glob.endswith(os.sep):
                batch_input_glob += os.sep
            batch_input_glob += '*'

        # get root directory of input glob pattern
        base_dir = batch_input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)

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
        nr_of_files = len(paths)

        add_tag_list = list(map(str.strip, additional_tags.split(',')))
        rem_tags = list(map(str.strip, exclude_tags.split(',')))

        # the first entry can be a regexp, if a string like /.*/
        # TODO: 3.11 supports re.NOFLAG, value is also 0.
        re_f = re.IGNORECASE if ingore_case else 0
        rem_re = None
        if len(rem_tags) > 0:
            if len(rem_tags[0]) > 2:
                if rem_tags[0][0] == '/' and rem_tags[0][-1] == '/':
                    rem_re = re.compile(rem_tags[0][1:-1], flags=re_f)
                    del rem_tags[0]

        rem_tags = set(rem_tags)

        def re_comp(x):
            return re.compile('^'+str.strip(x)+'$', flags=re_f)

        search_tag_list = list(map(re_comp, search_tags.split(',')))
        replace_tag_list = list(map(str.strip, replace_tags.split(',')))
        if len(search_tag_list) != len(replace_tag_list):
            print("search and replace strings have different counts, ignoring")
            search_tag_list = []
            replace_tag_list = []

        print(f'found {len(paths)} image(s)')
        for path in tqdm.tqdm(paths, disable=verbose, desc='Tagging'):
            try:
                image = Image.open(path)
            except UnidentifiedImageError:
                # just in case, user has mysterious file...
                print(f'${path} is not supported image type')
                continue

            # guess the output path
            base_dir_last = Path(base_dir).parts[-1]
            base_dir_last_idx = path.parts.index(base_dir_last)
            output_dir = Path(
                batch_output_dir) if batch_output_dir else Path(base_dir)
            output_dir = output_dir.joinpath(
                *path.parts[base_dir_last_idx + 1:]).parent

            output_dir.mkdir(0o777, True, True)

            # format output filename
            format_info = format.Info(path, 'txt')

            try:
                formatted_output_filename = format.pattern.sub(
                    lambda m: format.format(m, format_info),
                    batch_output_filename_format
                )
            except (TypeError, ValueError) as error:
                return ['', None, None, str(error)]

            output_path = output_dir.joinpath(formatted_output_filename)

            # instead of pre- or appending weights, update them (re-average).
            # ict, the interrogation count, is used to re-average
            # the default of 0.0 will be overridden, when read from file
            weights = {}
            ict = 0.0
            output = []

            if output_path.is_file():
                if batch_output_action_on_conflict != 'replace':
                    txt = output_path.read_text(errors='ignore').strip()
                    output = list(map(str.strip, txt.split(',')))
                    # read the previous interrogation count, if any
                    if len(output) > 0:
                        try:
                            ict = float(output[-1])
                            del output[-1]
                        except ValueError:
                            pass

                    n = len(output)
                    for i in range(n):
                        # split the key and weight, if present
                        k = output[i]
                        at = k.rfind(':')

                        if at > 0 and k[0] == '(' and k[-1] == ')':
                            v = float(k[at+1:-1])
                            weights[k[1:at]] = v
                            if not add_confident_as_weight:
                                output[i] = k[1:at]
                        else:
                            # FIXME: exponential, hyperbolic, or harmonic
                            # decline is probably a more realistic fit
                            if i == 0:
                                print(f'{path}: no weights, assumed linear')
                                # and that the prior was one iteration
                                ict = 1.0
                            weights[k] = (n - i) / n

                if batch_output_action_on_conflict == 'ignore':
                    if verbose:
                        print(f'skipping {path}')
                    for k, v in weights.items():
                        combined[k] = combined[k] + v if k in combined else v
                    continue
            if batch_output_action_on_conflict != 'filter':
                ratings, tags = interrogator.interrogate(image)
                processed_tags = Interrogator.postprocess_tags(
                    tags,
                    *postprocess_opts
                )
            else:
                for t in add_tag_list:
                    weights[t] = 1.0

                for k in list(weights.keys()):
                    if k in rem_tags or rem_re and rem_re.match(k):
                        del weights[k]
                        continue

                    for i in range(len(search_tag_list)):
                        search = search_tag_list[i]
                        replace = replace_tag_list[i]

                        if replace and re.match(search, k):
                            v = weights[k]
                            del weights[k]
                            k = re.sub(search, replace, k, 1)
                            if k in weights:
                                v = v + weights[k]
                            weights[k] = v
                            break

                processed_tags = weights
                tags = weights.keys()
                tags.sort(key=lambda x: weights[x], reverse=True)
                batch_output_save_json = False
                output = None

            if tag_count_threshold < len(processed_tags):
                processed_tags = [(k, v) for k, v in processed_tags.items()]
                processed_tags.sort(key=lambda x: x[1], reverse=True)
                processed_tags = dict(processed_tags[:int(tag_count_threshold)])

            if verbose:
                print(f'{path}: {len(processed_tags)}/{len(tags)} tags found')

            if batch_output_action_on_conflict == 'replace' or not output:
                for k, v in processed_tags.items():
                    # processed tags keys, if weighed, contain value, strip it.
                    at = k.rfind(':')
                    if at > 0 and k[0] == '(':
                        k = k[1:at]
                    combined[k] = combined[k] + v if k in combined else v
                output = [', '.join(processed_tags)]

            elif len(processed_tags) > 0:
                if add_confident_as_weight:
                    output = []

                    for k, v in processed_tags.items():
                        at = k.rfind(':')
                        if at > 0 and k[0] == '(':
                            k = k[1:at]

                        # recurrent weights or new ones are re-averaged
                        if k in weights:
                            v += weights[k] * ict
                            del weights[k]
                        v = v / (ict + 1.0)
                        output.append((k, v))
                        combined[k] = combined[k] + v if k in combined else v

                    # non-recurrent weights are also re-averaged
                    for k, v in weights.items():
                        v = (v * ict) / (ict + 1.0)
                        output.append((k, v))
                        combined[k] = combined[k] + v if k in combined else v

                    output.sort(key=lambda x: x[1], reverse=True)
                    output = list(map(lambda x: '(%s:%f)' % x, output))
                elif batch_output_action_on_conflict == 'prepend':
                    output.insert(0, ', '.join(processed_tags))
                else:
                    output.append(', '.join(processed_tags))

            if batch_remove_duplicated_tag and not add_confident_as_weight:
                output_str = ','.join(output).split(',')
                output = OrderedDict.fromkeys(map(str.strip, output_str))

            if add_confident_as_weight:
                # postpend the incremented interrogation count
                output.append(str(ict + 1.0))

            output_path.write_text(', '.join(output), encoding='utf-8')

            if batch_output_save_json:
                output_path.with_suffix('.json').write_text(
                    json.dumps([ratings, tags])
                )
        print('all done :)')

    if unload_model_after_running:
        interrogator.unload()

    tag_confidents = {}
    rating_confidents = {}
    tags = []
    for k, v in combined.items():
        if k[:7] == "rating:":
            rating_confidents[k[8:]] = v / nr_of_files
        elif v / nr_of_files >= threshold / 2:
            tag_confidents[k] = v / nr_of_files
            if v / nr_of_files >= threshold:
                tags.append(k)

    return [', '.join(tags), rating_confidents, tag_confidents, '']


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
                        batch_input_glob = utils.preset.component(
                            gr.Textbox,
                            label='Input directory',
                            placeholder='/path/to/images or /path/to/images/**/*'
                        )
                        batch_input_recursive = utils.preset.component(
                            gr.Checkbox,
                            label='Use recursive with glob pattern'
                        )

                        batch_output_dir = utils.preset.component(
                            gr.Textbox,
                            label='Output directory',
                            placeholder='Leave blank to save images to the same path.'
                        )

                        batch_output_filename_format = utils.preset.component(
                            gr.Textbox,
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

                        batch_output_action_on_conflict = utils.preset.component(
                            gr.Dropdown,
                            label='Action on existing caption',
                            value='ignore',
                            choices=[
                                'ignore',
                                'replace',
                                'append',
                                'prepend',
                                'filter'
                            ]
                        )

                        batch_remove_duplicated_tag = utils.preset.component(
                            gr.Checkbox,
                            label='Remove duplicated tag'
                        )

                        batch_output_save_json = utils.preset.component(
                            gr.Checkbox,
                            label='Save with JSON'
                        )

                submit = gr.Button(
                    value='Interrogate',
                    variant='primary'
                )

                info = gr.HTML()

                # preset selector
                with gr.Row(variant='compact'):
                    available_presets = utils.preset.list()
                    selected_preset = gr.Dropdown(
                        label='Preset',
                        choices=available_presets,
                        value=available_presets[0]
                    )

                    save_preset_button = gr.Button(
                        value=ui.save_style_symbol
                    )

                    ui.create_refresh_button(
                        selected_preset,
                        lambda: None,
                        lambda: {'choices': utils.preset.list()},
                        'refresh_preset'
                    )

                # option components

                # interrogator selector
                with gr.Column():
                    with gr.Row(variant='compact'):
                        interrogator_names = utils.refresh_interrogators()
                        interrogator = utils.preset.component(
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
                            lambda: {'choices': utils.refresh_interrogators()},
                            'refresh_interrogator'
                        )

                    unload_all_models = gr.Button(
                        value='Unload all interrogate models'
                    )

                threshold = utils.preset.component(
                    gr.Slider,
                    label='Threshold',
                    minimum=0,
                    maximum=1,
                    value=0.35
                )

                tag_count_threshold = utils.preset.component(
                    gr.Slider,
                    label='Tag count threshold',
                    minimum=0,
                    maximum=1000,
                    value=100,
                    step=1.0
                )

                additional_tags = utils.preset.component(
                    gr.Textbox,
                    label='Additional tags (split by comma)',
                    elem_id='additioanl-tags'
                )

                exclude_tags = utils.preset.component(
                    gr.Textbox,
                    label='Exclude tags (split by comma)',
                    elem_id='exclude-tags'
                )
                with gr.Row(variant='compact'):
                    search_tags = utils.preset.component(
                        gr.Textbox,
                        label='Search tags (split by comma)',
                        elem_id='search-tags'
                    )

                    replace_tags = utils.preset.component(
                        gr.Textbox,
                        label='Replacement tags (split by comma)',
                        elem_id='replace-tags'
                    )
                ingore_case = utils.preset.component(
                    gr.Checkbox,
                    label='Ignore case',
                    value=True
                )
                sort_by_alphabetical_order = utils.preset.component(
                    gr.Checkbox,
                    label='Sort by alphabetical order',
                )
                add_confident_as_weight = utils.preset.component(
                    gr.Checkbox,
                    label='Include confident of tags matches in results'
                )
                replace_underscore = utils.preset.component(
                    gr.Checkbox,
                    label='Use spaces instead of underscore',
                    value=True
                )
                replace_underscore_excludes = utils.preset.component(
                    gr.Textbox,
                    label='Excudes (split by comma)',
                    # kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
                    value='0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
                )
                escape_tag = utils.preset.component(
                    gr.Checkbox,
                    label='Escape brackets',
                )

                unload_model_after_running = utils.preset.component(
                    gr.Checkbox,
                    label='Unload model after running',
                )

                verbose = utils.preset.component(
                    gr.Checkbox,
                    # tooltip: 'Print tag counts per file, no progress bar',
                    label='Verbose'
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

                rating_confidents = gr.Label(
                    label='Rating confidents',
                    elem_id='rating-confidents'
                )
                tag_confidents = gr.Label(
                    label='Tag confidents',
                    elem_id='tag-confidents'
                )

        # register events
        selected_preset.change(
            fn=utils.preset.apply,
            inputs=[selected_preset],
            outputs=[*utils.preset.components, info]
        )

        save_preset_button.click(
            fn=utils.preset.save,
            inputs=[selected_preset, *utils.preset.components],  # values only
            outputs=[info]
        )

        unload_all_models.click(
            fn=unload_interrogators,
            outputs=[info]
        )

        for func in [image.change, submit.click]:
            func(
                fn=wrap_gradio_gpu_call(on_interrogate),
                inputs=[
                    # single process
                    image,

                    # batch process
                    batch_input_glob,
                    batch_input_recursive,
                    batch_output_dir,
                    batch_output_filename_format,
                    batch_output_action_on_conflict,
                    batch_remove_duplicated_tag,
                    batch_output_save_json,

                    # options
                    interrogator,
                    threshold,
                    tag_count_threshold,
                    additional_tags,
                    exclude_tags,
                    search_tags,
                    replace_tags,
                    ingore_case,
                    sort_by_alphabetical_order,
                    add_confident_as_weight,
                    replace_underscore,
                    replace_underscore_excludes,
                    escape_tag,

                    unload_model_after_running,
                    verbose
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
