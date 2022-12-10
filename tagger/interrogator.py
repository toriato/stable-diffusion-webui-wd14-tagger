import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

from typing import Tuple, List, Dict
from io import BytesIO
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download

from modules import shared
from modules.deepbooru import re_special as tag_escape_pattern

# i'm not sure if it's okay to add this file to the repository
from . import dbimutils

# tensorflow maps nearly all vram by default, so we limit this
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# TODO: is safe to set set_memory_growth...?
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

# select a device to process
if shared.cmd_opts.use_cpu == 'all' or shared.cmd_opts.use_cpu == 'interrogate':
    device_name = '/cpu:0'  # 0 by default?
else:
    device_name = '/gpu:0'  # 0 by default?

    if shared.cmd_opts.device_id is not None:
        try:
            device_name = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')


class Interrogator:
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],

        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False
    ) -> Dict[str, float]:

        tags = {
            **{t: 1.0 for t in additional_tags},
            **tags
        }

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

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        pass


class DeepDanbooruInterrogator(Interrogator):
    def __init__(self,  project_path: os.PathLike) -> None:
        self.model = None
        self.project_path = project_path

    def load(self) -> None:
        print(f'Loading DeepDanbooru project from {str(self.project_path)}')

        # deepdanbooru package is not include in web-sd anymore
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c81d440d876dfd2ab3560410f37442ef56fc663
        # TODO: is it okay to install a pip package here?
        from launch import is_installed, run_pip
        if not is_installed('deepdanbooru'):
            package = os.environ.get(
                'DEEPDANBOORU_PACKAGE',
                'git+https://github.com/KichangKim/DeepDanbooru.git@d91a2963bf87c6a770d74894667e9ffa9f6de7ff'
            )

            run_pip(f'install {package} tensorflow-io', 'deepdanbooru')

        with tf.device(device_name):
            import deepdanbooru.project as ddp

            self.model = ddp.load_model_from_project(
                project_path=self.project_path,
                compile_model=False
            )

            self.tags = ddp.load_tags_from_project(
                project_path=self.project_path
            )

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if self.model is None:
            self.load()

        import deepdanbooru.data as ddd

        # convert an image to fit the model
        image_bufs = BytesIO()
        image.save(image_bufs, format='PNG')
        image = ddd.load_image_for_evaluate(
            image_bufs,
            self.model.input_shape[2],
            self.model.input_shape[1]
        )

        image = image.reshape((1, *image.shape[0:3]))

        # evaluate model
        result = self.model.predict(image)

        confidents = result[0].tolist()
        ratings = {}
        tags = {}

        for i, tag in enumerate(self.tags):
            tags[tag] = confidents[i]

        return ratings, tags


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(self) -> None:
        self.model = None

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        repo = 'SmilingWolf/wd-v1-4-vit-tagger'
        model_files = [
            {'filename': 'saved_model.pb', 'subfolder': ''},
            {'filename': 'keras_metadata.pb', 'subfolder': ''},
            {'filename': 'variables.index', 'subfolder': 'variables'},
            {'filename': 'variables.data-00000-of-00001', 'subfolder': 'variables'},
        ]

        print(f'Downloading Waifu Diffusion tagger model files from {repo}')
        model_file_paths = []
        for elem in model_files:
            model_file_paths.append(Path(hf_hub_download(repo, **elem)))

        model_path = model_file_paths[0].parents[0]
        tags_path = Path(hf_hub_download(repo, filename='selected_tags.csv'))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        print(f'Loading Waifu Diffusion tagger model from {str(model_path)}')
        with tf.device(device_name):
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

        self.tags = pd.read_csv(tags_path)

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # convert an image to fit the model
        image.convert('RGB')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = dbimutils.smart_24bit(image)
        image = dbimutils.make_square(image, 448)
        image = dbimutils.smart_resize(image, 448)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # init model
        if self.model is None:
            self.load()

        # evaluate model
        confidents = self.model(image, training=False)

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags
