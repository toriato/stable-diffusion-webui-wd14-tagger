from PIL import Image, ImageFile

from modules import script_callbacks
from tagger.api import on_app_started
from tagger.ui import on_ui_tabs


# if you do not initialize the Image object
# Image.registered_extensions() returns only PNG
Image.init()

# PIL spits errors when loading a truncated image by default
# https://pillow.readthedocs.io/en/stable/reference/ImageFile.html#PIL.ImageFile.LOAD_TRUNCATED_IMAGES
ImageFile.LOAD_TRUNCATED_IMAGES = True


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
