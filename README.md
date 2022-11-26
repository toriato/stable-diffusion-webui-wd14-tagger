Tagger for [Automatic1111's WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
---
Interrogate booru style tags for single or multiple image files using various models, such as DeepDanbooru.

[한국어를 사용하시나요? 여기에 한국어 설명서가 있습니다!](README.ko.md)

## Disclaimer
I didn't make any models, and most of the code was heavily borrowed from the [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru) and MrSmillingWolf's tagger.

## Installation
1. *Extensions* -> *Install from URL* -> Enter URL of this repository -> Press *Install* button
   - or clone this repository under `extensions/`
      ```sh
      $ git clone https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git extensions/tagger
      ```

1. Add interrogate model
   - #### *DeepDanbooru*
      1. Various model files can be found below.
         - [DeepDanbooru models](https://github.com/KichangKim/DeepDanbooru/releases)
         - [e621 model by 🐾Zack🐾#1984](https://discord.gg/BDFpq9Yb7K)
            *(link contains NSFW contents!)*

      1. Move the project folder containing the model and config to `models/deepdanbooru`

      1. The file structure should look like:
         ```
         models/
         └╴deepdanbooru/
           ├╴deepdanbooru-v3-20211112-sgd-e28/
           │ ├╴project.json
           │ └╴...
           │
           ├╴deepdanbooru-v4-20200814-sgd-e30/
           │ ├╴project.json
           │ └╴...
           │
           ├╴e621-v3-20221117-sgd-e32/
           │ ├╴project.json
           │ └╴...
           │
           ...
         ```
    
   - #### *MrSmilingWolf's model (a.k.a. Waifu Diffusion 1.4 tagger)*
      Please ask the original author MrSmilingWolf#5991 for questions related to model or additional training.
      
      Quote from MrSmilingWolf:

      > Based on validation score I'd say this is pretty much production grade.
      >
      > I've launched a longer training run (50 epochs, ETA: 9 days), mainly to check how much more can be squeezed out of it, but I'm fairly confident this can be plugged into a real inference pipeline already.
      >
      > I'm also finetuning the ConvNext network, but so far ViT has always coped better with less popular classes, so I'm edging my bets on this one.
      OTOH, ensembling seems to give a decent boost in validation metrics, so if we ever want to do that, I'll be ready."
      1. Download the compressed model file.
         1. Join the [SD Training Labs](https://discord.gg/zUDeSwMf2k) discord server
         1. Click mega.nz link from [this message](https://discord.com/channels/1038249716149928046/1038249717001359402/1041160494150594671)

      1. Unzip and move all files to the cloned repository.

      1. The file structure should look like:
         ```
         extensions/
         └╴wd14-tagger/
           ├╴2022_0000_0899_6549/
           │ └╴selected_tags.csv
           │
           ├╴networks/
           │ └╴ViTB16_11_03_2022_07h05m53s/
           │   └╴ ...
           │
           ├╴scripts/
           │ └╴tagger.py
           │
           ...
         ```
1. Start or restart the WebUI.
   - or you can press refresh button after *Interrogator* dropdown box.


## Screenshot
![Screenshot](docs/screenshot.png)

Artwork made by [hecattaart](https://vk.com/hecattaart?w=wall-89063929_3767)

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)