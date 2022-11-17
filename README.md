WD 1.4 Tagger for [Automatic1111's WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
---
This is a extension version of Waifu Diffusion 1.4 tagger made by MrSmilingWolf.

quote from MrSmilingWolf:

> Based on validation score I'd say this is pretty much production grade.
>
> I've launched a longer training run (50 epochs, ETA: 9 days), mainly to check how much more can be squeezed out of it, but I'm fairly confident this can be plugged into a real inference pipeline already.
>
> I'm also finetuning the ConvNext network, but so far ViT has always coped better with less popular classes, so I'm edging my bets on this one.
OTOH, ensembling seems to give a decent boost in validation metrics, so if we ever want to do that, I'll be ready."

## Disclaimer

I didn't create a model and most of codes are heavily borrowed from original.

Please ask the original author MrSmilingWolf#5991 for questions related to model or additional training.

## Installation

1. *Extensions* -> *Install from URL* -> Enter URL of this repository -> Press *Install* button
   - or clone this repository under `extensions/`
     ```sh
     $ git clone https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git extensions/wd14-tagger
     ```

1. Download the compressed model file from the link below
   1. Join the [SD Training Labs](https://discord.gg/zUDeSwMf2k) server
   1. Click mega.nz link from [this message](https://discord.com/channels/1038249716149928046/1038249717001359402/1041160494150594671)

1. Unzip and move the file to the cloned repository

1. The file structure is will be something like this:
   ```
   extensions/
   └╴wd14-tagger/
     ├╴2022_0000_0899_6549/
     │ └╴selected_tags.csv
     ├╴networks/
     │ └╴ViTB16_11_03_2022_07h05m53s/
     │   └╴ ...
     ├╴scripts/
     │ └╴wd14-tagger.py
     ├╴Utils/
     │ └╴dbimutils.py
     ╵
     ...

   ```

1. Restart by press *Apply and Restart UI* button from *Installed* under *Extensions* tab and have fun :)

## Screenshot
![Screenshot](docs/screenshot.png)

## Copyright

Public domain, only for my parts