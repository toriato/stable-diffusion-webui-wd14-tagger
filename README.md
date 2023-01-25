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
   - #### *MrSmilingWolf's model (a.k.a. Waifu Diffusion 1.4 tagger)*
      Downloads automatically from the [HuggingFace repository](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger) the first time you run it.

      Please ask the original author MrSmilingWolf#5991 for questions related to model or additional training.

      ##### ViT vs Convnext
      > To make it clear: the ViT model is the one used to tag images for WD 1.4. That's why the repo was originally called like that. This one has been trained on the same data and tags, but has got no other relation to WD 1.4, aside from stemming from the same coordination effort. They were trained in parallel, and the best one at the time was selected for WD 1.4

      > This particular model was trained later and might actually be slightly better than the ViT one. Difference is in the noise range tho 

      — [SmilingWolf](https://github.com/SmilingWolf) from [this thread](https://discord.com/channels/930499730843250783/1052283314997837955) in the [東方Project AI server](https://discord.com/invite/touhouai) 

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

1. Start or restart the WebUI.
   - or you can press refresh button after *Interrogator* dropdown box.


## Model comparison

* Used image: [hecattaart's artwork](https://vk.com/hecattaart?w=wall-89063929_3767)
* Threshold: `0.5`

### DeepDanbooru
Used the same image as the one used in the Screenshot item

#### [`deepdanbooru-v3-20211112-sgd-e28`](https://github.com/KichangKim/DeepDanbooru/releases/tag/v3-20211112-sgd-e28)
```
1girl, animal ears, cat ears, cat tail, clothes writing, full body, rating:safe, shiba inu, shirt, shoes, simple background, sneakers, socks, solo, standing, t-shirt, tail, white background, white shirt
```

#### [`deepdanbooru-v4-20200814-sgd-e30`](https://github.com/KichangKim/DeepDanbooru/releases/tag/v4-20200814-sgd-e30)
```
1girl, animal, animal ears, bottomless, clothes writing, full body, rating:safe, shirt, shoes, short sleeves, sneakers, solo, standing, t-shirt, tail, white background, white shirt
```

#### `e621-v3-20221117-sgd-e32`
```
anthro, bottomwear, clothing, footwear, fur, hi res, mammal, shirt, shoes, shorts, simple background, sneakers, socks, solo, standing, text on clothing, text on topwear, topwear, white background
```

### Waifu Diffusion Tagger

#### [`wd14-vit`](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger)
```
1boy, animal ears, dog, furry, leg hair, male focus, shirt, shoes, simple background, socks, solo, tail, white background
```

#### [`wd14-convnext`](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger)
```
full body, furry, shirt, shoes, simple background, socks, solo, tail, white background
```

#### [`wd14-vit-v2`](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)
```
1boy, animal ears, cat, furry, male focus, shirt, shoes, simple background, socks, solo, tail, white background
```

#### [`wd14-convnext-v2`](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)
```
animal focus, clothes writing, earrings, full body, meme, shirt, shoes, simple background, socks, solo, sweat, tail, white background, white shirt
```

#### [`wd14-swinv2-v2`](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2)
```
1boy, arm hair, black footwear, cat, dirty, full body, furry, leg hair, male focus, shirt, shoes, simple background, socks, solo, standing, tail, white background, white shirt
```

## Screenshot
![Screenshot](docs/screenshot.png)

Artwork made by [hecattaart](https://vk.com/hecattaart?w=wall-89063929_3767)

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)