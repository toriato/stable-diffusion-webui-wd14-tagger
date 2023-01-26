What is Waifu Diffison 1.4 Tagger?
---

Image to text model created and maintained by [MrSmilingWolf](https://huggingface.co/SmilingWolf), which was used to train Waifu Diffusion.

Please ask the original author `MrSmilingWolf#5991` for questions related to model or additional training.

## SwinV2 vs Convnext vs ViT
> It's got characters now, the HF space has been updated too. Model of choice for classification is SwinV2 now. ConvNext was used to extract features because SwinV2 is a bit of a pain cuz it is twice as slow and more memory intensive

— [this message](https://discord.com/channels/930499730843250783/930499731451428926/1066830289382408285) from the [東方Project AI discord server](https://discord.com/invite/touhouai)

> To make it clear: the ViT model is the one used to tag images for WD 1.4. That's why the repo was originally called like that. This one has been trained on the same data and tags, but has got no other relation to WD 1.4, aside from stemming from the same coordination effort. They were trained in parallel, and the best one at the time was selected for WD 1.4
>
> This particular model was trained later and might actually be slightly better than the ViT one. Difference is in the noise range tho 

— [this thread](https://discord.com/channels/930499730843250783/1052283314997837955) from the [東方Project AI discord server](https://discord.com/invite/touhouai)

## Performance
> I stack them together and get a 1.1GB model with higher validation metrics than the three separated, so they each do their own thing and averaging the predictions sorta helps covering for each models failures. I suppose.
> As for my impression for each model:
> - SwinV2: a memory and GPU hog. Best metrics of the bunch, my model is compatible with timm weights (so it can be used on PyTorch if somebody ports it) but slooow. Good for a few predictions, would reconsider for massive tagging jobs if you're pressed for time
> - ConvNext: nice perfs, good metrics. A sweet spot. The 1024 final embedding size provides ample space for training the Dense layer on other datasets, like E621.
> - ViT: fastest of the bunch, at least on TPU, probably on GPU too? Slightly less then stellar metrics when compared with the other two. Onnxruntime and Tensorflow keep adding optimizations for Transformer models so that's good too.

— [this message](https://discord.com/channels/930499730843250783/930499731451428926/1066833768112996384) from the [東方Project AI discord server](https://discord.com/invite/touhouai)

## Links
- [MrSmilingWolf's HuggingFace profile](https://huggingface.co/SmilingWolf)
- [MrSmilingWolf's GitHub profile](https://github.com/SmilingWolf)
