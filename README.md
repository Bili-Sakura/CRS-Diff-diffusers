# CRS-Diff: Controllable Remote Sensing Image Generation with Diffusion Model
### [Paper (ArXiv)](https://arxiv.org/abs/2403.11614) 


<div align=center>
<img src="img/figure_1.png" height="100%" width="100%"/>
</div>

## TODO

- [x] Release inference code.
- [x] Release pretrained models.
- [x] Release Gradio UI.
- [ ] Release training code

## Environment

```bash
conda env create -f environment.yaml
conda activate csrldm
```
You can download pre-trained models [last.ckpt](https://huggingface.co/Sonetto702/AeroGen/tree/main) and put it to `./ckpt/` folder.

### Testing

You can run the code to start the gradio interface by:
```bash
python src/test/test.py
```
The demonstration effects of the project are as follows:
<div align=center>
<img src="img/figure_2.png" height="100%" width="100%"/>
</div>

You can also use the following code to generate images more quickly
```bash
python src/test/inference.py
```
Some of the results are shown below：
<div align=center>
<img src="img/figure_3.png" height="100%" width="100%"/>
</div>

## Acknowledgments:

This repo is built upon [ControlNet](https://github.com/lllyasviel/ControlNet/tree/main) and [Uni-ControlNet](https://github.com/ShihaoZhaoZSH/Uni-ControlNet/tree/main). 
Some of the functional implementations of remote sensing imagery refer to: [GeoSeg](https://github.com/WangLibo1995/GeoSeg),[Txt2Img-MHN](https://github.com/YonghaoXu/Txt2Img-MHN?tab=readme-ov-file#gen) and [SGCN](https://github.com/tist0bsc/SGCN). Sincere thanks to their excellent work!

## Diffusers-first workflow

This repository now includes a diffusers-style implementation in `src/diffusers_impl`:

- `src/diffusers_impl/pipeline_crs.py`: native `DiffusionPipeline` inference entrypoint
- `src/diffusers_impl/train.py`: native diffusers/accelerate training loop
- `src/diffusers_impl/convert_checkpoint.py`: legacy checkpoint → diffusers `save_pretrained` converter
- `src/diffusers_impl/inference.py`: CLI inference script for the new pipeline

### Optional in-repo external diffusers clone

If you want to use a local clone of huggingface/diffusers as the external library, clone it under:

`external/diffusers`

The new implementation auto-detects `external/diffusers/src` and prioritizes it.

### Convert legacy checkpoints

```bash
python -m src.diffusers_impl.convert_checkpoint \
  --config ./configs/crs.yaml \
  --checkpoint /path/to/legacy.ckpt \
  --output_dir ./converted_crs_diffusers
```

### Diffusers-style inference

```bash
python -m src.diffusers_impl.inference \
  --checkpoint /path/to/legacy.ckpt \
  --prompt "remote sensing image" \
  --local_control /path/to/local_control.npy \
  --global_control /path/to/global_control.npy
```

### Diffusers-style training

```bash
python -m src.diffusers_impl.train \
  --checkpoint /path/to/legacy.ckpt \
  --train_data /path/to/train_data.pt \
  --output_dir ./outputs_diffusers_train
```

## Citation
```
@article{tang2024crs,
  title={Crs-diff: Controllable remote sensing image generation with diffusion model},
  author={Tang, Datao and Cao, Xiangyong and Hou, Xingsong and Jiang, Zhongyuan and Liu, Junmin and Meng, Deyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
