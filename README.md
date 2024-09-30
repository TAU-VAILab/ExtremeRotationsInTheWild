# Extreme Rotation Estimation in the Wild
[Hana Bezalel]()¹,
[Dotan Ankri]()¹, 
[Ruojin Cai](https://www.cs.cornell.edu/~ruojin/)², 
[Hadar Averbuch-Elor](https://www.elor.sites.tau.ac.il/)¹²

¹[Tel Aviv University](https://english.tau.ac.il/),
²[Cornell University](https://www.cornell.edu/)

**[project page](https://tau-vailab.github.io/ExtremeRotationsInTheWild) | [paper]()**

This repository contains a PyTorch implementation of the paper:
Extreme Rotation Estimation in the Wild


## Introduction
We present a technique and benchmark dataset for estimating the relative 3D orientation between a pair of In
ternet images captured in an extreme setting, where the images have limited or non-overlapping field of views.
Prior work targeting extreme rotation estimation assume constrained 3D environments and emulate perspective images
by cropping regions from panoramic views. However, real images captured in the wild are highly diverse, exhibiting
variation in both appearance and camera intrinsics. In this work, we propose a Transformer-based method
for estimating relative rotations in extreme real-world settings, and contribute the ExtremeLandmarkPairs dataset,
assembled from scene-level Internet photo collections. Our evaluation demonstrates that our approach succeeds in
estimating the relative rotations in a wide variety of extreme-view Internet image pairs, outperforming various baselines,
including dedicated rotation estimation techniques and contemporary 3D reconstruction methods. We will release our
data, code, and trained models.
<p align="center">
<img src="webpage_assets/overview_new.JPG" width="90%"/>  
</p>
</br>

## Dependencies 
    conda env create -f ./tools/environment.yml
    conda activate rota_cuda_
</br>

## Getting the repo
    git clone https://github.com/TAU-VAILab/ExtremeRotationsInTheWild.git
    cd ExtremeRotationsInTheWild
</br>

## Dataset

Perspective images are randomly sampled from panoramas with a resolution of 256 × 256. 
To avoid generating textureless images that focus on the ceiling/sky or the floor, we limit the range over pitch angles to [−30◦, 30◦].

Download [StreetLearn](https://sites.google.com/view/streetlearn/dataset) datasets to obtain the full panoramas.

Metadata files about the training and test image pairs are available in the following google drive: [link]().
Download the `metadata.zip` file, unzip it and put it under the project root directory.

we used this script [`PanoBasic/pano2perspective_script.m`](https://github.com/RuojinCai/PanoBasic.git) that extracts perspective images from an input panorama. 
Before running it , you need to modify the path to the datasets and metadata files in the script.

## Pretrained Model 

Pretrained models are be available in the following link: [link]().
To use the pretrained models, download the `pretrained.zip` file, unzip it and put it under the project root directory.

#### Testing the pretrained model:
The following commands test the performance of the pre-trained models in the rotation estimation task.
The commands output the median geodesic error, and the percentage of pairs with a relative rotation error under 15◦ and 30◦ for different levels of overlap on the test set.
```bash
# Usage:
# python test.py <config> --pretrained <checkpoint_filename>

#sELP
python test.py configs/ELP/streetlearn_cv_distribution_selp.yaml \
    --pretrained pretrained/final_model.pt
#wELP
python test.py configs/ELP/streetlearn_cv_distribution_welp.yaml \
    --pretrained pretrained/final_model.pt

```
## Training

```bash
# Usage:
# python train.py <config>

#90 fov
python train.py configs/90_fov/streetlearn_cv_distribution_90_fov_overlap.yaml
python train.py configs/90_fov/streetlearn_cv_distribution_90_fov.yaml --resume --pretrained <checkpoint_filename>

#d_fov
python train.py configs/d_fov/streetlearn_cv_distribution_d_fov_overlap.yaml --resume --pretrained <checkpoint_filename>
python train.py configs/d_fov/streetlearn_cv_distribution_d_fov.yaml --resume --pretrained <checkpoint_filename>

#d_im
python train.py configs/d_im/streetlearn_cv_distribution_d_im.yaml --resume --pretrained <checkpoint_filename>

#ELP
python train.py configs/ELP/streetlearn_cv_distribution_welp.yaml --resume --pretrained <checkpoint_filename>
```
## Cite

## Acknowledgment
The dataset is constructed from Internet image pairs from the [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/),
[Cambridge Landmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3/),
[MegaScenes](https://megascenes.github.io/) datasets.

The code implementation is based on the official repository of [Extreme Rotation](https://github.com/RuojinCai/ExtremeRotation_code)
Code from following works has also been used: [LoFTR](https://zju3dv.github.io/loftr/),[Dust3R](https://github.com/naver/dust3r),
[semantic-segmentation](https://github.com/sithu31296/semantic-segmentation),[DinoV2](https://github.com/facebookresearch/dinov2).
.


