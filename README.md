# GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation

<p align="center"> 
<img src="/images/teaser.gif">
</p>

This is an official pytorch implementation of the following paper:

Y. Deng, J. Yang, J. Xiang, and X. Tong, **GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation**, IEEE Computer Vision and Pattern Recognition (CVPR), 2022. (_**Oral Presentation**_)

### [Project page](https://yudeng.github.io/GRAM/) | [Paper](https://arxiv.org/abs/2112.08867) | [Video](https://www.youtube.com/watch?v=hBJWZwl_JCI) ###

Abstract: _3D-aware image generative modeling aims to generate 3D-consistent images with explicitly controllable camera poses. Recent works have shown promising results by training neural radiance field (NeRF) generators on unstructured 2D images, but still cannot generate highly-realistic images with fine details. A critical reason is that the high memory and computation cost of volumetric representation learning greatly restricts the number of point samples for radiance integration during training. Deficient sampling not only limits the expressive power of the generator to handle fine details but also impedes effective GAN training due to the noise caused by unstable Monte Carlo sampling. We propose a novel approach that regulates point sampling and radiance field learning on 2D manifolds, embodied as a set of learned implicit surfaces in the 3D volume. For each viewing ray, we calculate ray-surface intersections and accumulate their radiance generated by the network. By training and rendering such radiance manifolds, our generator can produce high quality images with realistic fine details and strong visual 3D consistency._

## Requirements
- Currently only Linux is supported.
- 64-bit Python 3.6 installation or newer. We recommend using Anaconda3.
- One or more high-end NVIDIA GPUs, NVIDIA drivers, and CUDA toolkit 10.1 or newer. We recommend using 8 Tesla V100 GPUs with 32 GB memory for training to reproduce the results in the paper. 

## Installation
Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/microsoft/GRAM.git
cd GRAM
conda env create -f environment.yml
source activate gram
```

Alternatively, we provide a [Dockerfile](https://github.com/microsoft/GRAM/blob/main/Dockerfile) to build an image with the required dependencies.

## Pre-trained models
Checkpoints for pre-trained models used in our paper (default settings) are as follows.

|Dataset|Config|Resolution|Training iterations|Batchsize|FID 20k|KID 20k (x100)|Download|
|:----:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|FFHQ | FFHQ_default |256x256 |  150k | 32 | 14.5 | 0.65 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/FFHQ_default) |
|Cats | CATS_default |256x256 |  80k | 16 |14.6 | 0.75 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/CATS_default) |
|CARLA | CARLA_default |128x128 |  70k | 32 | 26.3 | 1.15 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/CARLA_default) |

## Generating multi-view images with pre-trained models
Run the following script to render multi-view images of generated subjects using a pre-trained model:
```
# face images are generated by default (FFHQ_default)
python render_multiview_images.py

# custom setting for image generation
python render_multiview_images.py --config=<CONFIG_NAME> --generator_file=<GENERATOR_PATH.pth> --output_dir=<OUTPUT_FOLDER> --seeds=0,1,2
```
By default, the script generates images with watermarks. Use --no_watermark argument to remove them.

## Training a model from scratch
### Data preparation
- FFHQ: Download the [original 1024x1024 images](https://github.com/NVlabs/ffhq-dataset). We additionally provide [detected 5 facial landmarks (google drive)](https://drive.google.com/file/d/1bOefjWzNGzjJ65J5WT9V0QrsrNhKjjCb/view?usp=sharing) for image preprocessing and [face poses (google drive)](https://drive.google.com/file/d/1kb-PeNhOEmN1Gs8e0xF3aLjsjHe01sVb/view?usp=sharing) estimated by [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for training. Download all files and organize them as follows:
```
GRAM/
│
└─── raw_data/
    |
    └─── ffhq/
	│
	└─── *.png   # original 1024x1024 images
	│
        └─── lm5p/   # detected 5 facial landmarks
	|   |
        |   └─── *.txt
	|
	└─── poses/  # estimated face poses
	    |
	    └─── *.mat    
```
- Cats: Download the original cat images and provided landmarks using this [link](https://archive.org/details/CAT_DATASET) and organize all files as follows:
```
GRAM/
│
└─── raw_data/
    |
    └─── cats/
	│
	└─── *.jpg   # original images
	│
        └─── *.jpg.cat   # provided landmarks
```
- CARLA: Download the original images and poses from [GRAF](https://github.com/autonomousvision/graf/tree/main/data) and organize all files as follows:
```
GRAM/
│
└─── raw_data/
    |
    └─── carla/
	│
	└─── *.png   # original images
	│
        └─── poses/  # provided poses
	    |
	    └─── *_extrinsics.npy
```
Finally, run the following script for data preprocessing:
```
python preprocess_dataset.py --raw_dataset_path=./raw_data/<CATEGORY> --cate=<CATEGORY>
```
It will align all images and save them with the estimated/provided poses into ./datasets for the later training process.

### Training networks
Run the following script to train a generator from scratch using the preprocessed data:
```
python train.py --config=<CONFIG_NAME> --output_dir=<OUTPUT_FOLDER>
```
The code will automatically detect all available GPUs and use DDP training. You can use the default configs provided in the configs.py or add your own config. By default, we use batch split suggested by [pi-GAN](https://github.com/marcoamonteiro/pi-GAN) to increase the effective batchsize during training.

The following table lists training times for different configs using 8 NVIDIA Tesla V100 GPUs (32GB memory):
|Config|Resolution|Training iterations|Batchsize|Times|
|:----:|:-----------:|:-----------:|:-----------:|:-----------:|
|FFHQ_default | 256x256 | 150k | 32 | 12d 4h |
|CATS_default | 256x256 | 80k | 16 | 4d 6h |
|CARLA_default | 128x128 | 70k | 32 | 3d 15h |

Training GRAM under 256x256 image resolution requires around 30GB memory for a typical forward-backward cycle with a batchsize of 1 using Pytorch Automatic Mixed Precision. To enable training using GPUs with limited memory, we provide an alternative way using patch-level forward and backward process (see [here](https://github.com/microsoft/GRAM/blob/main/images/patch_process.pdf) for a detailed explanation):
```
python train.py --config=<CONFIG_NAME> --output_dir=<OUTPUT_FOLDER> --patch_split=<NUMBER_OF_PATCHES> 
```
Currently we support a patch split of a power of 2 (e.g. 2, 4, 8, ...). It will effectively reduce the memory cost with a slight increase of the training time.

## Evaluation
Run the following script for FID&KID calculation:
```
python fid_evaluation.py --no_watermark --config=<CONFIG_NAME> --generator_file=<GENERATOR_PATH.pth> --output_dir=<OUTPUT_FOLDER>
```
By default, 8000 real images and 1000 generated images from EMA model are used for evaluation. You can adjust the number of images according to your own needs. 

## Contact
If you have any questions, please contact Yu Deng (dengyu2008@hotmail.com) and Jiaolong Yang (jiaoyan@microsoft.com)

## License

Copyright &copy; Microsoft Corporation.

Licensed under the [Microsoft Research license](https://github.com/microsoft/GRAM/blob/master/files/GRAM-Microsoft%20Research%20License%20Agreement.pdf).

## Citation

Please cite the following paper if this work helps your research:

    @inproceedings{deng2022gram,
		title={GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation},
    	author={Deng, Yu and Yang, Jiaolong and Xiang, Jianfeng and Tong, Xin},
	    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	    year={2022}
	}

## Acknowledgements
We thank Harry Shum for the fruitful advice and discussion to improve the paper. This implementation takes [pi-GAN](https://github.com/marcoamonteiro/pi-GAN) as a reference. We thank the authors for their excellent work. 
