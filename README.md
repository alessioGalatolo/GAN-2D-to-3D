# Unsupervised 3D shape retrieval from pre-trained GANs
Replication of [GAN2Shape](https://github.com/XingangPan/GAN2Shape).

## Results
| Cats (Ellipsoid) | Cats (Smoothed Box) | Cars (Ellipsoid) | Cars (Smoothed Box) | Faces (Ellipsoid) | Faces (Confidence) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ![Cat 0 - Ellipsoid](docs/Cat%20-%20Ellipsoid/plotly__im_0.gif) | ![Cat 0 - Smoothed Box](docs/Cat%20-%20Smoothed%20box/plotly__im_0.gif) | ![Car 0 - Ellipsoid](docs/Car%20-%20Ellipsoid/plotly__im_0.gif) | ![Car 0 - Smoothed Box](docs/Car%20-%20Smoothed%20box/plotly__im_0.gif) | ![Face 0 - Ellipsoid](docs/Face%20-%20Ellipsoid/plotly__im_0.gif) | ![Face 0 - Confidence](docs/Face%20-%20Confidence/plotly__im_0.gif) | 
| ![Cat 1 - Ellipsoid](docs/Cat%20-%20Ellipsoid/plotly__im_1.gif) | ![Cat 1 - Smoothed Box](docs/Cat%20-%20Smoothed%20box/plotly__im_1.gif) | ![Car 4 - Ellipsoid](docs/Car%20-%20Ellipsoid/plotly__im_4.gif) | ![Car 4 - Smoothed Box](docs/Car%20-%20Smoothed%20box/plotly__im_4.gif) | ![Face 1 - Ellipsoid](docs/Face%20-%20Ellipsoid/plotly__im_1.gif) | ![Face 1 - Confidence](docs/Face%20-%20Confidence/plotly__im_1.gif) |
| ![Cat 2 - Ellipsoid](docs/Cat%20-%20Ellipsoid/plotly__im_2.gif) | ![Cat 2 - Smoothed Box](docs/Cat%20-%20Smoothed%20box/plotly__im_2.gif) | ![Car 5 - Ellipsoid](docs/Car%20-%20Ellipsoid/plotly__im_5.gif) | ![Car 5 - Smoothed Box](docs/Car%20-%20Smoothed%20box/plotly__im_5.gif) | ![Face 2 - Ellipsoid](docs/Face%20-%20Ellipsoid/plotly__im_2.gif) | ![Face 2 - Confidence](docs/Face%20-%20Confidence/plotly__im_2.gif) |

The results are also available interactively at [alessiogalatolo.github.io/GAN-2D-to-3D/](https://alessiogalatolo.github.io/GAN-2D-to-3D/).
## Install instructions for Ubuntu 18.04.6 LTS with CUDA 10+ compatible GPU
```
sudo apt update
sudo apt install build-essential
```
```
# CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```
```
# Install Conda
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh
```
*Close and reopen terminal*
```
conda create --name 3D-GAN pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda activate 3D-GAN
```
```
# Install neural renderer
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer
python setup.py install
```
```
# Install all other dependencies
conda install numpy pandas distributed ipython lmdb matplotlib Pillow scipy tqdm scikit-image
conda install PyYAML -c conda-forge
conda install wandb -c conda-forge  # optional
conda install -y -c anaconda ipykernel ipython_genutils  # only needed if running on notebook
pip install plotly  # only needed for evaluating results
```

## Install instruction for Windows
The procedure is the same except for the installing the neural renderer that will not work out of the box on windows.
Please see our guide [here](https://github.com/alessioGalatolo/GAN-2D-to-3D/tree/nr-windows-instructions) for a procedure that might work for installing the neural renderer on Windows.

## How to run:
First you need to download the various datasets:
```sh
python download_data.py
python main.py --save-ckpts
```
To evaluate the results:
```
python evaluate_results.py
```
# Acknowledgments
Part of this code is borrowed from [Unsup3d](https://github.com/elliottwu/unsup3d), [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch), [Semseg](https://github.com/hszhao/semseg) and [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch).

# Todo-list
- ~~Add saving of models/checkpoints~~
- ~~Add support for batches:~~
  - ~~Image batches (Right now we are training for 1 image at a time, inefficient)~~
  - ~~Fix support for 100 projected samples of the same image (and implement batches for these too)~~
- ~~Fix ellipsoid/mask_net~~
- ~~Add proper logging~~
- ~~Make good looking graphs~~
- ~~Add 3D depth plots (maybe save pickles of tensors so we can decide how to visualize it later)~~
- Add plots/anims for multiple viewpoints and light? 
  (I think we need to construct a function which samples from an "interpolation" between two viewpoints and not randomly)
- ~~Refactor code - there is currently a lot of unnecessary repetition~~
- ~~Experiment with new priors (improve box + maybe something else?)?~~
- Experiment with loss/regularization?
- Training
  - ~~LSUN Car (replication)~~
  - LSUN Horse (replication)
  - LSUN Bus (extension)
  - LSUN Sheep (extension)
  - Roboflow’s Fruit dataset (extension)  



>Roboflow’s Fruit dataset (extension)  

Not sure we can do this (check stylegan is trained for this)
