# Unsupervised 3D shape retrieval from pre-trained GANs
Replication of [GAN2Shape](https://github.com/XingangPan/GAN2Shape). We participated with this code in the [Machine Learning Reproducibility Challenge 2021](https://paperswithcode.com/rc2021) and our paper for accepted for publication at [ReScience C](https://rescience.github.io/read) journal, our report is also temporarily available in the [OpenReview forum](https://openreview.net/forum?id=B8mxkTzX2RY).

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
