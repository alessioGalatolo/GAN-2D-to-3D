# Todo-list
- Add saving of models/checkpoints
- Fix ellipsoid/mask_net
- Add proper logging/make good looking graphs
- Add 3D depth plots (maybe save pickles of tensors so we can decide how to visualize it later)
- Add plots/anims for multiple viewpoints and light? 
  (I think we need to construct a function whichs samples from an "interpolation" between two viewpoints and not randomly)
- Refactor code - there is currently a lot of unnecessary repitition
- Experiment with new priors (improve box + maybe something else?)?
- Experiment with loss/regularization?
- Training
  - LSUN Car (replication)
  - LSUN Horse (replication)
  - LSUN Bus (extention)
  - LSUN Sheep (extention)
  - Roboflow’s Fruit dataset (extention)

# Unsupervised 3D shape retrieval from pre-trained GANs
Replication of [GAN2Shape](https://github.com/XingangPan/GAN2Shape).

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
conda install wandb PyYAML -c conda-forge
```

## How to run:
First you need to download the various datasets:
```sh
./download.sh
```
Then, simply run the main.py
