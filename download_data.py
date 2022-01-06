import shutil
import tarfile
import urllib.request
from glob import glob
import os

print("Downloading data...")
urllib.request.urlretrieve('https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/data.tar.gz', "data.tar.gz")
urllib.request.urlretrieve('https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x00', 'x00')
urllib.request.urlretrieve('https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x01', 'x01')
urllib.request.urlretrieve('https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x02', 'x02')
urllib.request.urlretrieve('https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x03', 'x03')

print("Data downloaded, saving it...")
shutil.copyfile('x00', 'x0.tar.gz')
shutil.copyfile('x01', 'x0.tar.gz')
shutil.copyfile('x02', 'x0.tar.gz')
shutil.copyfile('x03', 'x0.tar.gz')
print("Extracting data...")
with tarfile.open('data.tar.gz', mode="r:gz") as data_tar:
    data_tar.extractall()
with tarfile.open('x0.tar.gz', mode="r:gz") as checkpoints_tar:
    checkpoints_tar.extractall()

os.remove('data.tar.gz')
os.remove('x0.tar.gz')
os.remove('x00')
os.remove('x01')
os.remove('x02')
os.remove('x03')
for category in os.walk("data"):
    filelists = glob(f'data/{category}/list*')
    if len(filelists) > 1:
        for filelist in filelists:
            shutil.copyfile(filelist, f"data/{category}/list.txt")
