import shutil
import tarfile
import urllib.request
from glob import glob
import os

print("Downloading data...", flush=True)
urllib.request.urlretrieve('https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/data.tar.gz', "data.tar.gz")
parts = [f'x0{i}' for i in range(4)]
for part in parts:
    urllib.request.urlretrieve(f'https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/{part}', part)

print("Data downloaded, saving it...", flush=True)
with open('x0.tar.gz', 'wb') as dest:
    for file in parts:
        with open(file, 'rb') as src:
            shutil.copyfileobj(src, dest)

print("Extracting data...", flush=True)
with tarfile.open('data.tar.gz', mode="r:gz") as data_tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(data_tar)
with tarfile.open('x0.tar.gz', mode="r:gz") as checkpoints_tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(checkpoints_tar)

print("Removing temp files...")
os.remove('data.tar.gz')
os.remove('x0.tar.gz')
for part in parts:
    os.remove(part)

for _, dirs, _ in os.walk("data"):
    for category in dirs:
        filelists = glob(f'data/{category}/list*')
        if len(filelists) > 1:
            with open(f"data/{category}/list.txt", 'wb') as dest:
                for filelist in filelists:
                    try:
                        with open(filelist, 'rb') as src:
                            shutil.copyfileobj(src, dest)
                    except shutil.SameFileError:
                        pass

os.rename("data/celeba", "data/face")
