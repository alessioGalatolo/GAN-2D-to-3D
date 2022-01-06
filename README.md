# Instructions for installing the neural renderer on Windows
Install cuda 9.2, pytorch 1.2

Install Visual Studio Community Edition 2017 (NOT 2022!). When installing, go to individual components and deselect VC++ 2017 and select VC++ 2015.

Reboot

Go to C:\Users\<user>\anaconda3\envs\<your env>\Lib\site-packages\torch\include\c10\util and remove flat_hash_map.h and replace it with the one above

Inside the project do:
```sh
git clone https://github.com/adambielski/neural_renderer.git
```
(NOTE: this is a different repo from the original one
go to neural_renderer/neural_renderer/cuda and replace rasterize_cuda_kernel.cu with the one above

Copy these files: ```rc.exe rcdll.dll```
From

C:\Program Files (x86)\Windows Kits\8.x\bin\x86

To

C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin

inside the project:
```sh
cd neural_renderer
python setup.py install
```

Enjoy!

(Hopefully this solves your issue)
