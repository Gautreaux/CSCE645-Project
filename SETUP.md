# Setup

_All these things that I've done_

## Intro

Setup for Ubuntu 20.X?

## Install G++

`sudo apt-get install g++`

## Install CUDA

[CUDA Docs link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

__Attempt 1__ (failed)

~2.5 GB

~~~
mkdir ~/Desktop/tmp
cd ~/Desktop/tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
~~~

__Attempt 2__

~3GB

~~~
sudo apt install nvidia-cuda-toolkit
sudo reboot
~~~

Install checks:

`nvcc`
vim
~~~
cd /usr/local/cuda/samples/4_Finance/BlackScholes
sudo make
./BlackScholes
~~~


## Python

Python three should be installed. Developed on python 3.8 so that or later should work.