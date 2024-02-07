# Linux_GPU_CUDA
A documentation for the installation of Nvidia Driver and Cuda on a Linux VM hosted under proxmox.

# Documentation: Configuration and installation of NVIDIA drivers and CUDA on Ubuntu Server 22.04 LTS 🚀🚀🚀


This documentation covers the steps to install NVIDIA drivers and CUDA on Ubuntu 22.04, including disabling Secure Boot and using Python scripts to verify the installation.


Documentation: Configuration and installation of NVIDIA drivers and CUDA on Ubuntu 22.04
This documentation covers the steps to install NVIDIA drivers and CUDA on Ubuntu 22.04, including disabling Secure Boot and using Python scripts to verify the installation.


## Preparation: Disabling Secure Boot

Secure Boot may need to be disabled in the BIOS as it can cause issues with NVIDIA drivers. The process varies depending on the hardware, but generally you will need to go into the BIOS at system startup and turn off the Secure Boot option.

System update
Update your system to ensure that all packages are up to date:

`sudo apt update`

`sudo apt upgrade`


## Installing the NVIDIA drivers

Check the available drivers
Find out which drivers are recommended for your graphics card:

`ubuntu-drivers devices`


## Automatic installation

Install the recommended driver automatically:

`sudo ubuntu-drivers autoinstall`


## Manuelle Installation

Installieren Sie bei Bedarf einen spezifischen Treiber manuell:

`sudo apt install nvidia-driver-xxx`


## Neustart

Starten Sie das System neu, um die Treiberinstallation abzuschließen:

`sudo reboot`


## Verification

After restarting, you can check the installation with the following command:

`nvidia-smi`



## Blacklisting of Nouveau & AMD drivers

### Create a blacklist file

Create a blacklist file for Nouveau and AMD drivers:


### sudo nano /etc/modprobe.d/blacklist-nouveau.conf


Add the following lines:


`blacklist radeon`

`blacklist nouveau`

`options nouveau modeset=0`




### Updating initramfs

Update initramfs to apply the changes:

`sudo update-initramfs -u`


Restart
Restart the system.

`sudo reboot`



## Installing CUDA


### Prerequisites

Remove previous CUDA versions, if available:

`sudo apt-get --purge remove cuda`


### Add CUDA repository

Add the CUDA repository and update:


`sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub`


`sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"`


`sudo apt-get update`



## CUDA installieren

Installieren Sie CUDA:

`sudo apt-get install -y cuda`


Configure environment variables
Add CUDA to your PATH and configure the library paths:


`echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc`
`echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc`
`source ~/.bashrc`


Checking the CUDA installation
Check the installation of CUDA:

`nvcc --version`



## Testing the GPU & CUDA with Python

TensorFlow test script

Install TensorFlow and execute the following script:

`pip install tensorflow`


```code
[python]

import tensorflow as tf

print("Is Built With CUDA: ", tf.test.is_built_with_cuda())
print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

with tf.device('/device:GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print("Matrix Multiplication Result:\n", c)


```

### Output

```

python3 gpu_testing_tensorflow.py
2024-02-07 19:05:05.395808: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-02-07 19:05:05.395963: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-02-07 19:05:05.422322: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-02-07 19:05:05.483101: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-02-07 19:05:07.054165: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Is Built With CUDA:  True
2024-02-07 19:05:08.242293: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.316966: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.317276: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
Available GPUs:  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
2024-02-07 19:05:08.319155: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.319430: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.319689: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.560707: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.560995: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.561274: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2024-02-07 19:05:08.561470: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3484 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1
Matrix Multiplication Result:
 tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)

```




## PyTorch test script

Install PyTorch and execute the following script:

`pip install torch`


```code
[python]

import torch

print("Is CUDA available: ", torch.cuda.is_available())
print("CUDA Version: ", torch.version.cuda)

a = torch.tensor([1., 2.], device=torch.device('cuda:0'))
b = torch.tensor([3., 4.], device=torch.device('cuda:0'))
c = a + b

print("Addition Result: ", c)

```


## Summary

This documentation provides a comprehensive guide to installing and configuring NVIDIA drivers and CUDA on Ubuntu 22.04, including testing the installation with Python. Note that this guide covers general steps and specific system configurations may vary.



## Fixing the warnings [If warnings occur during the installation of tensorflow or pytorch:]

The warnings state that the scripts and programs added during the installation of TensorFlow, PyTorch or other Python packages are located in /home/philo/.local/bin. Since this directory is not included in your PATH, you cannot run these programs or scripts directly from the terminal.



### Open your shell configuration file:

If you are using bash, this is usually `.bashrc` or `.bash_profile` in your home directory.
Use nano or another text editor to open the file, e.g:

`nano ~/.bashrc`



### Add the directory to the PATH:

Add the following line at the end of the file:


`export PATH="/home/philo/.local/bin:$PATH"`


### Update your shell:

For the changes to take effect, reload the configuration file, either by closing and reopening the terminal or by entering:

`source ~/.bashrc`


### Check the PATH:

Check that the directory has been added correctly:

`echo $PATH`



