# GGEMS

website: <https://ggems.fr>

forum: <https://ggems.discourse.group>

## Description

GGEMS is an advanced Monte Carlo simulation platform using CPU and GPU architecture targeting medical applications (imaging and particle therapy). This code is based on the well-validated Geant4 physics model and capable to be executed in both CPU and GPU devices using the OpenCL library.

Features:
* Photon particle tracking
* Multithreaded CPU (Intel, AMD not tested)
* GPU (NVIDIA, Intel, AMD not tested)
* Multi devices (GPUs+CPU) approach
* Single or double float precision for dosimetry application
* External X-ray source
* Navigation in simple box volume or voxelized volume
* Flat or curved detector for CBCT/CT application
* OpenGL visualization
* Dosimetry application (photon)
* TLE (Track Length Estimator) method for dosimetry

## Requirements

GGEMS is a multiplatform application using OpenCL.

OpenCL v1.2 or more must be installed on your system.

Supported operating system:

* Windows
* Linux

Tested compilers:

* gcc 7/8/9 on Linux
* clang from version 9 to 13 on Linux and Windows
* Visual C++ 2022 on Windows

## Installation

To install GGEMS, please follow the procedure here: <https://doc.ggems.fr/v1.2/building_and_installing.html>

## Installation using Python

On Windows or Linux system, GGEMS can be installed using a single python command:

```console
foo@bar~: python setup.py build_ext --generator=Ninja --opengl=ON --examples=ON install
```

By default, the options 'opengl' and 'examples' are set to 'OFF'. In the previous command line, the 'Ninja' generator is activated, a defaut navigator is selected if this option is not used.

# GGEMS using Docker for Linux users

A docker image for GGEMS version 1.2 is available here:

```console
foo@bar~: docker pull ggems/ggems:v1.2.1
```

### Important

To use the docker image on your linux machine, the nvidia driver must be installed as well as the 'nvidia-container' library. To install 'nvidia-container' run the following commands:

```console
foo@bar~: sudo apt install curl
foo@bar~: curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
foo@bar~: sudo apt update
foo@bar~: sudo apt-get install -y nvidia-container-toolkit
```

To test the docker image, run this command:

```console
foo@bar~: docker run -it --rm --gpus all ggems/ggems:v1.2.1 nvidia-smi
Sun Oct 20 14:26:23 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.52.04              Driver Version: 555.52.04      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 980 Ti      Off |   00000000:01:00.0 Off |                  N/A |
| 20%   34C    P8             17W /  260W |       2MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Running CT scanner example in docker image:

```console
foo@bar~: docker run -it --rm --gpus all ggems/ggems:v1.2.1
foo@bar~: cd examples/2_CT_Scanner
foo@bar~: python ct_scanner.py
```

## Copyright

GGEMS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GGEMS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GGEMS.  If not, see <https://www.gnu.org/licenses>.
