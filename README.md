## GGEMS

website: <https://ggems.fr>

forum: <https://forum.ggems.fr>

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

# GGEMS using Docker for Linux users

A docker image for GGEMS version 1.2 is available here:

```console
foo@bar~: docker pull ggems/ggems:v1.2
```

### Important

To use the docker image on your linux machine, the nvidia driver must be installed as well as the 'nvidia-container' library. To install 'nvidia-container' run the following commands:

```console
foo@bar~: sudo apt install curl
foo@bar~: curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
foo@bar~: distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
foo@bar~: curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
foo@bar~: sudo apt update
foo@bar~: sudo apt install nvidia-container-runtime
```

To test the docker image, run this command:

```console
foo@bar~: docker run -it --rm --gpus all ggems/ggems:v1.2 nvidia-smi
Thu Jun  9 16:39:54 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.39.01    Driver Version: 510.39.01    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:17:00.0 Off |                  N/A |
| 30%   27C    P8    N/A /  75W |     11MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Quadro P400         On   | 00000000:65:00.0  On |                  N/A |
| 34%   35C    P8    N/A /  N/A |    280MiB /  2048MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
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
