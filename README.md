# GGEMS

website: <https://ggems.fr>

forum: <https://ggems.discourse.group>

## Description

GGEMS is an advanced Monte Carlo simulation platform using the OpenCL library managing CPU and GPU architecture. GGEMS is fully developed in C++ and accessible via Python command line. Well-validated Geant4 physic models are used in GGEMS and implemented using OpenCL. The aim of GGEMS is to provide a fast simulation platform for imaging application (CT/CBCT for moment) and particle therapy. To favor speed of computation, GGEMS is not a very generic platform as Geant4 or GATE. For very realistic simulation with lot of information results, Geant4 and GATE are still recommended.

GGEMS features:
* Photon particle tracking
* Multithreaded CPU
* GPU (NVIDIA or Intel HD Graphics)
* Multi devices (GPUs+CPU) approach
* Single or double float precision for dosimetry application
* External X-ray source
* Voxelized source
* Navigation in simple box volume, voxelized volume or meshed volume
* Flat or curved detector for CBCT/CT application
* Visualisation using OpenGL

## Requirements

GGEMS is a multi-architecture application using OpenCL.

OpenCL v3.0 must be installed on your system. This OpenCL version could be downloaded from CUDA Toolkit 12.6.

Supported and tested operating system:

* Windows 11
* Ubuntu 24.04 LTS

Tested compilers:

* GNU Compiler Collection (GCC) Version 13.3 for Linux
* Clang version 18.1.3 for Linux
* Visual C/C++ Compiler Version 19.44 or 19.50 for x64 for Windows

## Installation

GEMS can be install on Linux or Windows system using setuptools. Simply use the following command in the GGEMS directory:

```console
foo@bar~: python setup.py build_ext --opengl=ON install
```

For more details, please read the installation recommendation <https://doc.ggems.fr/v1.3/building_and_installing.html>

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
