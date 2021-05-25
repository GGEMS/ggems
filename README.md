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

## Requirements

GGEMS is a multiplatform application using OpenCL.

OpenCL v1.2 has to be installed on your system.

Supported operating system:

* Windows
* Linux

Tested compilers:

* gcc 7/8/9 on Linux
* clang v9/10/11 on Linux and Windows
* Visual C++ 2019 on Windows

## Installation

To install GGEMS, please follow the procedure here: <https://doc.ggems.fr/v1.1/building_and_installing.html>

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