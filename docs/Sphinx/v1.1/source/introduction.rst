************
Introduction
************

GGEMS (GPU Geant4-based Monte Carlo Simulations) is an advanced Monte Carlo simulation platform using the OpenCL library managing CPU and GPU architecture. GGEMS is written in C++, and can be used using python commands. The reader is assumed to have some basic knowledge of object-oriented programming using C++.

Well-validated `Geant4 <https://geant4.web.cern.ch>`_ physic models are used in GGEMS and implemented using OpenCL.

The aim of GGEMS is to provide a fast simulation platform for imaging application and particle therapy. To favor speed of computation, GGEMS is not a very generic platform as `Geant4 <https://geant4.web.cern.ch>`_ or `GATE <http://www.opengatecollaboration.org/>`_. For very realistic simulation with lot of information results, Geant4 and GATE are still recommended.

GGEMS features:

* Photon particle tracking
* Multithreaded CPU
* GPU
* Multi devices (GPUs+CPU) approach
* Single or double float precision for dosimetry application
* External X-ray source
* Navigation in simple box volume or voxelized volume
* Flat or curved detector for CBCT/CT application

GGEMS medical applications:

* CT/CBCT imaging (standard, dual-energy)
* External radiotherapy (IMRT and VMAT)
* Portal imaging from LINAC system

In the next GGEMS releases, the aim is to implement the following applications and features:

* Visualization
* Positron particle tracking
* Electron particle tracking
* Mesh volume
* Voxelized source
* PET imaging
* SPECT imaging
* Intra-operative radiotherapy (brachytherapy and intrabeam)
* AMD architecture validation
* MacOS system validation