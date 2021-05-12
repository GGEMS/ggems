**********
Change log
**********

CMAKE
=====
* Example are now installed in GGEMS install path

C++
===
* Smart pointers removed and replaced by classic 'new' and 'delete' methods.
* For Windows user, options application can be handled by methods defined in GGEMSWinGetOpt.

GGEMS
=====
* New classes GGEMSProfilerManager, GGEMSProfiler and GGEMSProfilerItem can be used to display details about elapsed time in OpenCL kernels.
* GGEMS can be run on multi-devices GPU and/or CPU.
* A new method in C++ and python handles the balance computation between each device.
* In GGEMSOpenCLManager, 'clean' method cleans all GGEMS C++ singletons.
* MHD file suffix is checked at the beginning of simulation.
* New OpenCL kernel 'is_alive' checks if particles are alive after each batch
* Problem reading material file on Linux windows is fixed
* C++ Singleton GGEMSManager is deleted and replaced by GGEMS class
* A security has been added to prevent infinite loop during tracking

Features
========
* For CT application, scatter histogram can be saved.
* New class GGEMSWorld stores data (fluence (photon tracking), energy deposit, energy deposite squared and momentum) outside navigator (phantom and detector).

Examples
========
* New example 5_World_Tracking illustrating new GGEMSWorld feature
