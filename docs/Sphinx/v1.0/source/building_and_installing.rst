*********************
Building & Installing
*********************

.. NOTE::

  GGEMS is written in C++ and using the OpenCL C++ API. However, the most useful GGEMS functions have been wrapped to be called in pure Python version 3. Python is not mandatory, GGEMS can be used in pure C++ too. Lot of C++ and python examples are given in this manual.

Prerequisites
=============
GGEMS code is based on the OpenCL library. For each platform (NVIDIA, Intel, AMD) in your computer, you have to install a specific driver provided by the vendor.

NVIDIA
------
Linux & Windows
~~~~~~~~~~~~~~~
CUDA and NVIDIA driver have to be installed if you want to use GGEMS on a NVIDIA architecture. The easiest way to install OpenCL on NVIDIA platform is to install CUDA and NVIDIA driver in same time from the following link: https://developer.nvidia.com/cuda-downloads.

GGEMS has be tested on the lastest CUDA and NVIDIA versions, and also older versions.

.. WARNING::

  CUDA is not used in GGEMS, but the OpenCL library file in included in CUDA folder.

.. WARNING::

  It is recommanded to install CUDA and the NVIDIA driver directly from the NVIDIA website. Using packaging tool (as apt for instance) is very convenient but can produce some troubles during GGEMS executation, for instance is there is a compatibility problem between CUDA and the NVIDIA driver.

INTEL
-----

AMD
---
Linux & Windows
~~~~~~~~~~~~~~~
AMD platform has not be tested, but surely with few modifications GGEMS will run on a AMD platform. The correct driver for CPU and/or GPU should be available on the following link: https://www.amd.com/en/support. Don't hesitate to contact the GGEMS team if you need help for AMD implementation. For the next releases, AMD platform will be tested and validated.

Unix
====

Windows
=======
