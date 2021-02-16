***************
Getting Started
***************
Once the installation is finished, the easiest way trying GGEMS is using the python console and call the OpenCL C++ singleton.

.. code-block:: console

  $ python
  Python 3.9.1 (tags/v3.9.1:1e5d33e, Dec  7 2020, 17:08:21) [MSC v.1927 64 bit (AMD64)] on win32
  Type "help", "copyright", "credits" or "license" for more information
  >>> from ggems import *
  >>> opencl_manager.print_infos()
  >>> opencl_manager.set_device_index(0)
  >>> opencl_manager.clean()
  >>> exit()

With the previous command lines, the user has the possibility checking which device is recognized by GGEMS and the first found OpenCL device is selected. 

.. IMPORTANT::

  If an OpenCL device is missing, please check your installation driver for the missing device.

The best way learing how GGEMS is working, is to try each example available in the examples folder. Each examples is detailed for the user in the respective part of this manual. Using GGEMS, for a personnal project, from scratch, using python or C++ is explained in the developer part.