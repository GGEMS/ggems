***************
Getting Started
***************
Once the installation is finished, the easiest way to try GGEMS is to use the python console and call the OpenCL C++ singleton.

.. code-block:: console

  $ python
  Python 3.9.1 (tags/v3.9.1:1e5d33e, Dec  7 2020, 17:08:21) [MSC v.1927 64 bit (AMD64)] on win32
  Type "help", "copyright", "credits" or "license" for more information
  >>> from ggems import *
  >>> opencl_manager.print_infos()
  >>> opencl_manager.clean()
  >>> exit()

With the previous command lines, the user can check which device is recognized by GGEMS, and all the device caracteristics are listed.

.. IMPORTANT::

  If an OpenCL device is missing, please check your installation driver for the missing device.

The best way to learn how GGEMS is working, is to try each example available in the examples folder.