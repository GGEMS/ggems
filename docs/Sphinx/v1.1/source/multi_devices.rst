************
Multi-Device
************

GGEMS can be used on multi-devices using OpenCL library. Two different ways are implemented to activate device:

* Device can be selected using device index

.. code-block:: python

  opencl_manager = GGEMSOpenCLManager()
  opencl_manager.set_device_index(0) # Activate device id 0
  opencl_manager.set_device_index(2) # Activate device id 2, if it exists

* Device can be selected using a string for a list of devices

.. code-block:: python

  opencl_manager = GGEMSOpenCLManager()
  opencl_manager.set_device_to_activate('gpu', 'nvidia') # Activate all NVIDIA GPU only
  opencl_manager.set_device_to_activate('gpu', 'intel') # Activate all Intel GPU only
  opencl_manager.set_device_to_activate('gpu', 'amd') # Activate all AMD GPU only
  opencl_manager.set_device_to_activate('all') # Activate all found devices
  opencl_manager.set_device_to_activate('0;2') # Activate devices 0 and 2
