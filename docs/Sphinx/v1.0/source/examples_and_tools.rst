****************
Examples & Tools
****************

A list of examples and tools are provided for GGEMS users. Each time, when it is possible, C++ and python macros are given. For C++ macros, a CMakeLists.txt file is mandatory for compilation.

.. NOTE::

  Examples are compiled and installed when the compilation option 'BUILD_EXAMPLES' is set to ON. C++ macro executables are installed in the same location than example folders.


All provided examples are explained in details in the following parts. Only python macros are explained. C++ macros are very similar and not need more explanations.

Example 0: Cross section computation
====================================

The purpose of this example is to provide a tool computing cross section for a specific material and a specific photon physical process. The energy (in MeV) and the OpenCL device have to be set by the user.

.. code-block:: console

  $ python cross_sections.py [-h] [-d DEVICE] [-m MATERIAL] -p [PROCESS]-e [ENERGY]
  -h/--help           Printing help into the screen
  -d/--device         Setting OpenCL id
  -m/--material       Setting one of material defined in GGEMS (Water, Air, ...)
  -p/--process        Setting photon physical process (Compton, Rayleigh, Photoelectric)
  -e/--energy         Setting photon energy in MeV

The macro is defined in the file 'cross_section.py'. The most important lines are explained there.

The verbosity level is defined in the range [0;3]. For a silent GGEMS execution, the level has to be set to 0, otherwise 3 for maximum information.

.. code-block:: python

  GGEMSVerbosity(0)
  opencl_manager.set_device_index(device_id)

A GGEMSMaterial is created, and each new material can be selected. The initialization step is mandatory and compute all physical tables, and store them on an OpenCL device.

.. code-block:: python

  materials = GGEMSMaterials()
  materials.add_material(material_name)
  materials.initialize()

Before using a physical process, GGEMSCrossSection object has to be created. Then each process can be added individually. And finally the cross sections are computing by giving the list of materials.

.. code-block:: python

  cross_sections = GGEMSCrossSections()
  cross_sections.add_process(process_name, 'gamma')
  cross_sections.initialize(materials)

Getting the cross section value (in cm2.g-1) for a specific energy (in MeV) is done by the following command:

.. code-block:: python

  cross_sections.get_cs(process_name, material_name, energy_MeV, 'MeV')

And finally to exit the code properly

.. code-block:: python

  opencl_manager.clean()
  exit()