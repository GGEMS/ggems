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

The macro is defined in the file 'cross_section.py'. The most important line are explained for the user.

.. code-block:: python

  GGEMSVerbosity(0)
  opencl_manager.set_device_index(device_id)

The verbosity level is defined in the range [0;3]. For a silent GGEMS execution, the level has to be set to 0, otherwise 3 for maximum information.

.. code-block:: python

  materials_database_manager.set_materials('../../data/materials.txt')

  materials = GGEMSMaterials()
  materials.add_material(material_name)
  materials.initialize()

In GGEMS, all materials have to be loaded at the beginning of the execution. All materials are defined in 'data/materials.txt' in the GGEMS source folder. A new material can be defined by the user.

Once all materials are loaded, a GGEMSMaterial is created, and each new necessary material can be added. The initialization step is mandatory and computate all physical tables, and store them on OpenCL device.