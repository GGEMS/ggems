*************************************
Examples 0: Cross-Section Computation
*************************************

The purpose of this example is to provide a tool computing cross-section for a specific material and a specific photon physical process. The energy (in MeV) and the OpenCL device are set by the user.

.. code-block:: console

  $ python cross_sections.py [-h] [-d DEVICE] [-m MATERIAL] -p [PROCESS]-e [ENERGY] [-v VERBOSE]
  -h/--help           Printing help into the screen
  -d/--device         Setting OpenCL id
  -m/--material       Setting one of material defined in GGEMS (Water, Air, ...)
  -p/--process        Setting photon physical process (Compton, Rayleigh, Photoelectric)
  -e/--energy         Setting photon energy in MeV
  -v/--verbose        Setting level of verbosity

The macro is in the file 'cross_section.py'.

Verbosity level is defined in the range [0;3]. For a silent GGEMS execution, the level is set to 0, otherwise 3 for lot of informations.

.. code-block:: python

  GGEMSVerbosity(verbosity_level)
  opencl_manager.set_device_index(device_id)

GGEMSMaterial object is created, and each new material can be added. The initialization step is mandatory and compute all physical tables, and store them on an OpenCL device.

.. code-block:: python

  materials = GGEMSMaterials()
  materials.add_material(material_name)
  materials.initialize()

Before using a physical process, GGEMSCrossSection object is created. Then each process can be added individually. And finally cross sections are computing by giving the list of materials.

.. code-block:: python

  cross_sections = GGEMSCrossSections()
  cross_sections.add_process(process_name, 'gamma')
  cross_sections.initialize(materials)

Getting the cross section value (in cm2.g-1) for a specific energy (in MeV) is done by the following command:

.. code-block:: python

  cross_sections.get_cs(process_name, material_name, energy_MeV, 'MeV')