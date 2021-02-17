********************
Make a GGEMS Project
********************

GGEMS is designed as a library, and can be called using either python or C++. The performance are the same, it depends only which language the user is familiar. In this part, only GGEMS commands shared between all examples are explained. For more explanations for other commands please read example documentation.

.. NOTE::

  On Windows or Unix operating system all the following steps are the same.

Template
========

GGEMS (C++ or python) macros as the same following template:

.. image:: ../images/template.png
  :width: 700
  :align: center

Python
======

Before using python and GGEMS, check GGEMS 'python_module' is in PYTHONPATH variale. PYTHONPATH has to point to the GGEMS library too.

Using GGEMS with python is very simple. A folder storing the project should be created. Inside this folder, write a python file importing GGEMS.

.. code-block:: python

  from ggems import *

The verbosity level is defined in the range [0;3]. For a silent GGEMS execution, the level has to be set to 0, otherwise 3 for maximum information.

.. code-block:: python

  GGEMSVerbosity(0)

Next step, an OpenCL device has to be selected. Device is set by its index. If the user wants the first found device by OpenCL so index is 0

.. code-block:: python

  opencl_manager.set_device_index(0)

Then a material database has to be loaded in GGEMS. The material file provided by GGEMS is in 'data' folder. This file can be copy and paste in your project, and a new material can be added.

.. code-block:: python

  materials_database_manager.set_materials('materials.txt')

The physical tables can be customized by changing the number of bins and the energy range. The following values are the default values.

.. code-block:: python

  processes_manager.set_cross_section_table_number_of_bins(220)
  processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
  processes_manager.set_cross_section_table_energy_max(10.0, 'MeV')

The photon physical processes are selecting using the process name, the concerning particle and the associated phantom (or 'all' for all defined phantoms).

.. code-block:: python

  processes_manager.add_process('Compton', 'gamma', 'all')
  processes_manager.add_process('Photoelectric', 'gamma', 'all')
  processes_manager.add_process('Rayleigh', 'gamma', 'all')

In GGEMS, range cuts are defined in distance, particle type has to be specified and cuts are associated to a phantom (or 'all' for all defined phantoms). The distance is converted in energy during the initialization step. During the particle tracking, if the energy particle is inferior to the cut, then the particle is killed and the energy is stored.

.. code-block:: python

  range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')

GGEMS C++ singleton is called in python with 'ggems_manager' variable. All verboses can be set to 'True' or 'False' depending on the amount of details the user needs. In 'tracking_verbose', the second parameters in the index of particle to track. All objects in GGEMS are initialized with the method 'initialize'. The GGEMS simulations starts with the method 'run'.

.. code-block:: python

  ggems_manager.opencl_verbose(True)
  ggems_manager.material_database_verbose(True)
  ggems_manager.navigator_verbose(True)
  ggems_manager.source_verbose(True)
  ggems_manager.memory_verbose(True)
  ggems_manager.process_verbose(True)
  ggems_manager.range_cuts_verbose(True)
  ggems_manager.random_verbose(True)
  ggems_manager.kernel_verbose(True)
  ggems_manager.tracking_verbose(True, 10)

  ggems_manager.initialize()
  ggems_manager.run()

The last step, exit GGEMS properly by cleaning OpenCL C++ singleton

.. code-block:: python

  opencl_manager.clean()

C++
===

Building a project from scratch using GGEMS library in C++ is a little more difficult. A small example is given using CMake.

First create your project folder (named 'my_project'), then inside it 'include' and 'src' folder can be created if your own classes are compiled with the GGEMS library. A file named 'main.cc' is created for this example and 'CMakeLists.txt' file is also created. At this stage, the folder structure is:

.. code-block:: text

  <my_project>
  |-- include\
  |-- src\
  |-- main.cc
  |-- CMakeLists.txt

Compiling this project can be done using the following 'CMakeLists.txt' example:

.. code-block:: cmake

  CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

  SET(ENV{CC} "clang")
  SET(ENV{CXX} "clang++")

  PROJECT(MYPROJECT LANGUAGES CXX)

  FIND_PACKAGE(OpenCL REQUIRED)

  SET(GGEMS_INCLUDE_DIRS "" CACHE PATH "Path to the GGEMS include directory")
  SET(GGEMS_LIBRARY "" CACHE FILEPATH "GGEMS library")

  INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include ${GGEMS_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(SYSTEM ${OpenCL_INCLUDE_DIRS})

  LINK_DIRECTORIES(${OpenCL_LIBRARY})

  FILE(GLOB source ${PROJECT_SOURCE_DIR}/src/*.cc)

  ADD_EXECUTABLE(my_project main.cc ${source})
  TARGET_LINK_LIBRARIES(my_project ${OpenCL_LIBRARY} ${GGEMS_LIBRARY})

All previous python commands can be written in C++.

including some GGEMS files:

.. code-block:: c++

  #include "GGEMS/global/GGEMSOpenCLManager.hh"
  #include "GGEMS/global/GGEMSManager.hh"
  #include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"
  #include "GGEMS/physics/GGEMSRangeCutsManager.hh"
  #include "GGEMS/physics/GGEMSProcessesManager.hh"

The verbosity level is defined in the range [0;3]. For a silent GGEMS execution, the level has to be set to 0, otherwise 3 for maximum information.

.. code-block:: c++

  GGcout.SetVerbosity(0);
  GGcerr.SetVerbosity(0);
  GGwarn.SetVerbosity(0);

Next step, an OpenCL device has to be selected. Device is set by its index. If the user wants the first found device by OpenCL so index is 0

.. code-block:: c++

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  opencl_manager.DeviceToActivate(0);

Then a material database has to be loaded in GGEMS. The material file provided by GGEMS is in 'data' folder. This file can be copy and paste in your project, and a new material can be added.

.. code-block:: c++

  GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  material_manager.SetMaterialsDatabase("materials.txt");

The physical tables can be customized by changing the number of bins and the energy range. The following values are the default values.

.. code-block:: c++

  GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
  processes_manager.SetCrossSectionTableNumberOfBins(220);
  processes_manager.SetCrossSectionTableMinimumEnergy(1.0f, "keV");
  processes_manager.SetCrossSectionTableMaximumEnergy(1.0f, "MeV");

The photon physical processes are selecting using the process name, the concerning particle and the associated phantom (or 'all' for all defined phantoms).

.. code-block:: c++

  processes_manager.AddProcess("Compton", "gamma", "all");
  processes_manager.AddProcess("Photoelectric", "gamma", "all");
  processes_manager.AddProcess("Rayleigh", "gamma", "all");

In GGEMS, range cuts are defined in distance, particle type has to be specified and cuts are associated to a phantom (or 'all' for all defined phantoms). The distance is converted in energy during the initialization step. During the particle tracking, if the energy particle is inferior to the cut, then the particle is killed and the energy is stored.

.. code-block:: c++

  GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();
  range_cuts_manager.SetLengthCut("all", "gamma", 0.1f, "mm");

GGEMS C++ singleton is called with 'ggems_manager' variable. All verboses can be set to 'True' or 'False' depending on the amount of details the user needs. In 'tracking_verbose', the second parameters in the index of particle to track. All objects in GGEMS are initialized with the method 'initialize'. The GGEMS simulations starts with the method 'run'.

.. code-block:: c++

  GGEMSManager& ggems_manager = GGEMSManager::GetInstance();
  ggems_manager.SetOpenCLVerbose(true);
  ggems_manager.SetNavigatorVerbose(true);
  ggems_manager.SetSourceVerbose(true);
  ggems_manager.SetMemoryRAMVerbose(true);
  ggems_manager.SetProcessVerbose(true);
  ggems_manager.SetRangeCutsVerbose(true);
  ggems_manager.SetRandomVerbose(true);
  ggems_manager.SetKernelVerbose(true);
  ggems_manager.SetTrackingVerbose(true, 10);

  ggems_manager.Initialize();
  ggems_manager.Run();

The last step, exit GGEMS properly by cleaning OpenCL C++ singleton

.. code-block:: c++

  opencl_manager.Clean();
