.. GGEMS documentation master file, created by
   sphinx-quickstart on Tue Oct 13 10:17:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GGEMS Documentation
==============================
GGEMS is an advanced Monte Carlo simulation platform using CPU and GPU architecture targeting medical applications (imaging and particle therapy). This code is based on the well-validated Geant4 physics model and capable to be executed in both CPU and GPU devices using the OpenCL library.

The documentation is divided into three parts:

* Preamble:
* User documentation:
* Developer documentation:

.. toctree::
   :maxdepth: 2
   :caption: Preamble

   requirements
   introduction
   building_and_installing
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   capabilities
   running_ggems
   benchmarks
   examples_and_tools

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   fundamentals
   executable_program
   python_interface
   changelog
