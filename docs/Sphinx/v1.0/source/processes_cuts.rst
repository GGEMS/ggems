*******************************
Physical Processes & Range cuts
*******************************

Physical Processes
==================

The photon processes impletemented are:

  * Compton scattering
  * Photoelectric effect
  * Rayleigh scattering

Each of these processes are extracted from Geant4 version 10.6. For more information about physics, please read the documentation on the Geant4 website.

By using python, the variable 'processes_manager' can be called to manage processes.

.. IMPORTANT::

  Secondary particles (photon and electron) are not simulated yet. For Photoelectric effect, the photon is killed during the interaction and the energy is locally deposited, and the fluorescence photon is not emitted.

Compton Scattering
------------------

The Geant4 model extracted is the 'G4KleinNishinaCompton' standard model. It is the fastest algorithm to simulate this process. Compton scattering is activated for all the navigators, or for a specific navigator.

.. code-block:: python

  processes_manager.add_process('Compton', 'gamma', 'all')

In the previous line, Compton scattering is activated for all the navigators.

.. code-block:: python

  processes_manager.add_process('Compton', 'gamma', 'my_phantom')

In the previous line, Compton scattering is activated only for a navigator named 'my_phantom'.

Photoelectric Effect
--------------------

The Geant4 model extracted is the 'G4PhotoElectricEffect' standard model using Sandia tables. Photoelectric effect is activated for all the navigators, or for a specific navigator.

.. code-block:: python

  processes_manager.add_process('Photoelectric', 'gamma', 'all')

In the previous line, Photoelectric effect is activated for all the navigators.

.. code-block:: python

  processes_manager.add_process('Photoelectric', 'gamma', 'my_phantom')

In the previous line, Photoelectric effect is activated only for a navigator named 'my_phantom'

Rayleigh Scattering
-------------------

The Geant4 model extracted is the 'G4LivermoreRayleighModel' livermore model. Rayleigh scattering is activated for all the navigators, or for a specific navigator.

.. code-block:: python

  processes_manager.add_process('Rayleigh', 'gamma', 'all')

In the previous line, Rayleigh scattering is activated for all the navigators.

.. code-block:: python

  processes_manager.add_process('Rayleigh', 'gamma', 'my_phantom')

In the previous line, Rayleigh scattering is activated only for a navigator named 'my_phantom'

Process Parameters Building
---------------------------

The cross-sections are computed during the GGEMS initialization step. The parameters used for the cross-sections building can be customized by the user, however it is recommanded to use the default parameters. The customizable parameters are:

  * Minimum energy of cross-section table
  * Maximum energy of cross-section table
  * Number of bins in cross-section table

The default parameters are defined as following:

.. code-block:: python

  processes_manager.set_cross_section_table_number_of_bins(220)
  processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
  processes_manager.set_cross_section_table_energy_max(1.0, 'MeV')

Process Verbosity
-----------------

Informations about processes can be printed by GGEMS:

  * Available processes
  * Global informations about processes
  * Cross-section value in tables

The list of commands are:

.. code-block:: python

  processes_manager.print_available_processes()
  processes_manager.print_infos()
  processes_manager.print_tables(True)

Range Cuts
==========

The cuts are defined for each particle in distance unit in all navigator or a specific navigator. During the GGEMS initialization the cuts are converted in energy for each defined material in navigator. If the particle energy is below the cut, then the particle is killed and the energy locally deposited. By default the cuts are 1 micron.

.. code-block:: python

  range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')

In the previous line, cuts are activated for photon for all navigators.

.. code-block:: python

  range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'my_phantom')

In the previous line, cuts are activated for photon for a navigator named 'my_phantom'.
